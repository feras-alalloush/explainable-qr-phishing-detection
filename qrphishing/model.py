"""Exact architecture of the selected three-system export. No downloads."""
from pathlib import Path
import hashlib
import json

import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms as T
from transformers import BertConfig, BertModel, AutoTokenizer

from .io import verify_receipt, sha256_file


def make_head(input_size):
    return nn.Sequential(
        nn.Linear(input_size, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Linear(128, 2),
    )


class VisualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_backbone = torchvision.models.mobilenet_v3_small(weights=None)
        self.visual_backbone.classifier = nn.Identity()
        self.visual_proj = nn.Sequential(
            nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = make_head(256)

    def features(self, images):
        return self.visual_proj(self.visual_backbone(images))


class TextModel(nn.Module):
    def __init__(self, source):
        super().__init__()
        config = BertConfig.from_dict(source["bert_config"])
        config._attn_implementation = "eager"
        if (config.hidden_size, config.num_hidden_layers, config.num_attention_heads) != (128, 2, 2):
            raise ValueError("Unexpected BERT-Tiny configuration")
        self.text_backbone = BertModel(config, add_pooling_layer=source["has_pooler"])
        self.text_proj = nn.Sequential(
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.fusion = make_head(256)

    def features(self, tokens):
        cls = self.text_backbone(**tokens, return_dict=True).last_hidden_state[:, 0]
        return self.text_proj(cls)


class ThreeSystems(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.visual = VisualModel()
        self.text = TextModel(source)
        self.fusion = make_head(512)

    def forward(self, images, tokens):
        visual = self.visual.features(images)
        text = self.text.features(tokens)
        return (
            self.visual.fusion(visual), self.text.fusion(text),
            self.fusion(torch.cat([visual, text], dim=1)),
        )


TRANSFORM = T.Compose([
    T.Resize((224, 224)), T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class ModelBundle:
    def __init__(self, checkpoint_path, tokenizer_path, device="cpu"):
        self.path = Path(checkpoint_path)
        verify_receipt(self.path)
        # The receipt verifies bytes; load only the owner's trusted training export.
        self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=False)
        source = self.checkpoint["source_model_identity"]
        self.device = torch.device(device)
        self.model = ThreeSystems(source)
        self.model.load_state_dict(self.checkpoint["model_state"], strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.to(self.device).eval()
        tokenizer_path = Path(tokenizer_path)
        for name, digest in source.get("tokenizer_files_sha256", {}).items():
            if sha256_file(tokenizer_path / name) != digest:
                raise ValueError("Tokenizer checksum mismatch: " + name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, use_fast=True
        )
        fingerprint = hashlib.sha256(json.dumps(
            self.tokenizer.get_vocab(), sort_keys=True, separators=(",", ":")
        ).encode()).hexdigest()
        if fingerprint != source["vocab_sha256"]:
            raise ValueError("Tokenizer vocabulary mismatch")

    def tokens(self, urls):
        tokens = self.tokenizer(
            urls, padding="max_length", truncation=True,
            max_length=128, return_tensors="pt",
        )
        return {name: value.to(self.device) for name, value in tokens.items()}

    @staticmethod
    def decision(probability):
        probability = float(probability)
        return {"label": "malicious" if probability >= 0.5 else "benign",
                "malicious_probability": probability}

    def predict(self, image_path=None, url=None, mode="all"):
        if mode not in {"all", "visual", "text", "fusion"}:
            raise ValueError("Unknown mode")
        available = bool(url and url.strip())
        needs_image = mode in {"all", "visual", "fusion"}
        if needs_image and image_path is None:
            raise ValueError("An image is required for this mode")
        if mode == "text" and not available:
            raise ValueError("Text-only inference requires a decoded URL")
        result = {"url_available": available, "threshold": 0.5}
        with torch.inference_mode():
            visual_features = None
            if needs_image:
                with Image.open(image_path) as image:
                    batch = TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
                visual_features = self.model.visual.features(batch)
                visual_p = self.model.visual.fusion(visual_features).softmax(1)[0, 1]
                if mode in {"all", "visual"}:
                    result["visual"] = self.decision(visual_p.cpu())
            if available and mode in {"all", "text", "fusion"}:
                text_features = self.model.text.features(self.tokens([url]))
                text_p = self.model.text.fusion(text_features).softmax(1)[0, 1]
                if mode in {"all", "text"}:
                    result["text"] = self.decision(text_p.cpu())
                if mode in {"all", "fusion"}:
                    fusion_p = self.model.fusion(torch.cat([
                        visual_features, text_features
                    ], dim=1)).softmax(1)[0, 1]
                    result["fusion"] = {**self.decision(fusion_p.cpu()), "route": "learned_features"}
                    if mode == "all":
                        result["text_with_visual_fallback"] = self.decision(text_p.cpu())
            elif mode in {"all", "fusion"}:
                result["fusion"] = {**self.decision(visual_p.cpu()), "route": "visual_fallback"}
                if mode == "all":
                    result["text"] = {"label": None, "reason": "No decoded URL supplied"}
                    result["text_with_visual_fallback"] = self.decision(visual_p.cpu())
        return result
