"""Grad-CAM for the image contribution of a fixed visual or fusion decision."""
import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--url")
    parser.add_argument("--mode", choices=["visual", "fusion"], default="fusion")
    parser.add_argument("--checkpoint", default="models/selected_three_systems.pt")
    parser.add_argument("--tokenizer", default="models/tokenizer")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target", type=int, choices=[0, 1])
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    import numpy as np
    import torch
    import matplotlib.pyplot as plt
    from PIL import Image
    from torchvision import transforms as T
    from qrphishing.model import ModelBundle, TRANSFORM
    from qrphishing.io import sha256_file
    torch.set_num_threads(1)
    bundle = ModelBundle(args.checkpoint, args.tokenizer, args.device)
    output = Path(args.output_dir or (
        "explanations/gradcam_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    ))
    output.mkdir(parents=True, exist_ok=False)
    with Image.open(args.image) as image:
        source = image.convert("RGB")
        input_tensor = TRANSFORM(source)
        display = T.Resize((224, 224))(source)
    captured = {}
    def capture(module, inputs, activation):
        captured["activation"] = activation
    hook = bundle.model.visual.visual_backbone.features[-1].register_forward_hook(capture)
    try:
        with torch.enable_grad():
            x = input_tensor.unsqueeze(0).to(bundle.device).requires_grad_(True)
            features = bundle.model.visual.features(x)
            route = "visual"
            if args.mode == "fusion" and args.url and args.url.strip():
                text = bundle.model.text.features(bundle.tokens([args.url]))
                logits = bundle.model.fusion(torch.cat([features, text], dim=1))
                route = "learned_features"
            else:
                logits = bundle.model.visual.fusion(features)
                if args.mode == "fusion":
                    route = "visual_fallback"
            target = args.target if args.target is not None else int(logits.argmax(1))
            probability = float(logits.detach().softmax(1)[0, 1].cpu())
            activation = captured["activation"]
            gradient = torch.autograd.grad(logits[0, target], activation)[0]
            cam = (gradient.mean((2, 3), keepdim=True) * activation).sum(1, keepdim=True).relu()
            feature_shape = list(activation.shape[-2:])
            cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
            cam = cam[0, 0].detach().cpu().numpy()
    finally:
        hook.remove()
        captured.clear()
    if not np.isfinite(cam).all():
        raise ValueError("Nonfinite Grad-CAM")
    zero_map = bool(cam.max() <= 1e-12)
    if zero_map:
        cam = np.zeros_like(cam)
    else:
        cam = cam / cam.max()
    original = np.asarray(display).astype(np.float32) / 255
    overlay = original if zero_map else np.clip(0.6 * original + 0.4 * plt.get_cmap("jet")(cam)[..., :3], 0, 1)
    display.save(output / "original.png")
    np.save(output / "cam.npy", cam)
    plt.imsave(output / "overlay.png", overlay)
    plt.imsave(output / "heatmap.png", cam, cmap="jet", vmin=0, vmax=1)
    report = dict(
        checkpoint_sha256=sha256_file(args.checkpoint), mode=args.mode, route=route,
        target_class=target, malicious_probability=probability, feature_shape=feature_shape,
        zero_map=zero_map,
        interpretation="Image contribution to a fixed class logit; URL held fixed. Not token attribution or causal proof.",
    )
    (output / "explanation.json").write_text(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    main()
