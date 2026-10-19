"""CPU-only architecture/forward check; no dataset or training."""
from pathlib import Path
import sys
import tempfile
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from PIL import Image
    from qrphishing.model import ModelBundle
    torch.set_num_threads(1)
    bundle = ModelBundle(ROOT / "models/selected_three_systems.pt", ROOT / "models/tokenizer", "cpu")
    with tempfile.TemporaryDirectory() as directory:
        image = Path(directory) / "synthetic_white.png"
        Image.new("RGB", (224, 224), "white").save(image)
        missing = bundle.predict(image_path=image, mode="all")
        assert missing["fusion"]["route"] == "visual_fallback"
        assert missing["fusion"]["malicious_probability"] == missing["visual"]["malicious_probability"]
        available = bundle.predict(image_path=image, url="https://example.com/", mode="all")
        assert available["fusion"]["route"] == "learned_features"
        for key in ["visual", "text", "fusion", "text_with_visual_fallback"]:
            assert 0 <= available[key]["malicious_probability"] <= 1
        assert available["text"]["malicious_probability"] == available["text_with_visual_fallback"]["malicious_probability"]
    print(json.dumps({"strict_load": True, "synthetic_forward": True, "missing_url_fallback": True,
                      "real_image_test": False}))


if __name__ == "__main__":
    main()
