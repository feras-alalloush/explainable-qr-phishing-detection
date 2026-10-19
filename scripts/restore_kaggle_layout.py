"""Restore packaged weights to the original notebook paths without overwriting."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", default="/kaggle/working")
    args = parser.parse_args()
    from qrphishing.io import sha256_file
    working = Path(args.working)
    working.mkdir(parents=True, exist_ok=True)
    mapping = {
        "selected_three_systems.pt": "qr_three_systems_random_fusion_v1/selected_three_systems_epoch_01_b06b88974bbe.pt",
        "source_visual.pt": "mobilenet_continue/robust_aug_phase2_v1/best_epoch_02_796c6ee005f6.pt",
        "source_text.pt": "runs/text_full_seed42.pt",
        "legacy_visual_200k.pt": "runs/abl_visual_n200000_seed42.pt",
        "visual_phase1.pt": "mobilenet_continue/full_train_phase1_fp32/best_epoch_04_10a6296afd7d.pt",
    }
    for name, relative in mapping.items():
        source, target = ROOT / "models" / name, working / relative
        if not source.is_file():
            print("Optional source absent:", name)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(source) != sha256_file(target):
                raise ValueError("Existing target differs; left untouched: " + str(target))
        else:
            shutil.copy2(source, target)
        receipt_path = target.with_suffix(".json")
        if not receipt_path.exists():
            receipt_path.write_text(json.dumps({"file": target.name, "sha256": sha256_file(target)}, indent=2))
        print("Verified:", target)
    destination = working / "qr_three_systems_random_fusion_v1/tokenizer"
    destination.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "models/tokenizer").iterdir():
        if not source.is_file():
            continue
        target = destination / source.name
        if target.exists():
            if sha256_file(source) != sha256_file(target):
                raise ValueError("Existing tokenizer differs; left untouched")
        else:
            shutil.copy2(source, target)
    print("Attach original dataset/CSV and run notebook00 to recreate the audited manifest.")


if __name__ == "__main__":
    main()
