"""Verify every file enumerated in MANIFEST.json (stdlib only)."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    failed = []
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if root not in path.parents or not path.is_file():
            failed.append(entry["path"] + ": missing or unsafe path")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"] or path.stat().st_size != entry["bytes"]:
            failed.append(entry["path"] + ": bytes differ")
    if failed:
        raise RuntimeError("\n".join(failed))
    print("Verified", len(manifest["files"]), "files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    verify(parser.parse_args().root)
