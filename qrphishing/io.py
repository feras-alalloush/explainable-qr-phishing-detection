"""Small integrity helpers; no ML dependencies."""
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_receipt(path):
    path = Path(path)
    receipt = read_json(path.with_suffix(".json"))
    if receipt["file"] != path.name:
        raise ValueError("Checkpoint filename differs from its receipt")
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    return receipt


def safe_relative_path(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("Path escapes the package root")
    return target
