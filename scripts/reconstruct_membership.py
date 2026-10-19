"""Recover exact retained CSV rows without publishing a million raw URLs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    from qrphishing.io import sha256_file
    record = json.loads((ROOT / "data/membership_record.json").read_text())
    if sha256_file(args.original_csv) != record["source_csv_sha256"]:
        raise ValueError("Original CSV differs from the recorded source")
    index_file = ROOT / "data/retained_source_row_indices.npy"
    if sha256_file(index_file) != record["indices_sha256"]:
        raise ValueError("Retained row index checksum mismatch")
    indices = np.load(index_file, allow_pickle=False)
    raw = pd.read_csv(args.original_csv, keep_default_na=False)
    if len(raw) != record["original_rows"]:
        raise ValueError("Original row count differs")
    selected = raw.iloc[indices.astype(np.int64)].copy()
    if len(selected) != record["retained_rows"] or not selected.path.is_unique:
        raise ValueError("Invalid retained membership")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    selected.to_csv(output, index=False)
    print("Wrote", len(selected), "rows. Image hashes/readability are audited separately in notebook00.")


if __name__ == "__main__":
    main()
