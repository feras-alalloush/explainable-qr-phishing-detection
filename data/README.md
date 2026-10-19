# External dataset and fixed membership

Images and the million-row decoded-URL table are not included. Attach/download the same original Kaggle inputs used by the author:

- Image dataset: `feras2004/qr-all-benign`; benign `QR_All_benign/qrs`, malicious `QR_All_Malicious/qrs`.
- Original CSV: `feras2004/spliteoriginal/splitsOriginal.csv`.

The exact bytes of the original source CSV are required; the collector records its SHA256. It also exports `retained_source_row_indices.npy`: zero-based original CSV row positions in the exact retained manifest order. This gives reviewers fixed membership without republishing all raw URLs. Reconstruct it with:

```bash
python scripts/reconstruct_membership.py --original-csv /path/splitsOriginal.csv --output /path/retained_membership.csv
```

The input CSV is supplied externally. Reconstructed rows retain original labels/URL/domain/fold/split; image hashes and readability must be re-audited using notebook00. Do not infer labels from filenames (some malicious-directory images are named benign_*). Do not recalculate hostname groups or reassign folds.

Original1005738 rows; retained1005700 after removing38 conflicting byte-duplicate rows. Train710746, Validation148529, Test146425. Recorded image-byte-hash/domain/successfully-decoded-URL cross-split intersections are0. Signature:
`d4509181f834a16144ad45e24b8ce686a4e608e07a9846a880757c705739d1ff`.

The signature binds original paths as well as labels/splits/URLs/hash values. Rebinding to another filesystem requires a documented path mapping, not silently replacing the original signature. Supply exact Kaggle dataset version/access details when publishing the repository; this template does not invent a dataset release/version ID.

## Dataset access

- QR images: [/kaggle/input/datasets/feras2004/qr-all-benign]
- Original split CSV: [/kaggle/input/datasets/feras2004/spliteoriginal]


Required source CSV filename: splitsOriginal.csv

Expected source CSV SHA256:
4528f7378e5f99c9ae573fb01436eeb3de4f7b5ae3079c1c9b9d3e0cfc30dd35

The original source CSV is required to reconstruct the exact retained
membership. A newly generated or modified CSV must not be substituted.
