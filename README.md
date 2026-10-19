# Explainable QR-code phishing detection

Research artifacts comparing **visual-only MobileNetV3-Small**, **URL-only BERT-Tiny**, and **learned concatenation of their frozen features**, with an explicit visual fallback when the URL is missing. A text-with-visual-fallback baseline is reported separately.

## Package status

This completed package includes the selected model weights, tokenizer,
training notebooks, evaluation code, final Test reports, and fixed split
membership indices.

Dataset images and the original decoded-URL CSV are supplied separately
through Kaggle. See data/README.md for access and reconstruction instructions.

File integrity and saved prediction metrics were independently checked.
The package collector also performed a CPU synthetic forward check.
These checks do not constitute a new evaluation on the original Test images.

## Quick start (completed package)

Use Python3.12 as a starting point and consult `environment.recorded.json` / `requirements-recorded.txt` for the actual Kaggle environment. `requirements.txt` is a dependency inventory, not a tested lockfile. Install a compatible PyTorch/torchvision CPU or CUDA pair using official PyTorch instructions, then the remaining dependencies. Recorded CUDA build suffixes may need the corresponding PyTorch package index.

```bash
python -m pip install -r requirements.txt
python scripts/verify_package.py
python scripts/smoke_model.py
python -m qrphishing --image /path/to/qr.png --url 'https://example.com/' --mode all
python -m qrphishing --image /path/to/qr.png --mode visual
python -m qrphishing --url 'https://example.com/' --mode text
python -m qrphishing --image /path/to/qr.png --mode fusion
```

No URL is visited. URL-only CLI inference requires URL text. Without a URL, fusion routes to the visual classifier and the CLI reports text as unavailable. Benchmark tables also report the legacy text classifier's response to the saved placeholder URL, explicitly labelled as missing-URL results.

Explain the image contribution of a fixed fusion decision:

```bash
python scripts/explain_image.py --image /path/to/qr.png --url 'https://example.com/' --mode fusion
```

This is Grad-CAM for the image branch with the URL held fixed, not URL-token attribution or causal proof. Zero maps are preserved. Image preprocessing is RGB, torchvision Resize224x224, ToTensor and ImageNet normalization; no extra PIL resize is inserted before model input.

## Reported final Test results

All146425original Test images; threshold0.5, FP32. The table below is transcribed from the author's console output; authoritative full precision tables are collected under `reports/final_test/`.

| System | Accuracy % | Macro-F1 % | Malicious recall % | ROC-AUC |
|---|---:|---:|---:|---:|
| Visual | 83.13813 | 82.84535 | 81.78356 | 0.90262 |
| Text | 96.81270 | 96.70367 | 97.76359 | 0.99505 |
| Learned feature fusion + missing-URL visual fallback | 97.62677 | 97.55298 | 97.75778 | 0.99578 |
| Text + missing-URL visual fallback | 97.56667 | 97.49285 | 97.53135 | 0.99680 |

Most of the overall gain over text comes from the missing-URL visual fallback. The learned fusion head has a small descriptive gain over routing in Test Accuracy/Macro-F1, while routing has higher ROC-AUC. No statistical significance or general state-of-the-art claim is made. Visual robustness against the six tested shifts/resizes improved in the separate experiment; see `docs/results_and_limitations.md`.

## Reproduction and review

- See `docs/reproduction.md` for ordered notebooks and the distinction between source pretraining, continuation and head training.
- See `data/README.md` for external images/source CSV and exact retained row indices. No million-image dataset or raw URL table is redistributed.
- See `models/README.md` for artifact roles. The selected complete checkpoint contains all three systems, not just the small fusion head.
- See `docs/results_and_limitations.md` for selection, Test reuse, missing-URL policy and provenance limits.

The legacy text checkpoint lacks a split fingerprint; historical Train membership must be substantiated separately. Test was previously opened for visual experiments and later reused for fixed-weight multimodal reporting. These limitations must accompany research claims. Frozen branches, single-seed experiments and diagnostic feature interventions do not establish visual semantic understanding.

## Publishing

Extract the completed ZIP and commit the repository contents, not just the ZIP. `checkpoints/` is optional private optimizer-state backup and is ignored by Git. Do not add feature caches, raw datasets, credentials or unrelated PDF papers. Check upstream data/model terms before redistribution. No publication DOI, invented license or claim of a clean first-look Test is included. See `NOTICE.md` and the citation metadata.

GitHub's browser upload limit is25MiB per file; regular Git blocks files above100MiB. Use GitHub Releases or Git LFS for larger assets rather than adding dataset archives. Reference: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github .
