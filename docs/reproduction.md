# Reproduction scope and training sequence

## Inference / reported Test review

1. Assemble the completed package and run `python scripts/verify_package.py`.
2. Run `python scripts/smoke_model.py` (CPU; synthetic forward only).
3. To repeat the exact Kaggle Test workflow, attach original dataset/CSV, run `python scripts/restore_kaggle_layout.py`, then notebook00 to rebuild the audited manifest, and notebook07 for fixed-weight evaluation. Notebook07 is the authoritative same-cohort, all/URL-available/URL-missing evaluation.

Historical notebooks preserve author paths, fixed source hashes and run configuration guards. Read paths before running; for new hyperparameters or new training runs choose a new experiment output folder. Never bypass identity checks to mix feature caches/checkpoints.

## Ordered notebooks

- 00: audit saved membership and image bytes/readability; no training or fold reassignment.
- 01: visual continuation from the legacy200K checkpoint on all710746Train images, up to5epochs; selected phase1 epoch4.
- 02: robust visual augmentation; geometric perturbations on Train; clean+transformed Validation selection. The reported run completed2epochs in this phase.
- 02b: equivalent experiment's parallel-loader continuation, one completed epoch per invocation. Worker count changes augmentation RNG sequences; runtime changes are logged.
- 03: paired visual Test robustness comparison (already-opened Test history must be disclosed).
- 04: frozen features for visual/text branches and text-initialized fusion-head trial. This selected epoch0 (routing-equivalent baseline), not a learned multimodal gain.
- 05: independent randomly initialized512→256→128→2fusion head on the existing features. Both source branches remain fixed. Available-URL Train rows update the head; missing-URL inference routes to visual. Up to10epochs; patience3; observed stop at epoch4 and selected trained epoch1.
- 07: final same-pass Test evaluation for visual/text/learned fusion plus routing baseline. Threshold0.5, FP32; no Test-based tuning.

Frozen feature extraction uses original224RGBimages and cached URL text, tokenizer max length128. Visual source augmentation is already reflected in its weights; fusion-head training does not add image augmentation to these cached vectors.

## What cannot be claimed as fully reproduced from this package

The original200K visual training and legacy full-text training occurred before this package's continuation notebooks. Supplied weights and reported settings are retained, but the original text checkpoint has no saved data_signature proving its Train membership. The dossier describes grouped splitting; exact old membership needs independent provenance. This is a continuation/evaluation package, not an asserted from-scratch, bit-for-bit reproduction of all legacy training.

Legacy visual best_epoch4 is a warm-start. Five later full-Train visual epochs were executed, with phase1 epoch4 selected, followed by two robust epochs. Thus the final visual model did not start from scratch for just two epochs. See collected histories for exact executed epochs and changes in runtime.

Exact installed versions are collected from the actual Kaggle runtime. Those versions can differ from earlier stages; stage configs and source hashes remain the authoritative historical records. The portable code's actual-weight CPU smoke test runs during collection; notebooks are not retrained by collection.
