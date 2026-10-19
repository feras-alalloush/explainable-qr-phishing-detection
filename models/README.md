# Model artifacts

This completed package contains the following verified model artifacts:

- `selected_three_systems.pt` + `.json`: complete selected epoch1 state; visual.*, text.* and fusion.* keys. All inference systems are derived from this one export.
- `tokenizer/`: the exact saved tokenizer; no Internet is needed for inference.
- `source_visual.pt`, `source_text.pt`: byte-verified frozen sources for continuation review.
- `visual_phase1.pt` and `legacy_visual_200k.pt`: optional historical sources when still available; any absence is recorded in PACKAGE_STATUS.json.
- `weights_manifest.json`: roles and SHA256 digests.

Source artifacts are owner's trusted PyTorch files; load with the provided checks. The small head checkpoint alone is insufficient for inference. Optimizer state, when found, is under ignored `checkpoints/`; the large feature cache is deliberately excluded.
