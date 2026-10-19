# Results and limits of interpretation

The main systems are visual-only, text-only and learned feature fusion, with missing-URL visual fallback made explicit. Text+visual fallback is a separate fourth baseline. Never rename the routing-only epoch0 trial as successful learned fusion.

On all146425Test images, author-reported Accuracy is83.13813%,96.81270%,97.62677%,97.56667% respectively. Most gain over text is fallback on the7120missing-URL cases. On139305available-URL cases, text Accuracy97.7345% vs learned fusion97.7976%; this is a small descriptive difference. Routing has higher full-Test ROC-AUC than learned fusion. No significance or state-of-the-art claim is justified by these numbers alone.

## Robustness

The separate visual-only experiment raised mean Macro-F1 across six fixed shifts/resizes from57.3978% to76.1953% on a balanced Test sample. The full original-image visual Macro-F1 changed82.8611%→82.8453%, and malicious recall fell83.5091%→81.7836%. Report that tradeoff. Shifts were4pixels without resampling;224→208shrinks usedNearest/Bilinear on a white224canvas. These are limited synthetic conditions, not all camera/print degradations. The six conditions reuse images, so the mean is not an independent6000-image sample. Translation eligibility and retained sample counts must be read from the original robustness reports.

## Explainability and modality reliance

Exploratory old visual Grad-CAM highlighted right-side/border areas, motivating controlled diagnostics. That does not prove a border shortcut or URL semantic understanding. New `scripts/explain_image.py` targets the chosen visual/fusion class logit with the URL fixed; it attributes image contribution only. Do not describe it as explaining URL tokens.

Random-fusion Validation interventions on available URLs gave Macro-F1 intact96.7871%,zero-visual96.6228%,permute-visual96.6523%,zero-text58.7363%,permute-text50.3720%. These indicate much stronger text dependence and a small image effect under these interventions; zeroing/permutation can be out of distribution and are not causal proofs.

## Selection and provenance

- All trainable fusion-head selection used full Validation Macro-F1; best trained epoch1 was fixed before the final multimodal Test run.
- Test was previously opened for visual baselines and robustness reporting. Describe its later reuse, not a pristine first-look holdout.
- The legacy text checkpoint lacks a Train-membership fingerprint. Public data/CSV grouping does not itself prove a checkpoint was trained only on the matching Train set.
- One seed and one corpus; no external photographed QR cohort, unseen distortion suite or repeated-seed significance study.
- Available URL strings are saved decoder outputs. Missing-URL text responses use a legacy placeholder and must be interpreted separately.
- Reported runtime is not end-to-end image acquisition plus QR decoding; raw URLs are never visited.

Definitive full-precision machine-readable tables, counts and predictions are in `reports/final_test/` after collection. `reports/reported/` is clearly labelled transcription, not a newly computed evaluation.
