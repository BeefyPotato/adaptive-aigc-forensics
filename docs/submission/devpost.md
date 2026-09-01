# Adaptive AIGC Forensics

## Problem and solution

AI-generated-image detectors can lose useful evidence after everyday JPEG recompression, resizing, blur, noise, color adjustment, or cropping. Adaptive AIGC Forensics makes those transformations explicit through a reproducible corruption harness and source-disjoint development partitions. The selected design is **learned-static-fusion**: a frozen Community Forensics 384 RGB expert and a frozen low-level signal expert contribute complementary evidence through one learned, condition-independent allocation.

## Technical implementation

The 21,811,969-parameter RGB expert uses the immutable Community Forensics 384 checkpoint with RGB decoding, orientation correction, resizing, center cropping, and normalization. The signal expert applies a deterministic 26-value representation and a frozen 26→16→1 tanh MLP with 449 trainable scalar parameters to the same losslessly materialized observation. Both logits are calibrated, then learned static fusion combines them using **0.677 RGB / 0.323 signal**, fitted only on the source-disjoint fusion-training set.

Static fusion uses one trust allocation for every condition. The per-image degradation gate remains a separate research component and is not part of this selected design. The final accepted interface must emit exactly `{ "image_path": string, "pred": number }`, where `pred` is finite and from 0 to 1; the portable command is pending Issue #10 acceptance and final CLI binding.

## Development tools

Python runs model inference, signal processing, and contract tests. Node.js 22 plus Sharp generates and validates corruption-harness artifacts. Git records source revisions; SHA-256 binds selected data, materialized observations, checkpoint files, manifests, and any shareable result output.

## Models and APIs

The RGB component uses Community Forensics 384 from OwensLab, revision `6076002bf0d9dd37537f965ee2f06f826c333b61`, with SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`. The signal model is bound by SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5`. The selected static weight is bound to trusted generation `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181` and bundle SHA-256 `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2`. Both experts are frozen for submission inference.

## Libraries and frameworks

The implementation uses PyTorch, torchvision, timm, NumPy, Pillow, safetensors, huggingface-hub, Node.js, and Sharp. Pinned versions appear in `requirements-rgb.txt`, `requirements-signal.txt`, and `package-lock.json`; attribution and license records are in [attributions](attributions.md).

## Datasets and assets

SID_Set is the controlled development source pool. The source-level allocation is 8,000 expert-training sources for signal fitting, 2,000 fusion-training sources for calibrators/static fusion, 2,000 internal-validation sources for development decisions, and 2,000 sealed-internal-test sources reserved for one-time internal reporting. Selection preserves class balance, source-level partitioning, local file checksums, and exact/perceptual overlap checks. COCO val2017 and DALL-E Advanced organizer materials are evaluation-only: they cannot influence training, calibration, any selection, weights, thresholds, templates, or narrative. Dataset access and provenance constraints are documented in [data sources](../data-sources.md).

## Robustness and error analysis

The candidate-bound evidence at `docs/submission/evidence/submission-evidence.json` and its completion receipt produce `docs/submission/results/robustness-and-errors.md` and `docs/submission/results/clean-vs-transformed.svg`. The report receipt binds every published file. On the source-disjoint **internal validation** set, learned-static-fusion records:

| Metric | Value |
| --- | ---: |
| Clean AUROC | 0.981975 |
| Mean transformed AUROC | 0.9567506944444445 |
| All-condition macro AUROC | 0.9603541666666667 |
| Weakest family / severity / AUROC | noise / sigma-0.1 / 0.810425 |
| Brier score | 0.09982875909583232 |
| Condition-balanced Brier score | 0.09680062702417022 |
| Provisional-threshold balanced accuracy | 0.87625 |
| False-positive rate | 0.08399999999999996 |
| False-negative rate | 0.16349999999999998 |

These are candidate-bound development results, not sealed, independent-test, official, or organizer scores. The [claim ledger](claim-ledger.md) records the generation, bundle, manifest, expert, evidence, and report bindings.

At each candidate's provisional internal-validation threshold, the signal expert corrected **768/1218 = 0.6305418719211823** calibrated-RGB errors. Learned static fusion's all-condition macro-AUROC gain over calibrated RGB-only was **0.016795535714285936**, with deterministic source-bootstrap interval **[0.011076105794972707, 0.0234869800759804]**. This is descriptive complementary-value evidence on internal validation, not a causal, sealed, independent-test, or organizer claim.

For error analysis, inspect false positives and false negatives only within the source-disjoint internal validation set, recording the applicable corruption condition and provisional threshold. Do not use organizer labels to select examples or tune the candidate.

## Innovation and complementary value

The contribution is a provenance-conscious way to combine explicit low-level signal evidence with a frozen RGB expert under the same lossless materialized observation and source-disjoint splits. Complementary value is evaluated through held-out corrections rather than treated as feature importance. The selected system uses learned static fusion; an adaptive degradation gate remains future research.

## Impact and feasibility

Both expert paths operate on one checksummed materialized observation, and the wider harness makes robustness evidence inspectable rather than relying on undeclared augmentation. An implementation-authored same-device CPU smoke on two checked-in images took 23.18 seconds including startup/model validation, peaked at 466,698,240 working-set bytes, and produced the same output SHA-256 in explicit CPU and `auto` modes. This record is not independent Issue #10 acceptance. The accepted directory interface will produce a probability for review workflows, not an autonomous moderation or provenance decision; its exact command remains withheld until the acceptance gate passes.

## Limitations and next steps

Public Community Forensics metadata does not provide an image-level ledger proving every organizer demonstration image was absent from its upstream training. The organizer set remains evaluation-only and locally controlled training sources are overlap-checked. Learned static fusion does not adapt its trust per image, is not calibrated for deployment, and has no public organizer result. Final CLI acceptance remains a publication gate. Next steps include testing unseen generators and independently evaluating the organizer set without changing the selected system.

## Team contributions

Human team confirmation is required before naming any contributor or assigning credit. The repository provides a role-based contribution-record template in the [claim ledger](claim-ledger.md); it deliberately contains no inferred names or unconfirmed assignments.

## Demo and repository

The final learned-static-fusion command is pending Issue #10 acceptance and final CLI binding; do not substitute the RGB-expert diagnostic command. The repository README gives setup, data preparation, candidate-bound internal-validation evidence, and the required output schema. The [demo script](demo-script.md) is a 120-second recording plan. Before publishing a result, review the candidate-bound [claim ledger](claim-ledger.md).
