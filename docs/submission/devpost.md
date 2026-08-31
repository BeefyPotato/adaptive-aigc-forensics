# Adaptive AIGC Forensics

## Problem and solution

AI-generated-image detectors can lose useful evidence after everyday JPEG recompression, resizing, blur, noise, color adjustment, or cropping. Adaptive AIGC Forensics makes those transformations explicit through a reproducible corruption harness and source-disjoint development partitions. The current deadline-safe candidate is **raw RGB-only**: frozen Community Forensics 384 inference. It is not represented as an adaptive or fused system.

## Technical implementation

The current candidate uses the immutable Community Forensics 384 checkpoint, with RGB decoding, orientation correction, resizing, center cropping, normalization, and batched inference behind `rgb_cli.py`. Its emitted record is exactly `{ "image_path": string, "pred": number }`, where `pred` is a finite probability from 0 to 1.

The research architecture also includes a signal expert: a deterministic 26-value signal representation followed by a frozen MLP. Corruption variants are materialized losslessly before branch-specific preprocessing. These research components, including fusion and a degradation gate, are not claimed as part of the current candidate.

## Development tools

Python runs model inference, signal processing, and contract tests. Node.js 22 plus Sharp generates and validates corruption-harness artifacts. Git records source revisions; SHA-256 binds selected data, materialized observations, checkpoint files, manifests, and any shareable result output.

## Models and APIs

The candidate uses Community Forensics 384 from OwensLab at the revision and SHA-256 recorded in `config/community-forensics-models.json`. The checkpoint is frozen; no model weights are fine-tuned. The tool downloads through Hugging Face tooling when a local checkpoint is not supplied, then verifies the expected checkpoint checksum before PyTorch loads it.

## Libraries and frameworks

The implementation uses PyTorch, torchvision, timm, NumPy, Pillow, safetensors, huggingface-hub, Node.js, and Sharp. Pinned versions appear in `requirements-rgb.txt`, `requirements-signal.txt`, and `package-lock.json`; attribution and license records are in [attributions](attributions.md).

## Datasets and assets

SID_Set is the controlled development source pool. Selection preserves source-level partitioning, local file checksums, and exact/perceptual overlap checks. COCO val2017 and DALL-E Advanced organizer materials are evaluation-only: they cannot influence training, calibration, any selection, weights, thresholds, templates, or narrative. Dataset access and provenance constraints are documented in [data sources](../data-sources.md).

## Robustness and error analysis

The corruption harness reports clean performance and severity-averaged JPEG, blur, resize, RGB-noise, atomic-color, and center-crop families. Any number from this process is an **internal validation** result only when it names the raw-RGB candidate, materialized-manifest SHA-256, checkpoint SHA-256, output SHA-256, and generation revision. This public package intentionally publishes no unbound metric, error count, or organizer score.

For error analysis, inspect false positives and false negatives only within the source-disjoint internal validation set, recording the applicable corruption condition and provisional threshold. Do not use organizer labels to select examples or tune the candidate.

## Innovation and complementary value

The contribution is a provenance-conscious way to compare explicit low-level signal evidence with a frozen RGB expert under the same lossless materialized observation and source-disjoint splits. Complementary value is a held-out correction claim, not a feature-importance claim. It remains a research question here, not a performance claim for the current raw RGB-only candidate.

## Impact and feasibility

The candidate is straightforward to reproduce from a directory of images using one portable Python command, while the wider harness makes robustness evidence inspectable rather than relying on undeclared augmentation. The output is a probability for review workflows, not an autonomous moderation or provenance decision.

## Limitations and next steps

Public Community Forensics metadata does not provide an image-level ledger proving every organizer demonstration image was absent from its upstream training. The organizer set remains evaluation-only and locally controlled training sources are overlap-checked. The current candidate is not calibrated for deployment, does not adapt its trust per image, and has no public organizer result. Next steps are to freeze and review a candidate, report checksummed internal-validation artifacts, test unseen generators, and independently evaluate the organizer set without changing the candidate.

## Team contributions

Human team confirmation is required before naming any contributor or assigning credit. The repository provides a role-based contribution-record template in the [claim ledger](claim-ledger.md); it deliberately contains no inferred names or unconfirmed assignments.

## Demo and repository

Run the current candidate with:

```shell
python rgb_cli.py --input-dir ./images --output ./predictions.json --resolution 384 --device auto --batch-size 8
```

The repository README gives setup, data preparation, reproducible internal-validation commands, and the output schema. The [demo script](demo-script.md) is a 120-second recording plan. Before publishing a result, fill and review the candidate-bound [claim ledger](claim-ledger.md).
