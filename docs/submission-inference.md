# Frozen submission inference

Issue #10 ships directory inference for the frozen Issue #7 learned-static policy. The degradation gate is not a runtime dependency: Issue #8 was closed as deadline-driven, not planned, without training or evaluation. The selected policy always requires both frozen experts and never silently falls back to RGB-only.

## Canonical command

```powershell
python submission_inference_cli.py `
  --image-dir <directory> `
  --bundle-dir artifacts/issue-7-fusion-v2 `
  --rgb-checkpoint <community-forensics-384.safetensors> `
  --signal-model <signal-model.json> `
  --output predictions.json `
  --device auto `
  --batch-size 8
```

Install the pinned signal and RGB requirements and use Python 3.12. The first RGB checkpoint acquisition may require network access; after the revision-pinned checkpoint, signal model, and complete generation directory are present, inference is fully offline. Checkpoint acquisition uses `rgb_expert.download_checkpoint`, which verifies the SHA-256 from `config/community-forensics-models.json`.

The generation directory must be the exact corrected Issue #7 inventory. The reader validates every completion/artifact binding, calibrator, static weight, normalizer, preprocessing revision, expert revision, and content-derived revision. It additionally pins:

- generation `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181`;
- bundle `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179`;
- bundle file SHA-256 `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2`;
- signal profile `hackathon-v1`, checkpoint and normalization revisions carried by the bundle;
- calibrated-logit weights `0.677` RGB and `0.323` signal.

## Input and output behavior

Supported extensions are BMP, GIF, JPEG/JPG, PNG, PPM, TIFF/TIF, and WebP, case-insensitively. Unsupported files are ignored. Nested paths are emitted as unique, sorted POSIX-relative paths. Shared preprocessing handles EXIF orientation, grayscale, and alpha consistently for both experts. A supported file that cannot be decoded aborts the whole run; path redirection outside the declared root, wrong backend counts, and non-finite results also abort. Publication is atomic, so a failed run neither creates a partial output nor overwrites an existing output.

An empty input writes exactly a valid JSON array (pretty-printed with a trailing newline). Every non-empty output is a JSON array whose records contain exactly `image_path` and `pred`; `pred` is a finite probability in `[0, 1]`. No labels, thresholds, logits, corruption metadata, experiment manifest, organizer knowledge, or provenance enter submission records. The frozen threshold logit is evaluation provenance only.

`--device cpu` never initializes a CUDA model. `--device cuda` fails when CUDA is unavailable. `--device auto` selects CUDA when PyTorch reports it available and otherwise selects CPU. Output bytes are deterministic across repeated runs on the same runtime, device, artifacts, and inputs; cross-device results are compared using the RGB metadata numeric tolerance rather than byte equality.

## Reproduction and profiling

On a clean machine, install both requirements files, acquire and verify the frozen artifacts above, run the canonical command on the shared fixture twice, and compare SHA-256 hashes. Compare each fixture probability to the evaluation-path result within `config/community-forensics-models.json`'s declared tolerance. Record OS, Python/PyTorch/CUDA versions, device, image count, batch size, wall time, images/second, output hash, and peak allocated CUDA bytes reported by the RGB backend.

The real Issue #7 generation, RGB checkpoint, and signal model are ignored runtime artifacts and are not present in this checkout. Consequently no honest real-fixture CPU/GPU throughput, peak-memory, parity delta, or prediction hash can be published until those artifacts are supplied on this machine.
