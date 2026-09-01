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

The deployment bundle directory requires only `static-fallback.complete.json` and `static-fallback-bundle.json`; training labels, corruption manifests, matched logits, and calibrated evaluation caches are not inference dependencies. The receipt still binds the complete Issue #7 generation inventory, while the deployment reader deliberately opens only the receipt and bundle. It validates their content-derived revisions, the bundle checksum, calibrators, static weight, normalizers, preprocessing contract, and expert revisions. It additionally pins:

- generation `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181`;
- bundle `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179`;
- bundle file SHA-256 `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2`;
- Community Forensics 384 checkpoint SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`;
- signal-model SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5`;
- signal profile `hackathon-v1`, checkpoint and normalization revisions carried by the bundle;
- RGB checkpoint revision, 384/440 geometry, ImageNet normalization, preprocessing version, score direction, and shared-observation preprocessing version;
- calibrated-logit weights `0.677` RGB and `0.323` signal.

Both model files and every runtime binding are verified before either model is constructed. Runtime validation reads the exported shared-observation preprocessing revision and geometry map plus the RGB expert's actual ImageNet mean/scale arrays; it does not trust a second set of local preprocessing strings. Any substituted checkpoint, signal model, receipt, bundle, preprocessing contract, score direction, normalization, or shared geometry fails closed.

The public Python entry point is `submission_inference.run_submission_inference(...)`. It owns artifact validation, device resolution, both model constructors, recursive inference, and atomic publication. The command-line adapter routes exclusively through this entry point.

## Input and output behavior

Supported extensions are BMP, GIF, JPEG/JPG, PNG, PPM, TIFF/TIF, and WebP, case-insensitively. Unsupported files are ignored. Nested paths are emitted as unique, sorted POSIX-relative paths. Shared preprocessing handles EXIF orientation, grayscale, and alpha consistently for both experts. A supported file that cannot be decoded aborts the whole run; path redirection outside the declared root, wrong backend counts, and non-finite results also abort. Publication is atomic, so a failed run neither creates a partial output nor overwrites an existing output.

An empty input writes exactly a valid JSON array (pretty-printed with a trailing newline). Every non-empty output is a JSON array whose records contain exactly `image_path` and `pred`; `pred` is a finite probability in `[0, 1]`. No labels, thresholds, logits, corruption metadata, experiment manifest, organizer knowledge, or provenance enter submission records. The frozen threshold logit is evaluation provenance only.

`--device cpu` does not query the CUDA runtime. `--device cuda` fails when CUDA is unavailable, before artifact or model loading and before output publication. `--device auto` is the only mode that may probe CUDA; it selects CUDA when PyTorch reports it available and otherwise selects CPU. Output bytes are deterministic across repeated runs on the same runtime, device, artifacts, and inputs; cross-device results are compared using the RGB metadata numeric tolerance rather than byte equality.

## Reproduction and profiling

On a clean environment, install both requirements files, acquire and verify the frozen artifacts above, run the canonical command on the shared fixture twice, and compare SHA-256 hashes. Run the opt-in real-artifact gate with `SUBMISSION_REAL_BUNDLE_DIR`, `SUBMISSION_REAL_RGB_CHECKPOINT`, and `SUBMISSION_REAL_SIGNAL_MODEL` set:

```powershell
python -m unittest tests.test_submission_inference_real -v
```

It compares each submitted probability with the canonical RGB experiment path plus the canonical signal representation/model and calibrated-fusion path, using `config/community-forensics-models.json`'s `1e-5` tolerance. Set `SUBMISSION_REAL_IMAGE_DIR` to run the same gate on an ignored local sample directory. The test also enforces exact output keys, sorted relative paths, finite unit-interval probabilities, and byte-identical repeated CPU output.

## Single-device acceptance record

The 2026-09-01 acceptance run used Windows 11 `10.0.26200`, Python `3.12.10`, NumPy `2.3.5`, Pillow `12.0.0`, PyTorch `2.8.0+cpu`, timm `1.0.19`, and safetensors `0.6.2`. CUDA was unavailable, so GPU execution was inapplicable and `auto` correctly selected CPU.

On the two checked-in fixture images with batch size 2:

- the opt-in real parity/repeatability test passed in 20.37 seconds;
- repeated CPU fixture output was byte-identical; the observed maximum absolute delta from the independent canonical expert/fusion calculation was exactly `0` (within `1e-5`);
- explicit CPU and real `auto` produced SHA-256 `adcd0528bd98130421385fd7d579ea8ba4ae6aa773f1c4b6e90504a2c749c1b3`;
- a separately profiled CPU process completed in 23.18 seconds (0.086 images/second including interpreter, validation, and model loading) with a peak observed working set of 466,698,240 bytes;
- explicit unavailable CUDA exited with status 2 and published no output.

The sampled-real parity batch was selected before scoring as the lexicographically first two filenames in each of the local SID_Set internal-validation `authentic` and `full-synthetic` directories. It is not organizer data. The ignored source-byte identities were:

| SID_Set internal-validation identity | Bytes | SHA-256 |
| --- | ---: | --- |
| `authentic/0002d5c6b40edcd4.jpg` | 147,109 | `238ac3d883867d9253b9f66953ccca969793b1644d08df53f907bf2c400c54c6` |
| `authentic/00032d5bb63c29eb.jpg` | 136,504 | `4a7a351d5fff74295074834efbdad92b53d41754ed2bde70a9e0f3871abc4a5b` |
| `full-synthetic/full_synthetic_000021.jpg` | 173,637 | `ed04d319ba8c9d4dd688393e2b10dbe0172deffc10f1ccb0d4387744384fa9b0` |
| `full-synthetic/full_synthetic_000022.jpg` | 176,449 | `4e287901a8ecb69f783223e05d59af141f3d69e92bc3b7bc2a06c9c73cca835d` |

On this four-image sampled-real batch, repeated CPU output was byte-identical at SHA-256 `21bbd744c94927e674bd9f40b3f56c9ac3188580b49b2d32869cb576e65dd2c2`; the observed maximum absolute parity delta was exactly `0` (within `1e-5`).

The real Issue #7 receipt/bundle, 87,262,324-byte RGB checkpoint, and signal model remain ignored runtime artifacts; no weights, caches, labels, manifests, or image bytes are committed. This project records same-device evidence only.
