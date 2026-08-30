# Frozen Community Forensics RGB expert

Issue #4 integrates the existing Community Forensics detector as the project's RGB expert. The model is **frozen**: inference sets every parameter to non-trainable and never updates the checkpoint. The 384-pixel checkpoint is the primary baseline; the separately trained 224-pixel checkpoint is only a smoke-test and constrained-memory fallback.

## Pinned assets

`config/community-forensics-models.json` records the original repository and immutable revision, official OwensLab checkpoint repositories and revisions, LFS SHA-256 values, MIT license, input resolution, exact parameter count, score direction, and the provenance limitation governed by ADR 0001. A checkpoint is hashed before PyTorch loads it.

The public checkpoint metadata does not prove image-level exclusion of every organizer demonstration image. Do not use the organizer demonstration set for fine-tuning, calibration, model selection, or threshold selection.

## Install

Use Python 3.11 or 3.12 in a virtual environment. Install the GPU-specific PyTorch wheel appropriate to the teammate machine when necessary, then install the pinned dependencies:

```shell
python -m pip install -r requirements-rgb.txt
```

## Directory inference

The default command downloads the immutable 384 checkpoint into the Hugging Face cache, verifies it, performs batched inference, and writes the Track 5 submission contract:

```shell
python rgb_cli.py --input-dir ./images --output ./predictions.json --device cuda --batch-size 16
```

For the 224 smoke fallback:

```shell
python rgb_cli.py --input-dir ./images --output ./predictions-224.json --resolution 224 --device cpu
```

Pass `--checkpoint` to use an already downloaded file. Supported images are discovered recursively and sorted by stable relative path. An unreadable supported file fails the command instead of being silently omitted. Successful records contain exactly `image_path` and `pred`; `pred` is the sigmoid of the upstream fake-image logit.

## Shared preprocessing contract

Both `infer_directory` and `predict_experiment_observations` call the same `preprocess_image` function:

1. apply EXIF orientation;
2. convert grayscale, palette, or alpha inputs to RGB;
3. resize the shorter edge to 440 (primary) or 256 (fallback) with bilinear interpolation while preserving aspect ratio;
4. take the centered 384×384 or 224×224 crop;
5. convert to float32 CHW RGB in `[0, 1]` and apply ImageNet mean/std normalization.

Experiment cache records retain the stable source and corruption-variant identifiers, raw RGB logit, probability, preprocessing version, checkpoint revision, and a deterministic `rgb-cache-v1` key. They intentionally remain RGB-only artifacts; the combined `artifact-v1` record is assembled later when the signal workstream is available.

## Tests

The contract tests do not download weights:

```shell
python -m unittest tests/test_rgb_expert.py
```

After installing both Node and RGB dependencies, run both project suites with `npm run verify`.

The teammate GPU acceptance run must additionally verify both real checkpoints, batched 384 CUDA inference, and the 224 fixture fallback within the declared device tolerance.

After installing the RGB dependencies, the opt-in real 224 checkpoint smoke test is:

```shell
COMMUNITY_FORENSICS_INTEGRATION=1 python -m unittest tests/test_rgb_expert.py
```
