# adaptive-aigc-forensics

A degradation-aware AI-generated image detector that combines an RGB AIGC expert with low-level forensic signals such as FFT energy, neighbouring-pixel statistics, and residual features, then dynamically fuses both under real-world transformations like JPEG, blur, resize, and noise.

## Reproducible contract fixture

The issue #2 fixture establishes the versioned handoff contracts used by later experiment and submission work. It requires Node.js 22 or newer.

From a clean checkout, run the complete fixture with:

```shell
npm run fixture
```

The command decodes two checked-in images, applies seeded noise, passes the same corrupted observation to fixture RGB and signal expert seams, fuses their logits, validates cache metadata, evaluates the predictions, and writes these deterministic artifacts under `artifacts/fixture/`:

- `resolved_manifest.json`
- `cache.json` and individually readable records under `cache/`
- `predictions.json`
- `metrics.json`

`predictions.json` is the submission contract: a JSON array ordered by stable relative image path, with exactly `image_path` and `pred` in each record. Every `pred` must be finite and within `[0, 1]`. The fixture declares a cross-machine numeric tolerance of `1e-12` in its configuration and model bundle.

Run the smoke tests with:

```shell
npm test
```

The dependency-free P3 decoder and toy experts are fixture implementations, not the production checkpoint or general image-format support. Later tickets can replace the expert functions and decoder behind the established seams without changing the artifact contracts. See [docs/contracts.md](docs/contracts.md) for the schemas and compatibility rules.

## Track 5 experiment manifest

Issue #3 adds the production source-selection, leakage-audit, corruption, and balanced-sampling contracts. Install the exact dependency lock and run its controlled two-image fixture with:

```shell
npm ci
npm run track5:fixture
```

This writes a deterministic `track5-manifest.json` and `track5-leakage-audit.json` under `artifacts/track5-fixture/`. The fixture uses an explicit two-source split plan. Production preparation uses `node ./scripts/download-sid-set-candidates.mjs` to verify or download a local 14,600-image pool against the tracked content hashes; collision-aware selection backfills from that pool to produce the required 14,000 source partitions. See [docs/track5-manifest.md](docs/track5-manifest.md) for the downloader, production CLI, inventory schema, split allocation, corruption conditions, leakage policy, and sampler contract.

## Dataset access

Track 5 dataset locations, intended experiment roles, and organizer-set restrictions are documented in [docs/data-sources.md](docs/data-sources.md).

## RGB baseline

The project uses the existing Community Forensics detector as a frozen RGB expert: the checkpoint is evaluated but never retrained. Issue #4 pins the official 384-pixel primary model and 224-pixel smoke fallback, verifies their checksums, and provides shared experiment/directory preprocessing plus label-free JSON inference. See [docs/rgb-expert.md](docs/rgb-expert.md).

Issue #5 adds the reproducible robustness-baseline runner. It writes versioned logits for downstream fusion, internal-validation robustness metrics, runtime/GPU profiling, and a deterministic rerun check without exposing sealed-test labels. See [docs/rgb-baseline.md](docs/rgb-baseline.md).

The RGB baseline consumes a fully resolved Issue-3 materialized manifest. The signal runner accepts the finalized recipe manifest and resolves only its balanced training draw and complete internal-validation matrix in bounded, transient shards. Both routes use the same lossless, checksummed materialized observation as the common pixel handoff; neither expert may report a clean source image as a corruption result.

## Signal expert

Issue #6 adds the deterministic 26-value signal representation, the Issue-3 balanced sampler, leakage-safe normalization and MLP training, strictly versioned feature/logit/checkpoint artifacts, and canonical severity-first robustness metrics. Its public `signal_cli.py run` seam takes the finalized Issue-3 manifest and local images all the way to a frozen signal-only checkpoint and internal-validation report without materializing the roughly 202 GiB selected experiment at once. See [docs/signal-expert.md](docs/signal-expert.md).

## Submission overview

Adaptive AIGC Forensics investigates how image-authenticity evidence changes after common delivery transformations. The current, deadline-safe submission candidate is **raw RGB-only**: the frozen 384-pixel Community Forensics RGB expert. The signal expert and static-fusion work are retained as reproducible research paths, not public submission claims or the current inference candidate.

This repository distinguishes three evidence scopes: the source-disjoint **internal validation set** for development, a sealed internal test set for one-time internal reporting, and the organizer demonstration set for evaluation only. No organizer demonstration image may affect training, calibration, model selection, or threshold selection. Any future reported result must name its candidate, artifact revision, manifest/checkpoint SHA-256 values, and be labeled **internal validation** unless it is explicitly an organizer demonstration result.

## Architecture

The research architecture has two complementary evidence paths:

- The **RGB expert** is the frozen Community Forensics 384 checkpoint. It decodes and normalizes RGB pixels and returns an AI-generated-image probability.
- The **signal expert** is a deterministic 26-value signal representation plus a frozen MLP. It uses explicit low-level image statistics rather than a second learned RGB representation.
- The **degradation gate** and fusion routes are experimental research components. They are not part of the current raw-RGB submission candidate and must not be inferred from its output.

Every corruption variant is materialized as a lossless RGB observation before branch-specific preprocessing, and source images—not variants—own the data splits. See [the Track 5 manifest contract](docs/track5-manifest.md) and [the RGB expert contract](docs/rgb-expert.md).

## Setup

Use Python 3.11 or 3.12 and Node.js 22 or later. The public commands deliberately use portable `python` and `node` executables; activate your own virtual environment first if you use one.

```shell
python -m pip install -r requirements-rgb.txt
npm ci
```

Install `requirements-signal.txt` only when reproducing the non-candidate signal research path. GPU users should select a PyTorch build suitable for their system before installing the pinned RGB requirements.

## Data preparation and corruption generation

The repository never stores downloaded image data. Obtain the SID_Set assets according to their terms, then verify and select the pinned candidate pool:

```shell
node ./scripts/download-sid-set-candidates.mjs
node ./src/track5-cli.js build-manifest --inventory ./datasets/sid-set/inventory.jsonl --dataset-root ./datasets/sid-set/images --dataset-revision 'saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd' --output-dir ./artifacts/track5-production
node ./scripts/materialize-track5-observations.mjs --manifest ./artifacts/track5-production/track5-manifest.json --dataset-root ./datasets/sid-set/images --output-dir ./artifacts/track5-materialized --concurrency 4
```

Materialization generates clean and declared JPEG, blur, resize, noise, atomic-color, and crop corruption variants as lossless RGB PNGs and binds each to a SHA-256. Provide organizer hashes when available so overlap checking is enforced; details are in [data sources](docs/data-sources.md).

## Training, calibration, and reporting order

The frozen RGB candidate is not trained. For the wider research pipeline, preserve this order: source split and leakage audit; corruption materialization; frozen RGB scoring; signal-expert training; fusion-training calibration; internal-validation-only threshold diagnostics; then a one-time sealed internal report. The organizer demonstration set remains evaluation-only throughout.

To reproduce a candidate-bound internal-validation RGB run after materialization:

```shell
python rgb_baseline_cli.py --manifest ./artifacts/track5-materialized/track5-materialized-manifest.json --dataset-root ./artifacts/track5-materialized --output-dir ./artifacts/rgb-baseline --resolution 384 --device auto --batch-size 8
```

The resulting cache and reports bind the materialized-manifest SHA-256, checkpoint revision, preprocessing version, and corruption implementation version. Treat every metric emitted by this command as **internal validation** unless its report explicitly says otherwise; do not compare or combine it with organizer results.

## Canonical inference (current candidate)

The sole public inference command for the current raw-RGB candidate is:

```shell
python rgb_cli.py --input-dir ./images --output ./predictions.json --resolution 384 --device auto --batch-size 8
```

`rgb_cli.py` verifies the immutable Community Forensics 384 checkpoint (`b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`) before loading it. Its output is a JSON array, sorted by stable relative image path, with exactly this per-image schema:

```json
{"image_path": "relative/path/to/image.png", "pred": 0.73}
```

`pred` is a finite probability in `[0, 1]`; it is not a calibrated decision, a provenance verdict, or a fusion score. Save the output SHA-256 beside the command, checkpoint checksum, and candidate name before sharing an internal validation result. The [claim ledger](docs/submission/claim-ledger.md) is the required public-record template.

## Reproducing checks

Run the controlled Node fixture and project suites from a clean checkout:

```shell
npm run track5:fixture
npm run verify
```

The fixtures verify contracts, not benchmark performance. See [submission notes](docs/submission/devpost.md), [attributions](docs/submission/attributions.md), and the [demo script](docs/submission/demo-script.md) for the candidate-bound submission package.

## Limitations and improvements

The Community Forensics checkpoint is frozen and its public metadata cannot prove image-level exclusion of every organizer demonstration image. The organizer set therefore stays evaluation-only, and locally controlled sources receive exact and perceptual overlap checks. The current candidate does not adapt its trust to per-image degradation and should not be treated as a deployment-ready moderation system. Next work includes completing and independently reviewing candidate selection, calibrating only on source-disjoint data, evaluating organized demonstration data separately, and studying robustness across unseen generators and transformations.

## Contributions

Contribution authorship must be confirmed by the human team before publication. Use the role-based record in [the claim ledger](docs/submission/claim-ledger.md) to map a person, their reviewed contribution, and approval date. This repository intentionally does not infer or assign individual contributions.
