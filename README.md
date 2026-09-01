# adaptive-aigc-forensics

A provenance-bound AI-generated image detection system that combines a frozen RGB expert with a deterministic low-level signal expert. The selected design uses learned static fusion under real-world transformations such as JPEG, blur, resize, and noise; the per-image degradation gate remains a separate research path.

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

Adaptive AIGC Forensics investigates how image-authenticity evidence changes after common delivery transformations. The selected submission design is **learned-static-fusion**: the frozen 384-pixel Community Forensics RGB expert and the frozen deterministic 26-value signal expert are calibrated separately, then combined using the trusted fusion-training allocation of **0.677 RGB / 0.323 signal**. Static fusion uses the same learned allocation for every image; it is not the per-image degradation gate.

This repository distinguishes three evidence scopes: the source-disjoint **internal validation set** for development, a sealed internal test set for one-time internal reporting, and the organizer demonstration set for evaluation only. Organizer demonstration data cannot influence training, calibration, any selection, weights, thresholds, templates, or narrative. Every reported result must name its candidate, artifact revision, manifest/checkpoint SHA-256 values, and be labeled **internal validation** unless it is explicitly an organizer demonstration result.

## Architecture

The research architecture has two complementary evidence paths:

- The **RGB expert** is the frozen Community Forensics 384 checkpoint. It decodes and normalizes RGB pixels and returns an AI-generated-image probability.
- The **signal expert** is a deterministic 26-value signal representation plus a frozen MLP. It uses explicit low-level image statistics rather than a second learned RGB representation.
- **Learned static fusion** calibrates both expert logits and combines them with the fusion-training weights 0.677 RGB / 0.323 signal. The trusted weight is fixed across image conditions.
- The **degradation gate** is an experimental research component and is not part of the learned-static-fusion design.

Every corruption variant is materialized as a lossless RGB observation before branch-specific preprocessing, and source images—not variants—own the data splits. See [the Track 5 manifest contract](docs/track5-manifest.md) and [the RGB expert contract](docs/rgb-expert.md).

## Setup

Use Python 3.11 or 3.12 and Node.js 22 or later. The public commands deliberately use portable `python` and `node` executables; activate your own virtual environment first if you use one.

```shell
python -m pip install -r requirements-rgb.txt
npm ci
```

Install `requirements-signal.txt` to reproduce the signal expert and fusion pipeline. GPU users should select a PyTorch build suitable for their system before installing the pinned RGB requirements.

## Data preparation and corruption generation

The repository never stores downloaded image data. Obtain the SID_Set assets according to their terms, then verify and select the pinned candidate pool:

```shell
node ./scripts/download-sid-set-candidates.mjs
node ./src/track5-cli.js build-manifest --inventory ./datasets/sid-set/inventory.jsonl --dataset-root ./datasets/sid-set/images --dataset-revision 'saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd' --output-dir ./artifacts/track5-production
node ./scripts/materialize-track5-observations.mjs --manifest ./artifacts/track5-production/track5-manifest.json --dataset-root ./datasets/sid-set/images --output-dir ./artifacts/track5-materialized --concurrency 4
```

Materialization generates clean and declared JPEG, blur, resize, noise, atomic-color, and crop corruption variants as lossless RGB PNGs and binds each to a SHA-256. Provide organizer hashes when available so overlap checking is enforced; details are in [data sources](docs/data-sources.md).

## Training, calibration, and reporting order

The RGB checkpoint is frozen rather than retrained. Preserve this order: source split and leakage audit; corruption materialization; frozen RGB scoring; signal-expert training; expert calibration and static-weight fitting on the source-disjoint fusion-training set; internal-validation-only candidate and threshold diagnostics; then a one-time sealed internal report. The organizer demonstration set remains evaluation-only throughout.

To reproduce the RGB component of the internal-validation generation after materialization:

```shell
python rgb_baseline_cli.py --manifest ./artifacts/track5-materialized/track5-materialized-manifest.json --dataset-root ./artifacts/track5-materialized --output-dir ./artifacts/rgb-baseline --resolution 384 --device auto --batch-size 8
```

The resulting cache and reports bind the materialized-manifest SHA-256, checkpoint revision, preprocessing version, and corruption implementation version. Treat every metric emitted by this command as **internal validation** unless its report explicitly says otherwise; do not compare or combine it with organizer results.

## Submission inference status

The learned-static-fusion design is bound to these trusted artifacts:

- generation `static-fallback-generation-v2-67220d1f7a2329f2c9d68d306fd77cd6a19125c66bd313be5d3c85e4bd19f181`;
- bundle `static-fallback-bundle-v2-7e7422a210136e62258ac62ae5dd8447803203d5b35d281fa5ec6da029187179`, SHA-256 `9c80b66553d10a4fc66f443c45672434800efb0731dfe2ea59036757ba959cd2`;
- RGB checkpoint SHA-256 `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387`;
- signal-model SHA-256 `cc1e98788ef09036c916065aca1d5b62751357d9eeaba90f50fe2532b9351ab5`.

The candidate-bound [submission evidence](docs/submission/evidence/submission-evidence.json) and its [completion receipt](docs/submission/evidence/submission-evidence.complete.json) produce the public [robustness and error report](docs/submission/results/robustness-and-errors.md), [clean-versus-transformed SVG](docs/submission/results/clean-vs-transformed.svg), and [report receipt](docs/submission/results/submission-report.complete.json). On the source-disjoint **internal validation** set, learned static fusion records clean AUROC 0.981975, mean transformed AUROC 0.9567506944444445, all-condition macro AUROC 0.9603541666666667, and its weakest persisted condition at noise / sigma-0.1 / AUROC 0.810425. These are development results, not an official organizer score.

The final portable directory-to-JSON command is **pending Issue #10 acceptance and final CLI binding**. Until that gate is supplied, `rgb_cli.py` is an RGB-expert component diagnostic and must not be presented as the learned-static-fusion submission system. The accepted command must emit a JSON array, sorted by stable relative image path, with exactly this per-image schema:

```json
{"image_path": "relative/path/to/image.png", "pred": 0.73}
```

`pred` is a finite probability in `[0, 1]` produced by learned static fusion; it is not a provenance verdict or an autonomous moderation decision. Save the output SHA-256 beside the accepted command, both expert checksums, bundle checksum, and candidate name before sharing a result. The [claim ledger](docs/submission/claim-ledger.md) is the release gate.

## Reproducing checks

Run the controlled Node fixture and project suites from a clean checkout:

```shell
npm run track5:fixture
npm run verify
```

The fixtures verify contracts, not benchmark performance. See [submission notes](docs/submission/devpost.md), [attributions](docs/submission/attributions.md), and the [demo script](docs/submission/demo-script.md) for the candidate-bound submission package.

## Limitations and improvements

The Community Forensics checkpoint is frozen and its public metadata cannot prove image-level exclusion of every organizer demonstration image. The organizer set therefore stays evaluation-only, and locally controlled sources receive exact and perceptual overlap checks. Learned static fusion does not adapt its trust per image and should not be treated as a deployment-ready moderation system. Final directory inference remains gated on Issue #10 acceptance and exact artifact validation. Further work includes independently evaluating organizer demonstration data without changing the selected system and studying robustness across unseen generators and transformations.

## Contributions

Contribution authorship must be confirmed by the human team before publication. Use the role-based record in [the claim ledger](docs/submission/claim-ledger.md) to map a person, their reviewed contribution, and approval date. This repository intentionally does not infer or assign individual contributions.
