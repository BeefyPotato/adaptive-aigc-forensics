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
