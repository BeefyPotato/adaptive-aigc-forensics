# Signal-only experiment

Issue #6 is a separate, leakage-safe experiment for the low-cost signal expert. It does not train fusion, a degradation gate, a calibrator, or the RGB expert.

## Pixel path and representation

The public experiment starts from the finalized Issue-3 recipe manifest and its existing SID_Set images. Each selected corruption variant is resolved by the shared Node corruption harness at decoded native resolution and encoded as a checksummed, lossless `track5-materialized-observations-v1` PNG. The signal branch then decodes that materialized observation through `shared_observation.py`, applies the shared 440-short-edge resize and 384 center crop, converts the resulting RGB values to luminance, and extracts features. Corruption therefore always happens before expert geometry, luminance, or signal extraction.

`signal-representation-v1` contains exactly 26 finite values in this literal order:

1. Sixteen Fourier values: mean log power in normalized radial bands 00 through 15 after luminance mean removal and a two-dimensional Hann window.
2. Six neighbour values: mean and standard deviation of absolute horizontal differences, mean and standard deviation of absolute vertical differences, and the two diagonal mean absolute differences.
3. Four residual values: mean absolute residual, residual standard deviation, central excess kurtosis, and pooled horizontal/vertical sign-change rate.

The residual is luminance minus the output of the fixed separable `[1, 4, 6, 4, 1] / 16` kernel. This residual kernel is an internal feature operator, not a sampled blur corruption.

## Partitions, balanced sampling, and checkpoint selection

Only `expert-training` records may fit normalization or update the 26→16→1 tanh MLP. Only source-disjoint `internal-validation` records may select the checkpoint, produce development metrics, or select the provisional threshold. `fusion-training`, sealed-test labels, and organizer demonstration data are rejected at the planning, training, cache, and evaluation seams.

The default, time-boxed Issue #6 profile is `hackathon-v1`. With seed 61 it contains 8,064 balanced expert-training draws and 400 deterministically ranked internal-validation sources: 200 per authenticity class, with the complete 20-condition matrix retained for every selected source. That produces 8,000 validation observations and 16,064 observations in total. The profile uses the finalized Issue-3 hierarchy:

```text
class → source → clean/corruption family → severity
```

The sampler first gives equal total allocation to both authenticity classes. Within a class, it distributes draws over sources so all 8,000 expert-training sources remain represented; in `hackathon-v1`, 64 sources receive a second draw. It then rotates equally through clean and the six corruption families and through the declared severities within each family. Thus a source with more available variants, or a family with more severity settings, cannot dominate merely because it has more manifest rows. If an identical variant is selected more than once, the plan stores one feature record with an integer `sample_weight`; normalization and gradient updates use that weight exactly as if the draw had been expanded.

The checkpoint is frozen at the epoch with the lowest `signal-condition-balanced-bce-v1` on the selected internal-validation matrix. This score averages observation BCE within each severity, averages severities within each family, and finally gives equal weight to clean and each of the six corruption families. It follows the same severity-first/family-macro principle as reporting metrics. The selected internal-validation sources are whole-source, class-balanced, deterministic from `validation_seed`, and source-disjoint from training.

`issue-6-full-v1` remains available for an optional post-hackathon replication with 40,320 training draws and all 2,000 internal-validation sources (40,000 observations). `custom-v1` exists for fixtures and diagnostics and is always marked `non-acceptance`. A profile's `acceptance_scope` is fixed and validated; relabelling a custom or time-boxed artifact as another scope fails closed.

## Bounded materialization and fail-closed artifacts

The production planner estimates about 40.40 GiB of uncompressed RGB for `hackathon-v1` and about 202 GiB for `issue-6-full-v1`. The CLI does not materialize either run all at once. It creates deterministic whole-source shards bounded by an uncompressed-byte budget (1 GiB by default), materializes one shard through the canonical harness, verifies declared Sharp/libvips versions, source pins, native RGB geometry, PNG encoding, materialized checksums, and all source/variant relationships, extracts and validates its feature cache, writes a separate completion receipt, and then evicts that shard's PNGs. A rerun reuses a shard only when the cache and receipt exactly bind the current plan, source set, variant set, feature records, runtime, and implementation snapshot and every source still matches its manifest SHA-256 pin. Missing or incomplete shards are freshly materialized and extracted; stale or incompatible caches and receipts are rejected.

The final output directory contains:

- `signal-plan.json`
- `signal-plan.shards/`
- `signal-feature-shards/`
- `signal-training-features.json`
- `signal-validation-features.json`
- `signal-normalization.json`
- `signal-model.json`
- `signal-validation-logits.json`
- `signal-internal-validation-metrics.json`
- `signal-run.json`

Plan, shard, feature, normalization, checkpoint, and logit readers independently bind the experiment profile and scope, manifest SHA-256, source and variant IDs, selected observation set, corruption and shared-preprocessing versions, materialized-image checksums and encoding, 26-feature order and resolution, normalization revision, and checkpoint revision. Normalization is bound only to the training plan and exact expert-training feature records; the checkpoint separately adds the exact internal-validation feature digest. Duplicate, missing, stale, non-finite, path-escaping, re-labelled, or incompatible relationships are rejected. The plan is staged before publication, artifacts are published atomically or compared with an existing exact result, and `signal-run.json` is written last as the completion marker. It records artifact hashes, implementation hashes, dependency/runtime versions, sample and source counts, shard bounds, seeds, selected epoch, and checkpoint-selection score without storing machine-specific paths.

Generated data, feature/logit caches, checkpoints, transient PNGs, and maps belong under ignored `artifacts/` or `datasets/`; none are committed.

## Clean installation and run

Python 3.12 and Node 22 are the supported production path. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv artifacts/venvs/signal
& .\artifacts\venvs\signal\Scripts\python.exe -m pip install --requirement requirements-signal.txt
npm ci
```

`requirements-signal.txt` pins NumPy and Pillow; `package-lock.json` pins Sharp and the Node dependency tree. The active Sharp and libvips versions must exactly match the finalized manifest. Run the time-boxed production experiment with paths appropriate to the local, ignored dataset checkout:

```powershell
& .\artifacts\venvs\signal\Scripts\python.exe signal_cli.py run `
  --manifest artifacts/track5-production/track5-manifest.json `
  --dataset-root datasets/sid-set/images `
  --output-dir artifacts/signal-hackathon `
  --experiment-profile hackathon-v1
```

For the optional full replication, use a fresh output directory and `--experiment-profile issue-6-full-v1`.

The signal experiment is CPU-based and does not require a GPU. Native-resolution seeded RGB noise is split across bounded worker threads while preserving the exact pre-optimization byte stream for every worker count. On a representative 1024-by-680 SID_Set image, five workers reduced the measured noise step from 5.34 seconds to 1.28 seconds (4.17x) with identical SHA-256 output. Actual end-to-end time also includes decoding, other corruptions, checksummed lossless PNG materialization, verification, shared geometry, and Fourier extraction.

Each completed feature shard receives a completion receipt bound to its exact plan, source set, feature records, runtime, and implementation snapshot. On retry, the runner re-hashes every pinned source used by a completed shard before reusing it and rematerializes only missing or incomplete shards; stale or incompatible caches and receipts fail closed. A cold run in a fresh output directory remains byte-identical. Use a fresh output directory for different profiles, seeds, hyperparameters, code, dependencies, or manifests.

## Verified `hackathon-v1` production result

The production run completed on 2026-08-31 from Issue-3 manifest SHA-256 `c9ea2d3b616b37844d21602a95e5f90c824a692ad609d31bbe0b982c5f45228a` and signal plan SHA-256 `a95547bfe46937cd769c9fa0c020a99a7abb6b71e0b1672258084e6382ab71ae`. It used Python 3.12.10, NumPy 2.3.5, Pillow 12.0.0, Node 24.16.0, Sharp 0.35.4, and libvips 8.18.6.

The run processed 8,064 expert-training draws over all 8,000 training sources and 8,000 internal-validation observations over 400 sources. It produced 42 validated completion receipts, retained no materialized PNGs, took 2:29:34 wall time from a cold output directory, and retained 117.98 MiB of feature/model/provenance artifacts. A second invocation rehashed every pinned source, accepted all 42 caches, recomputed the training result, and reproduced the exact artifact hashes.

The frozen checkpoint is epoch 199, revision `signal-checkpoint-v1-4a6b4d974722c9f8729a90d872387bb49e54d01e7bda98ddb69232c28604390e`, with normalization revision `signal-normalization-v1-25b16b78f7ecb5e02572e03650537e8b5e266f2f3e49a911a2ae2e2e11d45e80`. Its condition-balanced checkpoint-selection BCE is 0.5582595402644148. Canonical internal-validation results are:

- Clean AUROC: 0.85945.
- Mean corrupted AUROC: 0.8579222222222223.
- All-condition macro AUROC: 0.8581404761904762.
- Worst family/severity: color / brightness-1.2, AUROC 0.841375.
- Degradation retention: 0.998222377360198.

The maximum-Youden-J threshold remains explicitly provisional and internal-validation-only; it is not a sealed-test or organizer-derived decision.

## Metrics and diagnostic maps

Signal logits are evaluated by the shared canonical evaluator. It reports clean AUROC, AUROC for every declared severity, severity-first family AUROC, mean corrupted AUROC across the six families, all-condition macro AUROC across clean plus those six family values, the worst family/severity, degradation drop and retention, and a provisional maximum-Youden-J threshold selected only on internal validation.

Diagnostic maps are regenerable evidence, never model inputs or cache fields. After producing a bounded canonical materialized shard, render one verified variant with:

```powershell
& .\artifacts\venvs\signal\Scripts\python.exe signal_cli.py render-maps `
  --manifest artifacts/track5-production/track5-manifest.json `
  --materialized-manifest artifacts/one-signal-shard/track5-materialized-manifest.json `
  --variant-id <variant-id> `
  --output-dir artifacts/signal-maps/<variant-id>
```

The command verifies the parent recipe, materialized manifest, selected relationship, contained path, and materialized checksum before using the shared geometry. It writes deterministic 16-bit luminance, Fourier log-spectrum, neighbour high-pass, and residual PNGs plus `signal-diagnostic-maps.json`, which records fixed display scaling, the 26 features, provenance, and artifact hashes. The high-pass map is a neighbour-gradient magnitude and is intentionally distinct from the fixed-kernel residual.
