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

The default training draw contains 40,320 observations with seed 61 and uses the finalized Issue-3 hierarchy:

```text
class → source → clean/corruption family → severity
```

The sampler first gives equal total allocation to both authenticity classes. Within a class, it distributes draws over sources so each of the 8,000 expert-training sources receives five or six draws. It then rotates equally through clean and the six corruption families and through the declared severities within each family. Thus a source with more available variants, or a family with more severity settings, cannot dominate merely because it has more manifest rows. If an identical variant is selected more than once, the plan stores one feature record with an integer `sample_weight`; normalization and gradient updates use that weight exactly as if the draw had been expanded.

The checkpoint is frozen at the epoch with the lowest `signal-condition-balanced-bce-v1` on the complete 40,000-observation internal-validation matrix. This score averages observation BCE within each severity, averages severities within each family, and finally gives equal weight to clean and each of the six corruption families. It therefore follows the same severity-first/family-macro principle as reporting metrics.

## Bounded materialization and fail-closed artifacts

The complete selected experiment represents about 202 GiB of uncompressed RGB, so the CLI does not materialize it all at once. It creates deterministic whole-source shards bounded by an uncompressed-byte budget (1 GiB by default), materializes one shard through the canonical harness, verifies declared Sharp/libvips versions, source pins, native RGB geometry, PNG encoding, materialized checksums, and all source/variant relationships, extracts and validates its feature cache, writes a separate completion receipt, and then evicts that shard's PNGs. A rerun reuses a shard only when the cache and receipt exactly bind the current plan, source set, variant set, feature records, runtime, and implementation snapshot and every source still matches its manifest SHA-256 pin. Missing or incomplete shards are freshly materialized and extracted; stale or incompatible caches and receipts are rejected.

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

Plan, shard, feature, normalization, checkpoint, and logit readers independently bind the manifest SHA-256, source and variant IDs, selected observation set, corruption and shared-preprocessing versions, materialized-image checksums and encoding, 26-feature order and resolution, normalization revision, and checkpoint revision. Normalization is bound only to the training plan and exact expert-training feature records; the checkpoint separately adds the exact internal-validation feature digest. Duplicate, missing, stale, non-finite, path-escaping, or incompatible relationships are rejected. The plan is staged before publication, artifacts are published atomically or compared with an existing exact result, and `signal-run.json` is written last as the completion marker. It records artifact hashes, implementation hashes, dependency/runtime versions, sample and source counts, shard bounds, seeds, selected epoch, and checkpoint-selection score without storing machine-specific paths.

Generated data, feature/logit caches, checkpoints, transient PNGs, and maps belong under ignored `artifacts/` or `datasets/`; none are committed.

## Clean installation and run

Python 3.12 and Node 22 are the supported production path. From the repository root on Windows PowerShell:

```powershell
py -3.12 -m venv artifacts/venvs/signal
& .\artifacts\venvs\signal\Scripts\python.exe -m pip install --requirement requirements-signal.txt
npm ci
```

`requirements-signal.txt` pins NumPy and Pillow; `package-lock.json` pins Sharp and the Node dependency tree. The active Sharp and libvips versions must exactly match the finalized manifest. Run the production experiment with paths appropriate to the local, ignored dataset checkout:

```powershell
& .\artifacts\venvs\signal\Scripts\python.exe signal_cli.py run `
  --manifest artifacts/track5-production/track5-manifest.json `
  --dataset-root datasets/sid-set/images `
  --output-dir artifacts/signal-production
```

The signal experiment is CPU-based and does not require a GPU. A real run can take substantial time because it must corrupt, losslessly encode, verify, decode, resize, and Fourier-transform 80,320 selected observations. Each completed feature shard receives a completion receipt bound to its exact plan, source set, feature records, runtime, and implementation snapshot. On retry, the runner re-hashes every pinned source used by a completed shard before reusing it and rematerializes only missing or incomplete shards; stale or incompatible caches and receipts fail closed. A cold run in a fresh output directory remains byte-identical. Use a fresh output directory for different seeds, hyperparameters, code, dependencies, or manifests.

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
