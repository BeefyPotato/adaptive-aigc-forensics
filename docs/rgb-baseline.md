# RGB robustness baseline

Issue #5 turns the frozen Community Forensics RGB expert into a reproducible Track 5 experiment. The runner consumes `track5-manifest-v1`, scores observations from the three development partitions and optionally the sealed internal test, and writes:

- `rgb-logits.json`: versioned logits and runtime profile for calibration and fusion;
- `rgb-internal-validation-metrics.json`: clean and corruption-family robustness metrics;
- `rgb-rerun-check.json`: a deterministic subset comparison using the checkpoint tolerance.

The cache binds every result to the manifest SHA-256, manifest schema, checkpoint revision, preprocessing version, and corruption implementation version. A reader rejects stale metadata before downstream use. Sealed internal test cache records deliberately omit authenticity labels, and the metric seam accepts only internal-validation records.

## Preparing observations

The corruption harness owns image transformations. A recipe-only Track 5 manifest is deliberately rejected so clean source bytes cannot be mislabeled as corruption results. First run the materialization command documented in [track5-manifest.md](track5-manifest.md). It produces the lossless shared RGB observations and a `track5-materialized-manifest.json` containing checksummed, contained paths.

## GPU run

Install the pinned RGB dependencies and run the primary checkpoint:

```shell
python rgb_baseline_cli.py --manifest ./artifacts/track5-materialized/track5-materialized-manifest.json --dataset-root ./artifacts/track5-materialized --output-dir ./artifacts/rgb-baseline --device cuda --batch-size 16
```

The profile records wall-clock seconds, throughput, batch size, peak allocated GPU bytes, device, precision, failure count, and retry count. Pass `--retry-once` only when a failed batch should be retried once and recorded; completed batches are retained in memory rather than recomputed.

Internal-validation output reports clean AUROC, severity-averaged AUROC for all six corruption families, mean corrupted AUROC, all-condition macro AUROC, worst family/severity, degradation drop and retention. Threshold diagnostics are explicitly provisional and choose a maximum-Youden-J threshold using internal validation only.

The checked-in tests use a deterministic backend and do not download a checkpoint:

```shell
python -m unittest tests/test_rgb_baseline.py
```
