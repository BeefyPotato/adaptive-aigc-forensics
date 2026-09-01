# Static fusion fallback

Issue #7 calibrates the frozen RGB expert and frozen `hackathon-v1` signal expert, then fits one non-adaptive fallback on the source-disjoint fusion-training set. It does not update either expert, RGB preprocessing, or signal normalization.

## Contracts

`fusion_cache_cli.py` processes bounded, class-balanced whole-source shards. Each shard resolves every corruption variant through the canonical corruption harness, scores the official 384 RGB checkpoint and frozen signal checkpoint, validates matched materialized-observation digests and revisions, publishes a cache plus completion receipt, and evicts temporary PNGs. A completed receipt is revalidated before reuse. The final caches contain all 40,000 fusion-training observations from 2,000 sources and the 8,000 observations from the exact 400-source Issue #6 internal-validation selection.

`fusion_pipeline.py` fits monotone bounded Platt calibrators with deterministic damped Newton steps. It fits a static RGB weight on a `[0,1]` grid with step `0.001`, minimizing condition-balanced Brier score on fusion-training; equal objective values choose the larger RGB weight. Raw logits are retained separately from calibrated logits.

The fallback-selection rule was frozen before matched validation: learned static fusion must improve all-condition macro AUROC over calibrated RGB-only by at least `0.005`, worsen Brier by no more than `0.002`, and have a source-bootstrap 95% AUROC-gain lower bound above zero. Otherwise calibrated RGB-only is selected.

## Acceptance result

The RGB calibrator slope/intercept are `0.481469358380229` / `2.725716958843671`; the signal calibrator values are `2.1908644107462774` / `0.002442450220552084`. The condition-balanced-Brier fit selected RGB weight `0.677` and signal weight `0.323`.

On identical matched internal-validation observations, calibrated RGB-only has all-condition macro AUROC `0.9435586309523808` and Brier `0.10886216953081583`. Learned static fusion has macro AUROC `0.9603541666666667` and Brier `0.09982875909583232`, a macro gain of `0.016795535714285936`; the deterministic 1,000-draw source-bootstrap 95% interval is `[0.011076105794972707, 0.0234869800759804]`. The learned fallback is therefore selected. Equal 50/50 calibrated-logit fusion remains the transparent control and records macro AUROC `0.9630342261904763` and Brier `0.10267819307128742`.

At each candidate's provisional internal-validation-only maximum-Youden-J threshold, the signal expert corrects 768 of 1,218 calibrated-RGB errors (`0.6305418719211823`). This is held-out complementary value, not merely an ensemble-gain claim. Thresholds remain provisional and no sealed internal test or organizer demonstration labels or metrics were accessed.

Generated archives, materialized observations, caches, and per-observation logits remain under ignored `artifacts/` paths. The minimal first-party inference package is tracked under `models/track5`; it contains the signal model, aggregate fusion bundle, and completion receipt, but no datasets, image paths, labels, or per-image logits.
