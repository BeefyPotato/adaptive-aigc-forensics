# Signal expert

Issue #6 adds a deterministic, low-cost signal expert that consumes the same corrupted RGB observation and shared checkpoint crop as the RGB expert. Corruption happens first. Both branches apply the common resize/center-crop geometry; the RGB branch then applies checkpoint normalization, while this branch converts the shared crop to luminance in `[0, 1]`. Native dimensions remain available as materialization metadata for later quality evidence.

## Signal representation

`signal-representation-v1` always contains 26 finite values in this order:

1. **16 Fourier values:** mean log power in radial bands 0 through 15 after mean removal and a two-dimensional Hann window.
2. **6 neighbour values:** mean and standard deviation of absolute horizontal differences, mean and standard deviation of absolute vertical differences, and mean absolute differences for both diagonals.
3. **4 residual values:** mean absolute residual, residual standard deviation, excess kurtosis, and horizontal/vertical sign-change rate.

The residual is luminance minus a fixed separable `[1, 4, 6, 4, 1] / 16` smooth. This **residual kernel is an internal feature operator**, not a sampled Gaussian-blur corruption. `extract_signal_representation(..., include_maps=True)` optionally returns luminance, log spectrum, high-pass, and residual maps for debugging or figures. Maps are never required by inference or stored in feature caches.

## Leakage controls and training

Normalization accepts only records marked `expert-training` and stores the manifest schema, immutable dataset revision/hash metadata, source count, feature order/version, means, and scales. The deterministic 26→16→1 tanh MLP accepts only `expert-training` records for weight updates, requires source-disjoint `internal-validation` records for checkpoint selection, and rejects fusion-training, sealed internal test, or organizer demonstration records at these seams.

Prepare feature JSON with top-level `manifest_metadata` and `records`, using the balanced Track 5 sampler for the training records, then run:

```shell
python signal_cli.py --training-features artifacts/signal/train.json --validation-features artifacts/signal/validation.json --output artifacts/signal/model.json
```

Feature/logit caches retain all 26 features, source and variant IDs, representation/checkpoint/manifest metadata, and a deterministic cache key. Readers reject incompatible dimensions or stale metadata. `evaluate_signal_only` reports internal-validation AUROC separately for each corruption family so later fusion work can measure complementary value against RGB results.
