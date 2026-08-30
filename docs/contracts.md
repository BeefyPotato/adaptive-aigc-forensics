# Experiment and submission contracts

The issue #2 fixture defines the minimum versioned records exchanged by the RGB, signal, fusion, evaluation, and submission workstreams. Contract readers fail before inference when required metadata is missing or incompatible.

## Experiment configuration

`experiment-config-v1` requires:

| Field | Purpose |
| --- | --- |
| `dataset_revision` | Immutable identifier for the fixture or controlled dataset snapshot. |
| `split_seed` | Seed governing source-level partition selection. |
| `corruption_seed` | Root seed from which corruption-variant seeds are derived. |
| `preprocessing_version` | Shared decode, corruption, and geometry contract. |
| `checkpoint_revision` | Exact RGB expert revision expected by caches and the model bundle. |
| `artifact_schema_version` | Cache and output compatibility version. |
| `signal_representation_version` | Signal representation compatibility version. |
| `manifest_path` | Manifest path relative to the configuration file. |
| `model_bundle_path` | Model-bundle metadata path relative to the configuration file. |
| `numeric_tolerance` | Maximum accepted cross-machine prediction difference. |

The checked-in example is [fixtures/experiment/config.json](../fixtures/experiment/config.json).

## Source and corruption-variant manifest

`manifest-v1` contains source records and corruption-variant recipes. A source record owns its path, binary authenticity label, split, dataset, and dataset revision. A variant recipe identifies its source, condition family, corruption parameters, and deterministic seed offset.

The resolved manifest records the explicit corruption seed and a `variant-v1-<sha256>` identifier. The digest is computed from canonical JSON containing source identity, condition family, parameters, seed, preprocessing version, and artifact schema version. This keeps all observations derived from a source joinable without machine-specific paths.

The Track 5 production manifest is a recipe contract: its `image_path` identifies the source image, not already-corrupted bytes. Before either production expert runs, the recipe is resolved into `track5-materialized-observations-v1`. Each materialized observation retains its source/variant identity and adds a contained relative PNG path, SHA-256, and lossless encoding version. Both branches decode that same artifact. `shared_observation_preprocessing_version` describes decode/orientation/corruption materialization; `rgb_preprocessing_version` separately describes the RGB checkpoint resize, crop, tensor conversion, and normalization.

Issue #2 intentionally implements only the checked-in `noise` corruption fixture. The complete corruption harness belongs to later tickets.

## Cache artifacts

Each `artifact-v1` expert-logit cache record contains:

- artifact, preprocessing, checkpoint, and signal-representation versions;
- `variant_id`, `artifact_kind`, and a deterministic `cache-v1-<sha256>` key;
- finite `rgb_logit` and `signal_logit` values;
- the fixture fusion version and fused probability.

The cache reader validates each record against the experiment configuration and validated model bundle. It rejects missing fields, version mismatches, unsupported artifact kinds, and keys that do not match the declared variant and artifact kind, with the expected and received values in the error. Cache keys include every configuration revision that can change an expert result.

## Model-bundle metadata

`model-bundle-v1` declares the artifact, preprocessing, checkpoint, signal-representation, and fusion versions plus numeric tolerance. Its reader rejects incomplete metadata, version mismatches, unsupported fusion behavior, and a tolerance looser than the experiment configuration permits.

The fixture bundle contains metadata and deterministic toy expert behavior only. Real weights, normalization statistics, calibration parameters, and the selected production fusion mechanism will extend the bundle in their owning tickets.

## Prediction output

Prediction output is a JSON array sorted by stable relative path. Each record contains exactly:

```json
{
  "image_path": "images/example.png",
  "pred": 0.5
}
```

`image_path` must be non-empty and unique. `pred` is the finite `[0, 1]` likelihood that the image is AI-generated. Labels, logits, thresholds, errors, and provenance belong in separate artifacts and must not be added to submission records.

## Fixture outputs

`metric-v1` stores deterministic per-variant prediction records plus fixture accuracy and Brier score. These values prove that evaluation consumed the same predictions written through the submission contract; they are smoke metrics and are not robustness claims.
