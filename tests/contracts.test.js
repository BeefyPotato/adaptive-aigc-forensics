import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import {
  ContractError,
  ExperimentConfig,
  cacheKey,
  readCacheArtifact,
  readModelBundle,
  validatePredictionRecords,
  variantIdentifier,
} from "../src/contracts.js";

const config = ExperimentConfig.from({
  config_schema_version: "experiment-config-v1",
  dataset_revision: "fixture-dataset-v1",
  split_seed: 17,
  corruption_seed: 23,
  preprocessing_version: "shared-preprocessing-v1",
  checkpoint_revision: "fixture-rgb-v1",
  artifact_schema_version: "artifact-v1",
  signal_representation_version: "fixture-signal-v1",
  manifest_path: "manifest.json",
  model_bundle_path: "model_bundle.json",
  numeric_tolerance: 1e-12,
});

test("variant identifiers and cache keys are deterministic", () => {
  const identity = {
    sourceId: "source-authentic-001",
    conditionFamily: "noise",
    corruptionParameters: { sigma: 0.02 },
    corruptionSeed: config.corruptionSeed,
    preprocessingVersion: config.preprocessingVersion,
    artifactSchemaVersion: config.artifactSchemaVersion,
  };
  const firstVariant = variantIdentifier(identity);

  assert.equal(
    firstVariant,
    "variant-v1-188d164a907adaf7558e693496d8b57ce7654c6a0485bc6f8e2d31cd6ed1122a",
  );
  assert.equal(firstVariant, variantIdentifier(identity));
  assert.equal(
    cacheKey(config, firstVariant, "expert-logits"),
    "cache-v1-7197a278aa35f318e880c380c50e0d2abcb01107698620b4739cd7ed78c203a3",
  );
  assert.notEqual(
    firstVariant,
    variantIdentifier({ ...identity, corruptionSeed: config.corruptionSeed + 1 }),
  );
});

test("model-bundle reader reports missing and incompatible metadata", () => {
  const directory = mkdtempSync(join(tmpdir(), "aigc-contracts-"));
  const path = join(directory, "bundle.json");
  const completeBundle = {
    bundle_schema_version: "model-bundle-v1",
    artifact_schema_version: "artifact-v1",
    preprocessing_version: "shared-preprocessing-v1",
    checkpoint_revision: "fixture-rgb-v1",
    signal_representation_version: "fixture-signal-v1",
    fusion_version: "equal-logit-v1",
    numeric_tolerance: 1e-12,
  };
  writeFileSync(path, JSON.stringify(completeBundle));
  assert.equal(readModelBundle(path, config).checkpointRevision, "fixture-rgb-v1");

  delete completeBundle.checkpoint_revision;
  writeFileSync(path, JSON.stringify(completeBundle));
  assert.throws(() => readModelBundle(path, config), ContractError);
  assert.throws(() => readModelBundle(path, config), /checkpoint_revision/);

  completeBundle.checkpoint_revision = "wrong-revision";
  writeFileSync(path, JSON.stringify(completeBundle));
  assert.throws(
    () => readModelBundle(path, config),
    /checkpoint_revision.*wrong-revision.*fixture-rgb-v1/,
  );

  completeBundle.checkpoint_revision = "fixture-rgb-v1";
  completeBundle.fusion_version = "unknown-fusion-v1";
  writeFileSync(path, JSON.stringify(completeBundle));
  assert.throws(
    () => readModelBundle(path, config),
    /fusion_version.*unknown-fusion-v1.*equal-logit-v1/,
  );
});

test("cache reader rejects incomplete or stale metadata", () => {
  const directory = mkdtempSync(join(tmpdir(), "aigc-cache-"));
  const path = join(directory, "cache.json");
  const artifact = {
    artifact_schema_version: "artifact-v1",
    preprocessing_version: "shared-preprocessing-v1",
    checkpoint_revision: "fixture-rgb-v1",
    signal_representation_version: "fixture-signal-v1",
    variant_id: "variant-v1-example",
    artifact_kind: "expert-logits",
    cache_key: cacheKey(config, "variant-v1-example", "expert-logits"),
    rgb_logit: 0.25,
    signal_logit: -0.5,
    fusion_version: "equal-logit-v1",
    pred: 0.4,
  };
  const modelBundle = { fusionVersion: "equal-logit-v1" };
  writeFileSync(path, JSON.stringify(artifact));
  assert.equal(readCacheArtifact(path, config, modelBundle).variant_id, "variant-v1-example");

  artifact.cache_key = "cache-v1-stale";
  writeFileSync(path, JSON.stringify(artifact));
  assert.throws(
    () => readCacheArtifact(path, config, modelBundle),
    /cache_key.*cache-v1-stale.*cache-v1-/,
  );
  artifact.cache_key = cacheKey(config, "variant-v1-example", "expert-logits");

  artifact.artifact_kind = "unknown-kind";
  writeFileSync(path, JSON.stringify(artifact));
  assert.throws(
    () => readCacheArtifact(path, config, modelBundle),
    /artifact_kind.*unknown-kind.*expert-logits/,
  );
  artifact.artifact_kind = "expert-logits";

  artifact.fusion_version = "unknown-fusion-v1";
  writeFileSync(path, JSON.stringify(artifact));
  assert.throws(
    () => readCacheArtifact(path, config, modelBundle),
    /fusion_version.*unknown-fusion-v1.*equal-logit-v1/,
  );
  artifact.fusion_version = "equal-logit-v1";

  artifact.artifact_schema_version = "artifact-v0";
  writeFileSync(path, JSON.stringify(artifact));
  assert.throws(
    () => readCacheArtifact(path, config, modelBundle),
    /artifact_schema_version.*artifact-v0.*artifact-v1/,
  );
  artifact.artifact_schema_version = "artifact-v1";

  delete artifact.pred;
  writeFileSync(path, JSON.stringify(artifact));
  assert.throws(() => readCacheArtifact(path, config, modelBundle), /pred/);
});

test("prediction records contain exactly image_path and a finite unit-interval pred", () => {
  assert.doesNotThrow(() =>
    validatePredictionRecords([
      { image_path: "authentic.ppm", pred: 0 },
      { image_path: "synthetic.ppm", pred: 1 },
    ]),
  );
  assert.throws(
    () => validatePredictionRecords([{ image_path: "image.ppm", pred: 0.5, label: 1 }]),
    /exactly.*image_path.*pred/,
  );
  assert.throws(
    () =>
      validatePredictionRecords([
        { image_path: "z.ppm", pred: 0.5 },
        { image_path: "a.ppm", pred: 0.5 },
      ]),
    /ordered by image_path/,
  );
  for (const pred of [-0.1, 1.1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.throws(
      () => validatePredictionRecords([{ image_path: "image.ppm", pred }]),
      /finite number in \[0, 1\]/,
    );
  }
});
