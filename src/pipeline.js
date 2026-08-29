import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";

import {
  ContractError,
  ExperimentConfig,
  cacheKey,
  readCacheArtifact,
  readModelBundle,
  validatePredictionRecords,
} from "./contracts.js";
import { evaluateFixture } from "./evaluation.js";
import {
  equalLogitFusion,
  fixtureRgbExpert,
  fixtureSignalExpert,
} from "./fixture-experts.js";
import { prepareExpertInputs } from "./images.js";
import { resolveManifest } from "./manifest.js";

function writeJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function requireFiniteExpertOutputs(variantId, outputs) {
  for (const [name, value] of Object.entries(outputs)) {
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new ContractError(`${name} for ${variantId} must be finite.`);
    }
  }
}

export function runFixture(
  configPath,
  outputDirectory,
  {
    rgbExpert = fixtureRgbExpert,
    signalExpert = fixtureSignalExpert,
    fusion = equalLogitFusion,
  } = {},
) {
  const absoluteConfigPath = resolve(configPath);
  const configDirectory = dirname(absoluteConfigPath);
  const config = ExperimentConfig.read(absoluteConfigPath);
  const modelBundle = readModelBundle(resolve(configDirectory, config.modelBundlePath), config);
  const manifest = resolveManifest(resolve(configDirectory, config.manifestPath), config);
  const absoluteOutputDirectory = resolve(outputDirectory);
  const cacheDirectory = join(absoluteOutputDirectory, "cache");
  mkdirSync(cacheDirectory, { recursive: true });

  const cacheRecords = [];
  const evaluatedRecords = [];
  for (const variant of manifest.variants) {
    const imagePath = resolve(configDirectory, variant.image_path.split("/").join(sep));
    const inputs = prepareExpertInputs(
      imagePath,
      variant.corruption_parameters,
      variant.corruption_seed,
    );
    const rgbLogit = rgbExpert(inputs.rgb);
    const signalLogit = signalExpert(inputs);
    const pred = fusion(rgbLogit, signalLogit);
    requireFiniteExpertOutputs(variant.variant_id, {
      "rgb expert logit": rgbLogit,
      "signal expert logit": signalLogit,
      "fused prediction": pred,
    });

    const artifact = {
      artifact_schema_version: config.artifactSchemaVersion,
      preprocessing_version: config.preprocessingVersion,
      checkpoint_revision: config.checkpointRevision,
      signal_representation_version: config.signalRepresentationVersion,
      variant_id: variant.variant_id,
      artifact_kind: "expert-logits",
      cache_key: cacheKey(config, variant.variant_id, "expert-logits"),
      rgb_logit: rgbLogit,
      signal_logit: signalLogit,
      fusion_version: modelBundle.fusionVersion,
      pred,
    };
    const cachePath = join(cacheDirectory, `${artifact.cache_key}.json`);
    writeJson(cachePath, artifact);
    readCacheArtifact(cachePath, config, modelBundle);
    cacheRecords.push(artifact);
    evaluatedRecords.push({
      variant_id: variant.variant_id,
      image_path: variant.image_path,
      authenticity_label: variant.authenticity_label,
      pred,
    });
  }

  const predictions = evaluatedRecords.map(({ image_path: imagePath, pred }) => ({
    image_path: imagePath,
    pred,
  }));
  validatePredictionRecords(predictions);
  const metrics = evaluateFixture(evaluatedRecords, config, modelBundle);

  mkdirSync(absoluteOutputDirectory, { recursive: true });
  writeJson(join(absoluteOutputDirectory, "resolved_manifest.json"), manifest);
  writeJson(join(absoluteOutputDirectory, "cache.json"), cacheRecords);
  writeJson(join(absoluteOutputDirectory, "predictions.json"), predictions);
  writeJson(join(absoluteOutputDirectory, "metrics.json"), metrics);
  return Object.freeze({
    outputDirectory: absoluteOutputDirectory,
    predictionCount: predictions.length,
    numericTolerance: modelBundle.numericTolerance,
  });
}
