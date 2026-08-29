import {
  mkdirSync,
  readFileSync,
  writeFileSync,
} from "node:fs";
import { dirname, isAbsolute, join, resolve, sep } from "node:path";

import {
  ContractError,
  ExperimentConfig,
  cacheKey,
  readCacheArtifact,
  readModelBundle,
  validatePredictionRecords,
  variantIdentifier,
} from "./contracts.js";
import { prepareExpertInputs } from "./images.js";

const MANIFEST_VERSION = "manifest-v1";
const METRIC_VERSION = "metric-v1";

function readJson(path, contractName) {
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new ContractError(`${contractName} could not be read from ${path}: ${error.message}`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ContractError(`${contractName} at ${path} is not valid JSON: ${error.message}`);
  }
}

function requireFields(record, fields, contractName) {
  if (record === null || typeof record !== "object" || Array.isArray(record)) {
    throw new ContractError(`${contractName} must be a JSON object.`);
  }
  const missing = fields.filter((field) => !(field in record));
  if (missing.length > 0) {
    throw new ContractError(`${contractName} is missing required field(s): ${missing.join(", ")}.`);
  }
}

function validateRelativeImagePath(imagePath, contractName) {
  if (typeof imagePath !== "string" || imagePath.length === 0 || isAbsolute(imagePath)) {
    throw new ContractError(`${contractName}.image_path must be a non-empty relative path.`);
  }
  const segments = imagePath.replaceAll("\\", "/").split("/");
  if (segments.includes("..")) {
    throw new ContractError(`${contractName}.image_path must stay inside the manifest directory.`);
  }
}

export function resolveManifest(path, config) {
  const value = readJson(path, "experiment manifest");
  requireFields(value, ["manifest_schema_version", "sources", "variants"], "experiment manifest");
  if (value.manifest_schema_version !== MANIFEST_VERSION) {
    throw new ContractError(
      `experiment manifest.manifest_schema_version is incompatible: received ${JSON.stringify(value.manifest_schema_version)}; expected ${JSON.stringify(MANIFEST_VERSION)}.`,
    );
  }
  if (!Array.isArray(value.sources) || !Array.isArray(value.variants)) {
    throw new ContractError("experiment manifest.sources and experiment manifest.variants must be arrays.");
  }

  const sources = value.sources.map((source, index) => {
    const contractName = `experiment manifest source ${index}`;
    requireFields(
      source,
      [
        "source_id",
        "image_path",
        "authenticity_label",
        "split",
        "dataset",
        "dataset_revision",
      ],
      contractName,
    );
    for (const field of ["source_id", "split", "dataset", "dataset_revision"]) {
      if (typeof source[field] !== "string" || source[field].length === 0) {
        throw new ContractError(`${contractName}.${field} must be a non-empty string.`);
      }
    }
    validateRelativeImagePath(source.image_path, contractName);
    if (source.authenticity_label !== 0 && source.authenticity_label !== 1) {
      throw new ContractError(`${contractName}.authenticity_label must be 0 or 1.`);
    }
    if (source.dataset_revision !== config.datasetRevision) {
      throw new ContractError(
        `${contractName}.dataset_revision is incompatible: received ${JSON.stringify(source.dataset_revision)}; expected ${JSON.stringify(config.datasetRevision)}.`,
      );
    }
    return Object.freeze({ ...source });
  });
  const sourceById = new Map();
  for (const source of sources) {
    if (sourceById.has(source.source_id)) {
      throw new ContractError(`experiment manifest repeats source_id ${JSON.stringify(source.source_id)}.`);
    }
    sourceById.set(source.source_id, source);
  }

  const variants = value.variants.map((variant, index) => {
    const contractName = `experiment manifest variant ${index}`;
    requireFields(
      variant,
      [
        "source_id",
        "condition_family",
        "corruption_parameters",
        "corruption_seed_offset",
      ],
      contractName,
    );
    const source = sourceById.get(variant.source_id);
    if (!source) {
      throw new ContractError(
        `${contractName}.source_id ${JSON.stringify(variant.source_id)} has no source record.`,
      );
    }
    if (variant.condition_family !== "noise") {
      throw new ContractError(
        `${contractName}.condition_family ${JSON.stringify(variant.condition_family)} is unsupported by the issue #2 fixture; expected "noise".`,
      );
    }
    requireFields(variant.corruption_parameters, ["sigma"], `${contractName}.corruption_parameters`);
    if (!Number.isSafeInteger(variant.corruption_seed_offset) || variant.corruption_seed_offset < 0) {
      throw new ContractError(`${contractName}.corruption_seed_offset must be a non-negative safe integer.`);
    }
    const corruptionSeed = config.corruptionSeed + variant.corruption_seed_offset;
    if (!Number.isSafeInteger(corruptionSeed)) {
      throw new ContractError(`${contractName} produces a corruption seed outside the safe integer range.`);
    }
    const variantId = variantIdentifier({
      sourceId: source.source_id,
      conditionFamily: variant.condition_family,
      corruptionParameters: variant.corruption_parameters,
      corruptionSeed,
      preprocessingVersion: config.preprocessingVersion,
      artifactSchemaVersion: config.artifactSchemaVersion,
    });
    return Object.freeze({
      variant_id: variantId,
      source_id: source.source_id,
      image_path: source.image_path.replaceAll("\\", "/"),
      authenticity_label: source.authenticity_label,
      condition_family: variant.condition_family,
      corruption_parameters: Object.freeze({ ...variant.corruption_parameters }),
      corruption_seed: corruptionSeed,
      preprocessing_version: config.preprocessingVersion,
      artifact_schema_version: config.artifactSchemaVersion,
    });
  });
  variants.sort((left, right) => {
    if (left.image_path < right.image_path) return -1;
    if (left.image_path > right.image_path) return 1;
    return 0;
  });

  const seenVariantIds = new Set();
  for (const variant of variants) {
    if (seenVariantIds.has(variant.variant_id)) {
      throw new ContractError(`experiment manifest repeats variant_id ${variant.variant_id}.`);
    }
    seenVariantIds.add(variant.variant_id);
  }

  return Object.freeze({
    manifest_schema_version: MANIFEST_VERSION,
    dataset_revision: config.datasetRevision,
    split_seed: config.splitSeed,
    corruption_seed: config.corruptionSeed,
    preprocessing_version: config.preprocessingVersion,
    checkpoint_revision: config.checkpointRevision,
    artifact_schema_version: config.artifactSchemaVersion,
    signal_feature_version: config.signalFeatureVersion,
    sources,
    variants,
  });
}

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function fixtureRgbExpert(rgb) {
  const brightness = mean(rgb.map(([red, green, blue]) => (red + green + blue) / 3));
  return (brightness - 0.5) * 4;
}

export function fixtureSignalExpert({ luminance, width, height }) {
  const differences = [];
  for (let row = 0; row < height; row += 1) {
    for (let column = 0; column < width; column += 1) {
      const index = row * width + column;
      if (column + 1 < width) {
        differences.push(Math.abs(luminance[index] - luminance[index + 1]));
      }
      if (row + 1 < height) {
        differences.push(Math.abs(luminance[index] - luminance[index + width]));
      }
    }
  }
  return (mean(differences) - 0.15) * 8;
}

function sigmoid(logit) {
  if (logit >= 0) {
    return 1 / (1 + Math.exp(-logit));
  }
  const exponential = Math.exp(logit);
  return exponential / (1 + exponential);
}

export function equalLogitFusion(rgbLogit, signalLogit) {
  return sigmoid((rgbLogit + signalLogit) / 2);
}

function writeJson(path, value) {
  writeFileSync(path, `${JSON.stringify(value, null, 2)}\n`, "utf8");
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
    for (const [name, value] of [
      ["rgb expert logit", rgbLogit],
      ["signal expert logit", signalLogit],
      ["fused prediction", pred],
    ]) {
      if (typeof value !== "number" || !Number.isFinite(value)) {
        throw new ContractError(`${name} for ${variant.variant_id} must be finite.`);
      }
    }

    const artifact = {
      artifact_schema_version: config.artifactSchemaVersion,
      preprocessing_version: config.preprocessingVersion,
      checkpoint_revision: config.checkpointRevision,
      signal_feature_version: config.signalFeatureVersion,
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
    readCacheArtifact(cachePath, config);
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
  const threshold = 0.5;
  const correct = evaluatedRecords.filter(
    ({ authenticity_label: label, pred }) => Number(pred >= threshold) === label,
  ).length;
  const brierScore = mean(
    evaluatedRecords.map(({ authenticity_label: label, pred }) => (pred - label) ** 2),
  );
  const metrics = {
    metric_schema_version: METRIC_VERSION,
    artifact_schema_version: config.artifactSchemaVersion,
    prediction_count: evaluatedRecords.length,
    threshold,
    accuracy: correct / evaluatedRecords.length,
    brier_score: brierScore,
    numeric_tolerance: modelBundle.numericTolerance,
    records: evaluatedRecords.map(({ variant_id: variantId, authenticity_label: label, pred }) => ({
      variant_id: variantId,
      authenticity_label: label,
      pred,
    })),
  };

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
