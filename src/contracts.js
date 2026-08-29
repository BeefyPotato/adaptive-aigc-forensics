import { createHash } from "node:crypto";

import {
  ContractError,
  assertCompatible,
  readJson,
  requireFields,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
  requirePositiveFiniteNumber,
} from "./contract-validation.js";

export { ContractError } from "./contract-validation.js";

const EXPERIMENT_CONFIG_VERSION = "experiment-config-v1";
const MODEL_BUNDLE_VERSION = "model-bundle-v1";
const SUPPORTED_ARTIFACT_KIND = "expert-logits";
const SUPPORTED_FUSION_VERSION = "equal-logit-v1";

function canonicalize(value) {
  if (Array.isArray(value)) {
    return value.map(canonicalize);
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.keys(value)
        .sort()
        .map((key) => [key, canonicalize(value[key])]),
    );
  }
  if (typeof value === "number" && !Number.isFinite(value)) {
    throw new ContractError("Canonical contract values must contain only finite numbers.");
  }
  return value;
}

export function canonicalJson(value) {
  return JSON.stringify(canonicalize(value));
}

function hashIdentifier(prefix, value) {
  const digest = createHash("sha256").update(canonicalJson(value), "utf8").digest("hex");
  return `${prefix}-${digest}`;
}

export class ExperimentConfig {
  static requiredFields = [
    "config_schema_version",
    "dataset_revision",
    "split_seed",
    "corruption_seed",
    "preprocessing_version",
    "checkpoint_revision",
    "artifact_schema_version",
    "signal_representation_version",
    "manifest_path",
    "model_bundle_path",
    "numeric_tolerance",
  ];

  static from(mapping) {
    const value = requireObject(mapping, "experiment configuration");
    requireFields(value, ExperimentConfig.requiredFields, "experiment configuration");
    for (const field of [
      "config_schema_version",
      "dataset_revision",
      "preprocessing_version",
      "checkpoint_revision",
      "artifact_schema_version",
      "signal_representation_version",
      "manifest_path",
      "model_bundle_path",
    ]) {
      requireNonemptyString(value[field], field, "experiment configuration");
    }
    requireNonnegativeInteger(value.split_seed, "split_seed", "experiment configuration");
    requireNonnegativeInteger(value.corruption_seed, "corruption_seed", "experiment configuration");
    requirePositiveFiniteNumber(
      value.numeric_tolerance,
      "numeric_tolerance",
      "experiment configuration",
    );
    assertCompatible(
      value.config_schema_version,
      EXPERIMENT_CONFIG_VERSION,
      "config_schema_version",
      "experiment configuration",
    );

    return Object.freeze(
      Object.assign(new ExperimentConfig(), {
        configSchemaVersion: value.config_schema_version,
        datasetRevision: value.dataset_revision,
        splitSeed: value.split_seed,
        corruptionSeed: value.corruption_seed,
        preprocessingVersion: value.preprocessing_version,
        checkpointRevision: value.checkpoint_revision,
        artifactSchemaVersion: value.artifact_schema_version,
        signalRepresentationVersion: value.signal_representation_version,
        manifestPath: value.manifest_path,
        modelBundlePath: value.model_bundle_path,
        numericTolerance: value.numeric_tolerance,
      }),
    );
  }

  static read(path) {
    return ExperimentConfig.from(readJson(path, "experiment configuration"));
  }
}

export function variantIdentifier({
  sourceId,
  conditionFamily,
  corruptionParameters,
  corruptionSeed,
  preprocessingVersion,
  artifactSchemaVersion,
}) {
  const contractName = "variant identity";
  for (const [field, value] of [
    ["sourceId", sourceId],
    ["conditionFamily", conditionFamily],
    ["preprocessingVersion", preprocessingVersion],
    ["artifactSchemaVersion", artifactSchemaVersion],
  ]) {
    requireNonemptyString(value, field, contractName);
  }
  requireObject(corruptionParameters, `${contractName}.corruptionParameters`);
  requireNonnegativeInteger(corruptionSeed, "corruptionSeed", contractName);
  return hashIdentifier("variant-v1", {
    artifact_schema_version: artifactSchemaVersion,
    condition_family: conditionFamily,
    corruption_parameters: corruptionParameters,
    corruption_seed: corruptionSeed,
    preprocessing_version: preprocessingVersion,
    source_id: sourceId,
  });
}

export function cacheKey(config, variantId, artifactKind) {
  requireNonemptyString(variantId, "variantId", "cache identity");
  requireNonemptyString(artifactKind, "artifactKind", "cache identity");
  return hashIdentifier("cache-v1", {
    artifact_kind: artifactKind,
    artifact_schema_version: config.artifactSchemaVersion,
    checkpoint_revision: config.checkpointRevision,
    dataset_revision: config.datasetRevision,
    preprocessing_version: config.preprocessingVersion,
    signal_representation_version: config.signalRepresentationVersion,
    variant_id: variantId,
  });
}

export function readModelBundle(path, config) {
  const contractName = "model bundle metadata";
  const value = requireObject(readJson(path, contractName), contractName);
  const requiredFields = [
    "bundle_schema_version",
    "artifact_schema_version",
    "preprocessing_version",
    "checkpoint_revision",
    "signal_representation_version",
    "fusion_version",
    "numeric_tolerance",
  ];
  requireFields(value, requiredFields, contractName);
  for (const field of requiredFields.slice(0, -1)) {
    requireNonemptyString(value[field], field, contractName);
  }
  requirePositiveFiniteNumber(value.numeric_tolerance, "numeric_tolerance", contractName);
  assertCompatible(
    value.bundle_schema_version,
    MODEL_BUNDLE_VERSION,
    "bundle_schema_version",
    contractName,
  );
  for (const [field, expected] of [
    ["artifact_schema_version", config.artifactSchemaVersion],
    ["preprocessing_version", config.preprocessingVersion],
    ["checkpoint_revision", config.checkpointRevision],
    ["signal_representation_version", config.signalRepresentationVersion],
  ]) {
    assertCompatible(value[field], expected, field, contractName);
  }
  assertCompatible(
    value.fusion_version,
    SUPPORTED_FUSION_VERSION,
    "fusion_version",
    contractName,
  );
  if (value.numeric_tolerance > config.numericTolerance) {
    throw new ContractError(
      `${contractName}.numeric_tolerance is incompatible: received ${value.numeric_tolerance}; maximum allowed by configuration is ${config.numericTolerance}.`,
    );
  }

  return Object.freeze({
    bundleSchemaVersion: value.bundle_schema_version,
    artifactSchemaVersion: value.artifact_schema_version,
    preprocessingVersion: value.preprocessing_version,
    checkpointRevision: value.checkpoint_revision,
    signalRepresentationVersion: value.signal_representation_version,
    fusionVersion: value.fusion_version,
    numericTolerance: value.numeric_tolerance,
  });
}

export function readCacheArtifact(path, config, modelBundle) {
  const contractName = "cache artifact";
  const value = requireObject(readJson(path, contractName), contractName);
  const requiredFields = [
    "artifact_schema_version",
    "preprocessing_version",
    "checkpoint_revision",
    "signal_representation_version",
    "variant_id",
    "artifact_kind",
    "cache_key",
    "rgb_logit",
    "signal_logit",
    "fusion_version",
    "pred",
  ];
  requireFields(value, requiredFields, contractName);
  for (const field of [
    "artifact_schema_version",
    "preprocessing_version",
    "checkpoint_revision",
    "signal_representation_version",
    "variant_id",
    "artifact_kind",
    "cache_key",
    "fusion_version",
  ]) {
    requireNonemptyString(value[field], field, contractName);
  }
  for (const field of ["rgb_logit", "signal_logit", "pred"]) {
    if (typeof value[field] !== "number" || !Number.isFinite(value[field])) {
      throw new ContractError(`${contractName}.${field} must be a finite number.`);
    }
  }
  if (value.pred < 0 || value.pred > 1) {
    throw new ContractError(`${contractName}.pred must be in [0, 1].`);
  }
  for (const [field, expected] of [
    ["artifact_schema_version", config.artifactSchemaVersion],
    ["preprocessing_version", config.preprocessingVersion],
    ["checkpoint_revision", config.checkpointRevision],
    ["signal_representation_version", config.signalRepresentationVersion],
  ]) {
    assertCompatible(value[field], expected, field, contractName);
  }
  assertCompatible(value.artifact_kind, SUPPORTED_ARTIFACT_KIND, "artifact_kind", contractName);
  requireObject(modelBundle, "expected model bundle metadata");
  requireNonemptyString(
    modelBundle.fusionVersion,
    "fusionVersion",
    "expected model bundle metadata",
  );
  assertCompatible(
    modelBundle.fusionVersion,
    SUPPORTED_FUSION_VERSION,
    "fusionVersion",
    "expected model bundle metadata",
  );
  assertCompatible(value.fusion_version, modelBundle.fusionVersion, "fusion_version", contractName);
  const expectedCacheKey = cacheKey(config, value.variant_id, value.artifact_kind);
  assertCompatible(value.cache_key, expectedCacheKey, "cache_key", contractName);
  return Object.freeze({ ...value });
}

export function validatePredictionRecords(records) {
  if (!Array.isArray(records)) {
    throw new ContractError("prediction output must be a JSON array.");
  }
  const seenPaths = new Set();
  let previousPath;
  for (const [index, record] of records.entries()) {
    requireObject(record, `prediction record ${index}`);
    const fields = Object.keys(record).sort();
    if (fields.length !== 2 || fields[0] !== "image_path" || fields[1] !== "pred") {
      throw new ContractError(
        `prediction record ${index} must contain exactly image_path and pred fields.`,
      );
    }
    requireNonemptyString(record.image_path, "image_path", `prediction record ${index}`);
    if (seenPaths.has(record.image_path)) {
      throw new ContractError(
        `prediction record ${index} repeats image_path ${JSON.stringify(record.image_path)}.`,
      );
    }
    seenPaths.add(record.image_path);
    if (previousPath !== undefined && previousPath > record.image_path) {
      throw new ContractError("prediction records must be ordered by image_path.");
    }
    previousPath = record.image_path;
    if (typeof record.pred !== "number" || !Number.isFinite(record.pred) || record.pred < 0 || record.pred > 1) {
      throw new ContractError(`prediction record ${index}.pred must be a finite number in [0, 1].`);
    }
  }
  return records;
}
