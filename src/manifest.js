import { isAbsolute } from "node:path";

import {
  ContractError,
  assertCompatible,
  readJson,
  requireFields,
  requireNonemptyString,
} from "./contract-validation.js";
import { variantIdentifier } from "./contracts.js";

const MANIFEST_VERSION = "manifest-v1";

function validateRelativeImagePath(imagePath, contractName) {
  if (typeof imagePath !== "string" || imagePath.length === 0 || isAbsolute(imagePath)) {
    throw new ContractError(`${contractName}.image_path must be a non-empty relative path.`);
  }
  const segments = imagePath.replaceAll("\\", "/").split("/");
  if (segments.includes("..")) {
    throw new ContractError(`${contractName}.image_path must stay inside the manifest directory.`);
  }
}

function readSources(sourceRecords, config) {
  const sources = sourceRecords.map((source, index) => {
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
      requireNonemptyString(source[field], field, contractName);
    }
    validateRelativeImagePath(source.image_path, contractName);
    if (source.authenticity_label !== 0 && source.authenticity_label !== 1) {
      throw new ContractError(`${contractName}.authenticity_label must be 0 or 1.`);
    }
    assertCompatible(
      source.dataset_revision,
      config.datasetRevision,
      "dataset_revision",
      contractName,
    );
    return Object.freeze({ ...source });
  });

  const sourceById = new Map();
  for (const source of sources) {
    if (sourceById.has(source.source_id)) {
      throw new ContractError(`experiment manifest repeats source_id ${JSON.stringify(source.source_id)}.`);
    }
    sourceById.set(source.source_id, source);
  }
  return { sources, sourceById };
}

function resolveVariants(variantRecords, sourceById, config) {
  const variants = variantRecords.map((variant, index) => {
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
    assertCompatible(variant.condition_family, "noise", "condition_family", contractName);
    requireFields(variant.corruption_parameters, ["sigma"], `${contractName}.corruption_parameters`);
    if (!Number.isSafeInteger(variant.corruption_seed_offset) || variant.corruption_seed_offset < 0) {
      throw new ContractError(`${contractName}.corruption_seed_offset must be a non-negative safe integer.`);
    }
    const corruptionSeed = config.corruptionSeed + variant.corruption_seed_offset;
    if (!Number.isSafeInteger(corruptionSeed)) {
      throw new ContractError(`${contractName} produces a corruption seed outside the safe integer range.`);
    }
    return Object.freeze({
      variant_id: variantIdentifier({
        sourceId: source.source_id,
        conditionFamily: variant.condition_family,
        corruptionParameters: variant.corruption_parameters,
        corruptionSeed,
        preprocessingVersion: config.preprocessingVersion,
        artifactSchemaVersion: config.artifactSchemaVersion,
      }),
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
  return variants;
}

export function resolveManifest(path, config) {
  const value = readJson(path, "experiment manifest");
  requireFields(value, ["manifest_schema_version", "sources", "variants"], "experiment manifest");
  assertCompatible(
    value.manifest_schema_version,
    MANIFEST_VERSION,
    "manifest_schema_version",
    "experiment manifest",
  );
  if (!Array.isArray(value.sources) || !Array.isArray(value.variants)) {
    throw new ContractError("experiment manifest.sources and experiment manifest.variants must be arrays.");
  }

  const { sources, sourceById } = readSources(value.sources, config);
  const variants = resolveVariants(value.variants, sourceById, config);
  return Object.freeze({
    manifest_schema_version: MANIFEST_VERSION,
    dataset_revision: config.datasetRevision,
    split_seed: config.splitSeed,
    corruption_seed: config.corruptionSeed,
    preprocessing_version: config.preprocessingVersion,
    checkpoint_revision: config.checkpointRevision,
    artifact_schema_version: config.artifactSchemaVersion,
    signal_representation_version: config.signalRepresentationVersion,
    sources,
    variants,
  });
}
