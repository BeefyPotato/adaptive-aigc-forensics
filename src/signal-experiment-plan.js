import { createHash } from "node:crypto";

import { sampleBalancedTrainingObservations } from "./balanced-sampler.js";
import {
  ContractError,
  compareText,
  requireFields,
  requireLowercaseHex,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { canonicalJson, variantIdentifier } from "./contracts.js";
import { TRACK5_CONDITION_MATRIX } from "./track5-conditions.js";

const CONTROLLED_SPLITS = new Set([
  "expert-training",
  "fusion-training",
  "internal-validation",
  "sealed-internal-test",
]);
const SIGNAL_PHASES = Object.freeze(["expert-training", "internal-validation"]);
const PLAN_SCHEMA_VERSION = "signal-experiment-plan-v1";
const SHARD_SCHEMA_VERSION = "signal-experiment-shard-v1";

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function requirePositiveSafeInteger(value, field, contractName) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new ContractError(`${contractName}.${field} must be a positive safe integer.`);
  }
}

function rejectEvaluationOnlyRecord(record, contractName) {
  if (record.dataset === "organizer-demonstration") {
    throw new ContractError(
      `Organizer demonstration data is prohibited in ${contractName}.`,
    );
  }
  if (record.usage === "evaluation-only") {
    throw new ContractError(`Evaluation-only data is prohibited in ${contractName}.`);
  }
}

function validatePinnedSourceMetadata(record, contractName) {
  const hasByteLength = Object.hasOwn(record, "byte_length");
  const hasExactSha256 = Object.hasOwn(record, "exact_sha256");
  if (hasByteLength && !hasExactSha256) {
    throw new ContractError(
      `${contractName}.byte_length requires exact_sha256.`,
    );
  }
  if (hasByteLength) {
    requireNonnegativeInteger(record.byte_length, "byte_length", contractName);
    if (record.byte_length === 0) {
      throw new ContractError(`${contractName}.byte_length must be positive.`);
    }
  }
  if (!hasExactSha256) return false;
  requireLowercaseHex(record.exact_sha256, "exact_sha256", 64, contractName);
  return true;
}

function validatePinnedSourceAgreement(record, source, contractName) {
  if (!validatePinnedSourceMetadata(record, contractName)) return;
  if (
    record.exact_sha256 !== source.exact_sha256 ||
    (Object.hasOwn(record, "byte_length") && record.byte_length !== source.byte_length)
  ) {
    throw new ContractError(`${contractName} pinned source bytes disagree with its source.`);
  }
}

function conditionKey(family, severity) {
  return `${family}\0${severity}`;
}

const CONDITION_BY_KEY = new Map(
  TRACK5_CONDITION_MATRIX.map((condition, index) => [
    conditionKey(condition.family, condition.severity),
    { condition, index },
  ]),
);

function validateRecipeManifest(manifest) {
  requireObject(manifest, "signal recipe manifest");
  requireFields(
    manifest,
    [
      "manifest_schema_version",
      "source_contract_version",
      "observation_contract_version",
      "condition_matrix_version",
      "sampler_contract_version",
      "corruption",
      "organizer_demonstration_policy",
      "leakage_audit",
      "sources",
      "observations",
    ],
    "signal recipe manifest",
  );
  if (manifest.manifest_schema_version !== "track5-manifest-v1") {
    throw new ContractError("Signal planning requires track5-manifest-v1.");
  }
  for (const [field, expected, description] of [
    ["source_contract_version", "track5-source-v1", "source"],
    ["observation_contract_version", "track5-observation-v1", "observation"],
    ["condition_matrix_version", "track5-condition-matrix-v1", "condition matrix"],
    ["sampler_contract_version", "track5-balanced-sampler-v1", "sampler"],
  ]) {
    if (manifest[field] !== expected) {
      throw new ContractError(
        `Signal recipe manifest ${description} contract version must be ${expected}.`,
      );
    }
  }
  if (!Array.isArray(manifest.sources) || !Array.isArray(manifest.observations)) {
    throw new ContractError("Signal recipe manifest sources and observations must be arrays.");
  }
  requireObject(manifest.corruption, "signal recipe manifest.corruption");
  requireFields(
    manifest.corruption,
    [
      "root_seed",
      "preprocessing_version",
      "artifact_schema_version",
      "transform_implementation_version",
      "condition_count_per_source",
    ],
    "signal recipe manifest.corruption",
  );
  requireNonnegativeInteger(
    manifest.corruption.root_seed,
    "root_seed",
    "signal recipe manifest.corruption",
  );
  for (const field of [
    "preprocessing_version",
    "artifact_schema_version",
    "transform_implementation_version",
  ]) {
    requireNonemptyString(
      manifest.corruption[field],
      field,
      "signal recipe manifest.corruption",
    );
  }
  if (manifest.corruption.condition_count_per_source !== TRACK5_CONDITION_MATRIX.length) {
    throw new ContractError(
      `Signal planning requires exactly ${TRACK5_CONDITION_MATRIX.length} conditions per source.`,
    );
  }
  requireObject(
    manifest.organizer_demonstration_policy,
    "signal recipe manifest.organizer_demonstration_policy",
  );
  if (manifest.organizer_demonstration_policy.usage !== "evaluation-only") {
    throw new ContractError("Organizer demonstration data must remain evaluation-only.");
  }
  requireObject(manifest.leakage_audit, "signal recipe manifest.leakage_audit");
  if (manifest.leakage_audit.status !== "passed") {
    throw new ContractError("Signal planning requires a passed leakage audit.");
  }

  const sourceById = new Map();
  for (const [index, source] of manifest.sources.entries()) {
    const contractName = `signal recipe source ${index}`;
    requireObject(source, contractName);
    rejectEvaluationOnlyRecord(source, contractName);
    validatePinnedSourceMetadata(source, contractName);
    requireFields(
      source,
      ["source_id", "image_path", "authenticity_label", "split", "width", "height"],
      contractName,
    );
    requireNonemptyString(source.source_id, "source_id", contractName);
    requireNonemptyString(source.image_path, "image_path", contractName);
    if (sourceById.has(source.source_id)) {
      throw new ContractError(`Signal recipe manifest repeats source ${source.source_id}.`);
    }
    if (!CONTROLLED_SPLITS.has(source.split)) {
      throw new ContractError(`Signal recipe source ${source.source_id} has an invalid split.`);
    }
    if (source.authenticity_label !== 0 && source.authenticity_label !== 1) {
      throw new ContractError(`Signal recipe source ${source.source_id} must have label 0 or 1.`);
    }
    requirePositiveSafeInteger(source.width, "width", contractName);
    requirePositiveSafeInteger(source.height, "height", contractName);
    sourceById.set(source.source_id, source);
  }

  const observationByVariant = new Map();
  const observationBySourceCondition = new Map();
  for (const [index, observation] of manifest.observations.entries()) {
    const contractName = `signal recipe observation ${index}`;
    requireObject(observation, contractName);
    rejectEvaluationOnlyRecord(observation, contractName);
    requireFields(
      observation,
      [
        "observation_schema_version",
        "source_id",
        "variant_id",
        "image_path",
        "authenticity_label",
        "split",
        "condition_family",
        "severity",
        "corruption_parameters",
        "corruption_seed",
        "transform_implementation_version",
        "width",
        "height",
      ],
      contractName,
    );
    requireNonemptyString(observation.variant_id, "variant_id", contractName);
    if (observation.observation_schema_version !== "track5-observation-v1") {
      throw new ContractError(`${contractName} has an incompatible observation contract version.`);
    }
    if (observationByVariant.has(observation.variant_id)) {
      throw new ContractError(`Signal recipe manifest repeats variant ${observation.variant_id}.`);
    }
    const source = sourceById.get(observation.source_id);
    if (source === undefined) {
      throw new ContractError(`Signal recipe observation ${observation.variant_id} has no source.`);
    }
    validatePinnedSourceAgreement(observation, source, contractName);
    if (
      observation.image_path !== source.image_path ||
      observation.authenticity_label !== source.authenticity_label ||
      observation.split !== source.split ||
      observation.width !== source.width ||
      observation.height !== source.height
    ) {
      throw new ContractError(
        `Signal recipe observation ${observation.variant_id} disagrees with source ${source.source_id}.`,
      );
    }
    if (
      observation.transform_implementation_version !==
      manifest.corruption.transform_implementation_version
    ) {
      throw new ContractError(
        `Signal recipe observation ${observation.variant_id} has a stale corruption version.`,
      );
    }
    const declaredCondition = CONDITION_BY_KEY.get(
      conditionKey(observation.condition_family, observation.severity),
    );
    if (
      declaredCondition === undefined ||
      canonicalJson(observation.corruption_parameters) !==
        canonicalJson(declaredCondition.condition.parameters)
    ) {
      throw new ContractError(
        `Signal recipe observation ${observation.variant_id} is not a declared condition.`,
      );
    }
    const expectedVariantId = variantIdentifier({
      artifactSchemaVersion: manifest.corruption.artifact_schema_version,
      conditionFamily: observation.condition_family,
      corruptionParameters: observation.corruption_parameters,
      corruptionSeed: observation.corruption_seed,
      preprocessingVersion: manifest.corruption.preprocessing_version,
      severity: observation.severity,
      sourceId: observation.source_id,
      transformImplementationVersion: manifest.corruption.transform_implementation_version,
    });
    if (observation.variant_id !== expectedVariantId) {
      throw new ContractError(
        `Signal recipe observation ${observation.variant_id} has an incompatible variant identity.`,
      );
    }
    const sourceCondition = `${source.source_id}\0${observation.condition_family}\0${observation.severity}`;
    if (observationBySourceCondition.has(sourceCondition)) {
      throw new ContractError(
        `Signal recipe source ${source.source_id} repeats ${observation.condition_family}/${observation.severity}.`,
      );
    }
    observationByVariant.set(observation.variant_id, observation);
    observationBySourceCondition.set(sourceCondition, observation);
  }

  for (const source of sourceById.values()) {
    for (const condition of TRACK5_CONDITION_MATRIX) {
      const key = `${source.source_id}\0${condition.family}\0${condition.severity}`;
      if (!observationBySourceCondition.has(key)) {
        throw new ContractError(
          `Signal recipe source ${source.source_id} is missing ${condition.family}/${condition.severity}.`,
        );
      }
    }
  }
  if (observationByVariant.size !== sourceById.size * TRACK5_CONDITION_MATRIX.length) {
    throw new ContractError("Signal recipe manifest contains an unexpected condition observation.");
  }

  return { observationBySourceCondition, sourceById };
}

function recipeManifestHeader(manifest) {
  return Object.fromEntries(
    Object.entries(manifest).filter(([field]) => field !== "sources" && field !== "observations"),
  );
}

function orderedRecords(records) {
  return records.toSorted((left, right) => {
    const sourceOrder = compareText(left.source_id, right.source_id);
    if (sourceOrder !== 0) return sourceOrder;
    return (
      CONDITION_BY_KEY.get(conditionKey(left.condition_family, left.severity)).index -
      CONDITION_BY_KEY.get(conditionKey(right.condition_family, right.severity)).index
    );
  });
}

function trainingRecords(manifest, trainingCount, trainingSeed) {
  const sampled = sampleBalancedTrainingObservations(manifest, {
    count: trainingCount,
    seed: trainingSeed,
    split: "expert-training",
  });
  const recordByVariant = new Map();
  for (const observation of sampled) {
    const existing = recordByVariant.get(observation.variant_id);
    if (existing === undefined) {
      recordByVariant.set(observation.variant_id, { ...observation, sample_weight: 1 });
    } else {
      existing.sample_weight += 1;
    }
  }
  return orderedRecords([...recordByVariant.values()]);
}

function validationRecords(manifest) {
  return orderedRecords(
    manifest.observations
      .filter(({ split }) => split === "internal-validation")
      .map((observation) => ({ ...observation, sample_weight: 1 })),
  );
}

function groupRecordsBySource(records, sourceById) {
  const recordsBySource = new Map();
  for (const record of records) {
    const grouped = recordsBySource.get(record.source_id) ?? [];
    grouped.push(record);
    recordsBySource.set(record.source_id, grouped);
  }
  return [...recordsBySource]
    .toSorted(([left], [right]) => compareText(left, right))
    .map(([sourceId, sourceRecords]) => {
      const source = sourceById.get(sourceId);
      const rawByteEstimate = sourceRecords.reduce((total, record) => {
        const bytes = record.width * record.height * 3;
        if (!Number.isSafeInteger(bytes) || !Number.isSafeInteger(total + bytes)) {
          throw new ContractError(`Signal raw-byte estimate for ${sourceId} exceeds safe integer range.`);
        }
        return total + bytes;
      }, 0);
      return { rawByteEstimate, records: sourceRecords, source };
    });
}

function partitionPhase(phase, records, sourceById, rawByteBudget, header) {
  const groups = groupRecordsBySource(records, sourceById);
  if (groups.length === 0) {
    throw new ContractError(`Signal planning found no sources for ${phase}.`);
  }
  const partitions = [];
  let current = [];
  let currentBytes = 0;
  for (const group of groups) {
    if (group.rawByteEstimate > rawByteBudget) {
      throw new ContractError(
        `Signal source ${group.source.source_id} requires ${group.rawByteEstimate} raw bytes, exceeding budget ${rawByteBudget}.`,
      );
    }
    if (current.length > 0 && currentBytes + group.rawByteEstimate > rawByteBudget) {
      partitions.push(current);
      current = [];
      currentBytes = 0;
    }
    current.push(group);
    currentBytes += group.rawByteEstimate;
  }
  if (current.length > 0) partitions.push(current);

  return partitions.map((partition, index) => {
    const shardRecords = partition.flatMap(({ records: sourceRecords }) => sourceRecords);
    const sources = partition.map(({ source }) => source);
    const variantIds = shardRecords.map(({ variant_id: variantId }) => variantId).toSorted(compareText);
    return {
      shard_schema_version: SHARD_SCHEMA_VERSION,
      phase,
      index,
      count: partitions.length,
      raw_byte_budget: rawByteBudget,
      raw_byte_estimate: partition.reduce(
        (total, { rawByteEstimate }) => total + rawByteEstimate,
        0,
      ),
      variant_set_digest: sha256(canonicalJson(variantIds)),
      recipe_manifest_header: header,
      sources,
      records: shardRecords,
    };
  });
}

export function buildSignalExperimentPlan(
  manifest,
  {
    parentRecipeManifestSha256,
    rawByteBudget,
    trainingCount,
    trainingSeed,
  },
) {
  requireLowercaseHex(
    parentRecipeManifestSha256,
    "parentRecipeManifestSha256",
    64,
    "signal planning options",
  );
  requirePositiveSafeInteger(rawByteBudget, "rawByteBudget", "signal planning options");
  requireNonnegativeInteger(trainingCount, "trainingCount", "signal planning options");
  requireNonnegativeInteger(trainingSeed, "trainingSeed", "signal planning options");
  if (trainingCount === 0) {
    throw new ContractError("signal planning options.trainingCount must be greater than zero.");
  }
  const { sourceById } = validateRecipeManifest(manifest);
  const header = recipeManifestHeader(manifest);
  const phaseRecords = new Map([
    ["expert-training", trainingRecords(manifest, trainingCount, trainingSeed)],
    ["internal-validation", validationRecords(manifest)],
  ]);
  const phaseCores = SIGNAL_PHASES.map((phase) => ({
    phase,
    shards: partitionPhase(
      phase,
      phaseRecords.get(phase),
      sourceById,
      rawByteBudget,
      header,
    ).map((shard) => {
      const identifiedShard = {
        ...shard,
        parent_recipe_manifest_sha256: parentRecipeManifestSha256,
      };
      return {
        ...identifiedShard,
        shard_sha256: sha256(canonicalJson(identifiedShard)),
      };
    }),
  }));
  const identity = {
    plan_schema_version: PLAN_SCHEMA_VERSION,
    parent_recipe_manifest_sha256: parentRecipeManifestSha256,
    raw_byte_budget: rawByteBudget,
    training_count: trainingCount,
    training_seed: trainingSeed,
    phases: phaseCores,
  };
  const planSha256 = sha256(canonicalJson(identity));
  return Object.freeze({
    ...identity,
    plan_sha256: planSha256,
    phases: Object.freeze(
      phaseCores.map((phase) =>
        Object.freeze({
          ...phase,
          shards: Object.freeze(
            phase.shards.map((shard) =>
              Object.freeze({
                ...shard,
                plan_sha256: planSha256,
              }),
            ),
          ),
        }),
      ),
    ),
  });
}

export function validateSignalExperimentShard(
  shard,
  { expectedIndex, expectedPhase, expectedPlanSha256, expectedShardSha256 },
) {
  requireObject(shard, "signal experiment shard");
  requireFields(
    shard,
    [
      "shard_schema_version",
      "parent_recipe_manifest_sha256",
      "plan_sha256",
      "shard_sha256",
      "phase",
      "index",
      "count",
      "raw_byte_budget",
      "raw_byte_estimate",
      "variant_set_digest",
      "recipe_manifest_header",
      "sources",
      "records",
    ],
    "signal experiment shard",
  );
  if (shard.shard_schema_version !== SHARD_SCHEMA_VERSION) {
    throw new ContractError(`Signal materialization requires ${SHARD_SCHEMA_VERSION}.`);
  }
  for (const [field, value] of [
    ["parent_recipe_manifest_sha256", shard.parent_recipe_manifest_sha256],
    ["plan_sha256", shard.plan_sha256],
    ["shard_sha256", shard.shard_sha256],
    ["variant_set_digest", shard.variant_set_digest],
  ]) {
    requireLowercaseHex(value, field, 64, "signal experiment shard");
  }
  requireLowercaseHex(
    expectedPlanSha256,
    "expectedPlanSha256",
    64,
    "signal shard request",
  );
  requireLowercaseHex(
    expectedShardSha256,
    "expectedShardSha256",
    64,
    "signal shard request",
  );
  requireNonemptyString(expectedPhase, "expectedPhase", "signal shard request");
  requireNonnegativeInteger(expectedIndex, "expectedIndex", "signal shard request");
  if (!SIGNAL_PHASES.includes(shard.phase) || shard.phase !== expectedPhase) {
    throw new ContractError(
      `Signal shard phase ${JSON.stringify(shard.phase)} does not match requested ${JSON.stringify(expectedPhase)}.`,
    );
  }
  requireNonnegativeInteger(shard.index, "index", "signal experiment shard");
  requirePositiveSafeInteger(shard.count, "count", "signal experiment shard");
  if (shard.index >= shard.count || shard.index !== expectedIndex) {
    throw new ContractError(
      `Signal shard index ${shard.index} of ${shard.count} does not match requested ${expectedIndex}.`,
    );
  }
  if (shard.plan_sha256 !== expectedPlanSha256) {
    throw new ContractError("Signal shard plan SHA-256 does not match the requested plan.");
  }
  if (shard.shard_sha256 !== expectedShardSha256) {
    throw new ContractError("Signal shard SHA-256 does not match the requested shard.");
  }
  requirePositiveSafeInteger(
    shard.raw_byte_budget,
    "raw_byte_budget",
    "signal experiment shard",
  );
  requirePositiveSafeInteger(
    shard.raw_byte_estimate,
    "raw_byte_estimate",
    "signal experiment shard",
  );
  if (shard.raw_byte_estimate > shard.raw_byte_budget) {
    throw new ContractError("Signal shard raw-byte estimate exceeds its configured budget.");
  }
  requireObject(shard.recipe_manifest_header, "signal shard recipe manifest header");
  if (
    "sources" in shard.recipe_manifest_header ||
    "observations" in shard.recipe_manifest_header
  ) {
    throw new ContractError("Signal shard recipe header cannot override sources or observations.");
  }
  requireFields(
    shard.recipe_manifest_header,
    ["manifest_schema_version", "corruption"],
    "signal shard recipe manifest header",
  );
  if (shard.recipe_manifest_header.manifest_schema_version !== "track5-manifest-v1") {
    throw new ContractError("Signal shard recipe header must describe track5-manifest-v1.");
  }
  requireObject(shard.recipe_manifest_header.corruption, "signal shard recipe corruption");
  requireFields(
    shard.recipe_manifest_header.corruption,
    [
      "preprocessing_version",
      "artifact_schema_version",
      "transform_implementation_version",
    ],
    "signal shard recipe corruption",
  );
  if (!Array.isArray(shard.sources) || shard.sources.length === 0) {
    throw new ContractError("Signal shard sources must be a non-empty array.");
  }
  if (!Array.isArray(shard.records) || shard.records.length === 0) {
    throw new ContractError("Signal shard records must be a non-empty array.");
  }

  const sourceById = new Map();
  for (const [index, source] of shard.sources.entries()) {
    const contractName = `signal shard source ${index}`;
    requireObject(source, contractName);
    rejectEvaluationOnlyRecord(source, contractName);
    validatePinnedSourceMetadata(source, contractName);
    requireFields(
      source,
      ["source_id", "image_path", "authenticity_label", "split", "width", "height"],
      contractName,
    );
    if (sourceById.has(source.source_id)) {
      throw new ContractError(`Signal shard repeats source ${source.source_id}.`);
    }
    if (source.split !== shard.phase) {
      throw new ContractError(`Signal shard source ${source.source_id} has the wrong split.`);
    }
    sourceById.set(source.source_id, source);
  }

  const seenVariants = new Set();
  const referencedSources = new Set();
  const conditionCountBySource = new Map();
  let rawByteEstimate = 0;
  for (const [index, record] of shard.records.entries()) {
    const contractName = `signal shard record ${index}`;
    requireObject(record, contractName);
    rejectEvaluationOnlyRecord(record, contractName);
    requireFields(
      record,
      [
        "source_id",
        "variant_id",
        "image_path",
        "authenticity_label",
        "split",
        "condition_family",
        "severity",
        "corruption_parameters",
        "corruption_seed",
        "transform_implementation_version",
        "width",
        "height",
        "sample_weight",
      ],
      contractName,
    );
    if (seenVariants.has(record.variant_id)) {
      throw new ContractError(`Signal shard repeats variant ${record.variant_id}.`);
    }
    seenVariants.add(record.variant_id);
    const source = sourceById.get(record.source_id);
    if (source === undefined) {
      throw new ContractError(`Signal shard record ${record.variant_id} has no source.`);
    }
    validatePinnedSourceAgreement(record, source, contractName);
    if (
      record.image_path !== source.image_path ||
      record.authenticity_label !== source.authenticity_label ||
      record.split !== source.split ||
      record.width !== source.width ||
      record.height !== source.height
    ) {
      throw new ContractError(`Signal shard record ${record.variant_id} disagrees with its source.`);
    }
    if (!Number.isSafeInteger(record.sample_weight) || record.sample_weight <= 0) {
      throw new ContractError(`${contractName}.sample_weight must be a positive safe integer.`);
    }
    if (shard.phase === "internal-validation" && record.sample_weight !== 1) {
      throw new ContractError("Internal-validation signal observations must have sample weight 1.");
    }
    const declaredCondition = CONDITION_BY_KEY.get(
      conditionKey(record.condition_family, record.severity),
    );
    if (
      declaredCondition === undefined ||
      canonicalJson(record.corruption_parameters) !==
        canonicalJson(declaredCondition.condition.parameters)
    ) {
      throw new ContractError(`Signal shard record ${record.variant_id} is not a declared condition.`);
    }
    const expectedVariantId = variantIdentifier({
      artifactSchemaVersion: shard.recipe_manifest_header.corruption.artifact_schema_version,
      conditionFamily: record.condition_family,
      corruptionParameters: record.corruption_parameters,
      corruptionSeed: record.corruption_seed,
      preprocessingVersion: shard.recipe_manifest_header.corruption.preprocessing_version,
      severity: record.severity,
      sourceId: record.source_id,
      transformImplementationVersion:
        shard.recipe_manifest_header.corruption.transform_implementation_version,
    });
    if (
      record.variant_id !== expectedVariantId ||
      record.transform_implementation_version !==
        shard.recipe_manifest_header.corruption.transform_implementation_version
    ) {
      throw new ContractError(`Signal shard record ${record.variant_id} has stale identity metadata.`);
    }
    referencedSources.add(record.source_id);
    conditionCountBySource.set(
      record.source_id,
      (conditionCountBySource.get(record.source_id) ?? 0) + 1,
    );
    const recordBytes = record.width * record.height * 3;
    if (!Number.isSafeInteger(recordBytes) || !Number.isSafeInteger(rawByteEstimate + recordBytes)) {
      throw new ContractError("Signal shard raw-byte estimate exceeds safe integer range.");
    }
    rawByteEstimate += recordBytes;
  }
  if (referencedSources.size !== sourceById.size) {
    throw new ContractError("Signal shard contains a source without observation records.");
  }
  if (
    shard.phase === "internal-validation" &&
    [...conditionCountBySource.values()].some(
      (conditionCount) => conditionCount !== TRACK5_CONDITION_MATRIX.length,
    )
  ) {
    throw new ContractError("Internal-validation shards require the complete condition matrix per source.");
  }
  if (rawByteEstimate !== shard.raw_byte_estimate) {
    throw new ContractError("Signal shard raw-byte estimate does not match its records.");
  }
  const variantSetDigest = sha256(
    canonicalJson([...seenVariants].toSorted(compareText)),
  );
  if (variantSetDigest !== shard.variant_set_digest) {
    throw new ContractError("Signal shard variant-set digest does not match its records.");
  }
  const {
    plan_sha256: ignoredPlanSha256,
    shard_sha256: ignoredShardSha256,
    ...shardIdentity
  } = shard;
  if (sha256(canonicalJson(shardIdentity)) !== shard.shard_sha256) {
    throw new ContractError("Signal shard content does not match its SHA-256 identity.");
  }

  return Object.freeze({
    recipeManifest: Object.freeze({
      ...shard.recipe_manifest_header,
      parent_recipe_manifest_sha256: shard.parent_recipe_manifest_sha256,
      sources: shard.sources,
      observations: shard.records,
    }),
    shard: Object.freeze({ ...shard }),
  });
}
