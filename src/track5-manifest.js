import { createHash } from "node:crypto";

import sharp from "sharp";

import {
  compareText,
  ContractError,
  requireFields,
  requireLowercaseHex,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { variantIdentifier } from "./contracts.js";
import { deterministicHexRank } from "./deterministic-random.js";
import {
  assertLeakageAuditPassed,
  auditTrack5Sources,
  createPartitionLeakageGuard,
  ORGANIZER_DEMONSTRATION_POLICY,
} from "./leakage-audit.js";
import { validateSourceInventoryRecord } from "./source-inventory-contract.js";
import { TRACK5_CONDITION_MATRIX } from "./track5-conditions.js";

export { assertLeakageAuditPassed, auditTrack5Sources } from "./leakage-audit.js";
export { TRACK5_CONDITION_MATRIX } from "./track5-conditions.js";

export const TRACK5_SPLIT_PLAN = Object.freeze([
  Object.freeze({ split: "expert-training", datasetSplit: "train", perClass: 4_000 }),
  Object.freeze({ split: "fusion-training", datasetSplit: "train", perClass: 1_000 }),
  Object.freeze({ split: "internal-validation", datasetSplit: "validation", perClass: 1_000 }),
  Object.freeze({ split: "sealed-internal-test", datasetSplit: "validation", perClass: 1_000 }),
]);

export function selectTrack5CandidateInventoryRecords(
  inventory,
  { reservePerClass = 150, splitPlan = TRACK5_SPLIT_PLAN, splitSeed },
) {
  if (!Array.isArray(inventory)) {
    throw new ContractError("SID_Set candidate inventory must be an array.");
  }
  requireNonnegativeInteger(splitSeed, "splitSeed", "Track 5 candidate options");
  requireNonnegativeInteger(
    reservePerClass,
    "reservePerClass",
    "Track 5 candidate options",
  );
  if (!Array.isArray(splitPlan) || splitPlan.length === 0) {
    throw new ContractError("Track 5 split plan must be a non-empty array.");
  }

  const eligible = inventory.filter((record, index) => {
    validateSourceInventoryRecord(record, index);
    return record.label === 0 || record.label === 1;
  });
  const validationSourceIds = new Set(
    eligible
      .filter(({ dataset_split: datasetSplit }) => datasetSplit === "validation")
      .map(({ img_id: imageId }) => `sid-set:${imageId}`),
  );
  const retained = eligible.filter(
    ({ dataset_split: datasetSplit, img_id: imageId }) =>
      datasetSplit !== "train" || !validationSourceIds.has(`sid-set:${imageId}`),
  );
  const seenSourceIds = new Set();
  for (const { img_id: imageId } of retained) {
    const sourceId = `sid-set:${imageId}`;
    if (seenSourceIds.has(sourceId)) {
      throw new ContractError(`SID_Set inventory repeats source identity ${sourceId}.`);
    }
    seenSourceIds.add(sourceId);
  }

  const requiredPerClass = new Map();
  for (const allocation of splitPlan) {
    requireNonemptyString(allocation.split, "split", "Track 5 split allocation");
    if (allocation.datasetSplit !== "train" && allocation.datasetSplit !== "validation") {
      throw new ContractError("Track 5 split allocation.datasetSplit must be train or validation.");
    }
    requireNonnegativeInteger(allocation.perClass, "perClass", "Track 5 split allocation");
    requiredPerClass.set(
      allocation.datasetSplit,
      (requiredPerClass.get(allocation.datasetSplit) ?? 0) + allocation.perClass,
    );
  }

  const buckets = new Map();
  for (const datasetSplit of requiredPerClass.keys()) {
    for (const label of [0, 1]) buckets.set(`${datasetSplit}:${label}`, []);
  }
  for (const record of retained) {
    const key = `${record.dataset_split}:${record.label}`;
    if (!buckets.has(key)) continue;
    const sourceId = `sid-set:${record.img_id}`;
    buckets.get(key).push({
      rank: deterministicHexRank(splitSeed, sourceId),
      record,
      sourceId,
    });
  }

  const selected = [];
  for (const datasetSplit of ["train", "validation"]) {
    const required = requiredPerClass.get(datasetSplit);
    if (required === undefined) continue;
    for (const label of [0, 1]) {
      const candidates = buckets.get(`${datasetSplit}:${label}`).toSorted(
        (left, right) =>
          compareText(left.rank, right.rank) || compareText(left.sourceId, right.sourceId),
      );
      const candidateCount = required + reservePerClass;
      if (candidates.length < candidateCount) {
        throw new ContractError(
          `SID_Set inventory has ${candidates.length} eligible ${datasetSplit} class-${label} sources; ${candidateCount} candidates are required including reserve.`,
        );
      }
      for (const { record } of candidates.slice(0, candidateCount)) {
        selected.push(
          Object.freeze({
            ...record,
            provenance: Object.freeze({ ...record.provenance }),
          }),
        );
      }
    }
  }
  return Object.freeze(selected);
}

function normalizeInventoryRecord(record, index, datasetRevision) {
  const contractName = validateSourceInventoryRecord(record, index);
  requireFields(
    record,
    [
      "width",
      "height",
      "exact_sha256",
      "perceptual_hash",
    ],
    contractName,
  );
  for (const field of ["width", "height"]) {
    if (!Number.isSafeInteger(record[field]) || record[field] <= 0) {
      throw new ContractError(`${contractName}.${field} must be a positive safe integer.`);
    }
  }
  requireLowercaseHex(record.exact_sha256, "exact_sha256", 64, contractName);
  requireLowercaseHex(record.perceptual_hash, "perceptual_hash", 16, contractName);

  return Object.freeze({
    source_id: `sid-set:${record.img_id}`,
    image_path: record.image_path.replaceAll("\\", "/"),
    authenticity_label: record.label,
    split: undefined,
    dataset: "SID_Set",
    dataset_revision: datasetRevision,
    dataset_split: record.dataset_split,
    width: record.width,
    height: record.height,
    exact_sha256: record.exact_sha256,
    perceptual_hash: record.perceptual_hash,
    provenance: Object.freeze({ ...record.provenance }),
  });
}

export function selectTrack5Sources(
  inventory,
  {
    datasetRevision,
    perceptualDistance = 4,
    splitSeed,
    splitPlan = TRACK5_SPLIT_PLAN,
  },
) {
  if (!Array.isArray(inventory)) {
    throw new ContractError("SID_Set inventory must be an array.");
  }
  requireNonemptyString(datasetRevision, "datasetRevision", "Track 5 selection options");
  requireNonnegativeInteger(splitSeed, "splitSeed", "Track 5 selection options");
  if (!Array.isArray(splitPlan) || splitPlan.length === 0) {
    throw new ContractError("Track 5 split plan must be a non-empty array.");
  }

  const eligibleSources = inventory
    .map((record, index) => normalizeInventoryRecord(record, index, datasetRevision))
    .filter(({ authenticity_label: label }) => label === 0 || label === 1);
  const seenSourceIds = new Set();
  for (const source of eligibleSources) {
    if (seenSourceIds.has(source.source_id)) {
      throw new ContractError(`SID_Set inventory repeats source identity ${source.source_id}.`);
    }
    seenSourceIds.add(source.source_id);
  }

  const buckets = new Map();
  for (const datasetSplit of ["train", "validation"]) {
    for (const label of [0, 1]) {
      buckets.set(`${datasetSplit}:${label}`, []);
    }
  }
  for (const source of eligibleSources) {
    buckets.get(`${source.dataset_split}:${source.authenticity_label}`).push({
      rank: deterministicHexRank(splitSeed, source.source_id),
      source,
    });
  }
  for (const [key, rankedSources] of buckets) {
    rankedSources.sort(
      (left, right) =>
        compareText(left.rank, right.rank) ||
        compareText(left.source.source_id, right.source.source_id),
    );
    buckets.set(
      key,
      rankedSources.map(({ source }) => source),
    );
  }

  const leakageGuard = createPartitionLeakageGuard(perceptualDistance);
  const offsets = new Map();
  const requirements = new Map();
  const allocations = splitPlan.map((allocation, planIndex) => {
    requireNonemptyString(allocation.split, "split", "Track 5 split allocation");
    if (allocation.datasetSplit !== "train" && allocation.datasetSplit !== "validation") {
      throw new ContractError("Track 5 split allocation.datasetSplit must be train or validation.");
    }
    requireNonnegativeInteger(
      allocation.perClass,
      "perClass",
      "Track 5 split allocation",
    );
    return { allocation, planIndex };
  });
  const selectionOrder = allocations.toSorted(
    (left, right) =>
      (left.allocation.datasetSplit === "validation" ? 0 : 1) -
        (right.allocation.datasetSplit === "validation" ? 0 : 1) ||
      left.planIndex - right.planIndex,
  );
  const selectedByAllocation = splitPlan.map(() => []);
  for (const { allocation, planIndex } of selectionOrder) {
    for (const label of [0, 1]) {
      const key = `${allocation.datasetSplit}:${label}`;
      let offset = offsets.get(key) ?? 0;
      const previouslyRequired = requirements.get(key) ?? 0;
      const requiredThroughSplit = previouslyRequired + allocation.perClass;
      requirements.set(key, requiredThroughSplit);
      const available = buckets.get(key);
      let selectedForAllocation = 0;
      while (offset < available.length && selectedForAllocation < allocation.perClass) {
        const source = Object.freeze({ ...available[offset], split: allocation.split });
        offset += 1;
        if (leakageGuard.conflicts(source)) continue;
        selectedByAllocation[planIndex].push(source);
        leakageGuard.add(source);
        selectedForAllocation += 1;
      }
      if (selectedForAllocation < allocation.perClass) {
        throw new ContractError(
          `SID_Set inventory has ${available.length} eligible ${allocation.datasetSplit} class-${label} sources; ${requiredThroughSplit} collision-free sources are required through ${allocation.split}, but ${previouslyRequired + selectedForAllocation} were found.`,
        );
      }
      offsets.set(key, offset);
    }
  }
  return Object.freeze(selectedByAllocation.flat());
}

function observationSeed(rootSeed, sourceId, family, severity) {
  const digest = createHash("sha256")
    .update(`${rootSeed}\0${sourceId}\0${family}\0${severity}`, "utf8")
    .digest();
  return digest.readUIntBE(0, 6);
}

export function buildObservationRecords(
  sources,
  {
    artifactSchemaVersion,
    corruptionSeed,
    preprocessingVersion,
    transformImplementationVersion,
  },
) {
  if (!Array.isArray(sources) || sources.length === 0) {
    throw new ContractError("Track 5 observations require a non-empty source array.");
  }
  for (const [field, value] of [
    ["artifactSchemaVersion", artifactSchemaVersion],
    ["preprocessingVersion", preprocessingVersion],
    ["transformImplementationVersion", transformImplementationVersion],
  ]) {
    requireNonemptyString(value, field, "Track 5 observation options");
  }
  requireNonnegativeInteger(corruptionSeed, "corruptionSeed", "Track 5 observation options");

  const observations = [];
  for (const [index, source] of sources.entries()) {
    const contractName = `Track 5 source ${index}`;
    requireFields(
      source,
      ["source_id", "image_path", "authenticity_label", "split", "width", "height"],
      contractName,
    );
    for (const condition of TRACK5_CONDITION_MATRIX) {
      const seed = observationSeed(
        corruptionSeed,
        source.source_id,
        condition.family,
        condition.severity,
      );
      observations.push(
        Object.freeze({
          observation_schema_version: "track5-observation-v1",
          variant_id: variantIdentifier({
            sourceId: source.source_id,
            conditionFamily: condition.family,
            corruptionParameters: condition.parameters,
            corruptionSeed: seed,
            preprocessingVersion,
            artifactSchemaVersion,
            severity: condition.severity,
            transformImplementationVersion,
          }),
          source_id: source.source_id,
          image_path: source.image_path,
          authenticity_label: source.authenticity_label,
          split: source.split,
          condition_family: condition.family,
          severity: condition.severity,
          corruption_parameters: condition.parameters,
          corruption_seed: seed,
          transform_implementation_version: transformImplementationVersion,
          width: source.width,
          height: source.height,
        }),
      );
    }
  }
  return Object.freeze(observations);
}

function splitCounts(sources) {
  const counts = {};
  for (const source of sources) {
    const key = `${source.split}:class-${source.authenticity_label}`;
    counts[key] = (counts[key] ?? 0) + 1;
  }
  return Object.freeze(
    Object.fromEntries(
      Object.entries(counts).toSorted(([left], [right]) =>
        left < right ? -1 : left > right ? 1 : 0,
      ),
    ),
  );
}

export function buildTrack5Manifest(
  inventory,
  {
    artifactSchemaVersion,
    corruptionSeed,
    datasetRevision,
    organizerHashes,
    perceptualDistance = 4,
    preprocessingVersion,
    splitPlan = TRACK5_SPLIT_PLAN,
    splitSeed,
    transformImplementationVersion,
  },
) {
  const sources = selectTrack5Sources(inventory, {
    datasetRevision,
    perceptualDistance,
    splitSeed,
    splitPlan,
  });
  const leakageAudit = auditTrack5Sources(sources, {
    organizerHashes,
    perceptualDistance,
  });
  assertLeakageAuditPassed(leakageAudit);
  const observations = buildObservationRecords(sources, {
    artifactSchemaVersion,
    corruptionSeed,
    preprocessingVersion,
    transformImplementationVersion,
  });

  return Object.freeze({
    manifest_schema_version: "track5-manifest-v1",
    source_contract_version: "track5-source-v1",
    observation_contract_version: "track5-observation-v1",
    selection_contract_version: "track5-source-selection-v2",
    condition_matrix_version: "track5-condition-matrix-v1",
    sampler_contract_version: "track5-balanced-sampler-v1",
    dataset: Object.freeze({ name: "SID_Set", revision: datasetRevision }),
    selection: Object.freeze({
      split_seed: splitSeed,
      split_plan: Object.freeze(splitPlan.map((allocation) => Object.freeze({ ...allocation }))),
      source_count: sources.length,
      split_counts: splitCounts(sources),
      tampered_label_excluded: true,
      partition_unit: "source-image",
      collision_backfill: Object.freeze({
        exact_match_excluded: true,
        perceptual_distance_threshold: perceptualDistance,
        upstream_split_priority: Object.freeze(["validation", "train"]),
      }),
    }),
    corruption: Object.freeze({
      root_seed: corruptionSeed,
      preprocessing_version: preprocessingVersion,
      artifact_schema_version: artifactSchemaVersion,
      transform_implementation_version: transformImplementationVersion,
      sharp_version: sharp.versions.sharp,
      libvips_version: sharp.versions.vips,
      condition_count_per_source: TRACK5_CONDITION_MATRIX.length,
    }),
    organizer_demonstration_policy: ORGANIZER_DEMONSTRATION_POLICY,
    sources,
    observations,
    leakage_audit: leakageAudit,
  });
}
