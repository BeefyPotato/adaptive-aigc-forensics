import { createHash } from "node:crypto";
import { isAbsolute } from "node:path";

import sharp from "sharp";

import {
  ContractError,
  requireFields,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { variantIdentifier } from "./contracts.js";
import {
  assertLeakageAuditPassed,
  auditTrack5Sources,
  ORGANIZER_DEMONSTRATION_POLICY,
} from "./leakage-audit.js";

export { assertLeakageAuditPassed, auditTrack5Sources } from "./leakage-audit.js";

export const TRACK5_SPLIT_PLAN = Object.freeze([
  Object.freeze({ split: "expert-training", datasetSplit: "train", perClass: 4_000 }),
  Object.freeze({ split: "fusion-training", datasetSplit: "train", perClass: 1_000 }),
  Object.freeze({ split: "internal-validation", datasetSplit: "validation", perClass: 1_000 }),
  Object.freeze({ split: "sealed-internal-test", datasetSplit: "validation", perClass: 1_000 }),
]);

export const TRACK5_CONDITION_MATRIX = Object.freeze([
  Object.freeze({ family: "clean", severity: "clean", parameters: Object.freeze({}) }),
  ...[90, 70, 50, 30].map((quality) =>
    Object.freeze({
      family: "jpeg",
      severity: `quality-${quality}`,
      parameters: Object.freeze({ quality, chroma_subsampling: "4:2:0" }),
    }),
  ),
  ...[0.5, 1, 2].map((sigma) =>
    Object.freeze({
      family: "blur",
      severity: `sigma-${sigma}`,
      parameters: Object.freeze({ sigma }),
    }),
  ),
  ...[0.5, 0.25].map((factor) =>
    Object.freeze({
      family: "resize",
      severity: `factor-${factor}`,
      parameters: Object.freeze({
        factor,
        down_kernel: "lanczos3",
        up_kernel: "cubic",
      }),
    }),
  ),
  ...[0.02, 0.05, 0.1].map((sigma) =>
    Object.freeze({
      family: "noise",
      severity: `sigma-${sigma}`,
      parameters: Object.freeze({ sigma, color_space: "rgb-0-1" }),
    }),
  ),
  ...["brightness", "contrast", "saturation"].flatMap((property) =>
    [0.8, 1.2].map((factor) =>
      Object.freeze({
        family: "color",
        severity: `${property}-${factor}`,
        parameters: Object.freeze({ property, factor }),
      }),
    ),
  ),
  Object.freeze({
    family: "crop",
    severity: "center-0.8",
    parameters: Object.freeze({
      retained_fraction: 0.8,
      position: "center",
      restoration_kernel: "cubic",
    }),
  }),
]);

function deterministicRank(splitSeed, sourceId) {
  return createHash("sha256").update(`${splitSeed}\0${sourceId}`, "utf8").digest("hex");
}

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function requireHash(value, field, length, contractName) {
  if (typeof value !== "string" || !new RegExp(`^[0-9a-f]{${length}}$`, "u").test(value)) {
    throw new ContractError(`${contractName}.${field} must be a lowercase ${length}-digit hex value.`);
  }
}

function requireRelativeImagePath(imagePath, contractName) {
  requireNonemptyString(imagePath, "image_path", contractName);
  if (isAbsolute(imagePath) || imagePath.replaceAll("\\", "/").split("/").includes("..")) {
    throw new ContractError(`${contractName}.image_path must stay relative to the dataset root.`);
  }
}

function normalizeInventoryRecord(record, index, datasetRevision) {
  const contractName = `SID_Set inventory record ${index}`;
  requireObject(record, contractName);
  requireFields(
    record,
    [
      "img_id",
      "image_path",
      "label",
      "dataset_split",
      "width",
      "height",
      "exact_sha256",
      "perceptual_hash",
      "provenance",
    ],
    contractName,
  );
  requireNonemptyString(record.img_id, "img_id", contractName);
  requireRelativeImagePath(record.image_path, contractName);
  if (record.label !== 0 && record.label !== 1 && record.label !== 2) {
    throw new ContractError(`${contractName}.label must be 0, 1, or 2.`);
  }
  if (record.dataset_split !== "train" && record.dataset_split !== "validation") {
    throw new ContractError(`${contractName}.dataset_split must be "train" or "validation".`);
  }
  for (const field of ["width", "height"]) {
    if (!Number.isSafeInteger(record[field]) || record[field] <= 0) {
      throw new ContractError(`${contractName}.${field} must be a positive safe integer.`);
    }
  }
  requireHash(record.exact_sha256, "exact_sha256", 64, contractName);
  requireHash(record.perceptual_hash, "perceptual_hash", 16, contractName);
  requireObject(record.provenance, `${contractName}.provenance`);
  for (const field of ["source_dataset", "source_reference", "license"]) {
    requireNonemptyString(record.provenance[field], field, `${contractName}.provenance`);
  }

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
  { datasetRevision, splitSeed, splitPlan = TRACK5_SPLIT_PLAN },
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
      rank: deterministicRank(splitSeed, source.source_id),
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

  const offsets = new Map();
  const selected = [];
  for (const allocation of splitPlan) {
    requireNonemptyString(allocation.split, "split", "Track 5 split allocation");
    if (allocation.datasetSplit !== "train" && allocation.datasetSplit !== "validation") {
      throw new ContractError("Track 5 split allocation.datasetSplit must be train or validation.");
    }
    requireNonnegativeInteger(
      allocation.perClass,
      "perClass",
      "Track 5 split allocation",
    );
    for (const label of [0, 1]) {
      const key = `${allocation.datasetSplit}:${label}`;
      const start = offsets.get(key) ?? 0;
      const end = start + allocation.perClass;
      const available = buckets.get(key);
      if (available.length < end) {
        throw new ContractError(
          `SID_Set inventory has ${available.length} eligible ${allocation.datasetSplit} class-${label} sources; ${end} are required through ${allocation.split}.`,
        );
      }
      for (const source of available.slice(start, end)) {
        selected.push(Object.freeze({ ...source, split: allocation.split }));
      }
      offsets.set(key, end);
    }
  }
  return Object.freeze(selected);
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
    selection_contract_version: "track5-source-selection-v1",
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
    }),
    corruption: Object.freeze({
      root_seed: corruptionSeed,
      preprocessing_version: preprocessingVersion,
      artifact_schema_version: artifactSchemaVersion,
      transform_implementation_version: transformImplementationVersion,
      sharp_version: sharp.versions.sharp,
      libvips_version: sharp.versions.vips,
      node_version: process.versions.node,
      condition_count_per_source: TRACK5_CONDITION_MATRIX.length,
    }),
    organizer_demonstration_policy: ORGANIZER_DEMONSTRATION_POLICY,
    sources,
    observations,
    leakage_audit: leakageAudit,
  });
}
