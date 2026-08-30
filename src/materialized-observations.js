import { createHash } from "node:crypto";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { isAbsolute, relative, resolve, sep } from "node:path";

import sharp from "sharp";

import { ContractError, requireFields, requireNonemptyString, requireObject } from "./contract-validation.js";
import { applyCorruption, decodeSourceImage } from "./corruption-harness.js";

const CONTROLLED_SPLITS = new Set([
  "expert-training",
  "fusion-training",
  "internal-validation",
  "sealed-internal-test",
]);

function containedPath(root, path, field) {
  requireNonemptyString(path, field, "Track 5 materialization");
  if (isAbsolute(path)) {
    throw new ContractError(`Track 5 materialization.${field} must be relative.`);
  }
  const absoluteRoot = resolve(root);
  const absolutePath = resolve(absoluteRoot, path);
  const relation = relative(absoluteRoot, absolutePath);
  if (relation === ".." || relation.startsWith(`..${sep}`) || isAbsolute(relation)) {
    throw new ContractError(`Track 5 materialization.${field} escapes the dataset root.`);
  }
  return absolutePath;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

async function writeAtomically(path, bytes) {
  const temporaryPath = `${path}.tmp-${process.pid}`;
  await writeFile(temporaryPath, bytes);
  await rename(temporaryPath, path);
}

async function mapWithConcurrency(values, concurrency, mapper) {
  if (!Number.isSafeInteger(concurrency) || concurrency <= 0) {
    throw new ContractError("Track 5 materialization concurrency must be a positive integer.");
  }
  const results = new Array(values.length);
  let offset = 0;
  async function worker() {
    while (offset < values.length) {
      const index = offset;
      offset += 1;
      results[index] = await mapper(values[index], index);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, values.length) }, worker));
  return results;
}

export async function materializeTrack5Observations(
  manifest,
  { concurrency = 4, datasetRoot, outputDirectory },
) {
  requireObject(manifest, "Track 5 manifest");
  requireFields(
    manifest,
    ["manifest_schema_version", "corruption", "sources", "observations"],
    "Track 5 manifest",
  );
  if (manifest.manifest_schema_version !== "track5-manifest-v1") {
    throw new ContractError("Materialization requires track5-manifest-v1.");
  }
  if (!Array.isArray(manifest.sources) || !Array.isArray(manifest.observations)) {
    throw new ContractError("Track 5 manifest sources and observations must be arrays.");
  }
  requireObject(manifest.corruption, "Track 5 manifest.corruption");
  requireFields(
    manifest.corruption,
    ["preprocessing_version", "transform_implementation_version"],
    "Track 5 manifest.corruption",
  );

  const sourceById = new Map();
  for (const [index, source] of manifest.sources.entries()) {
    requireObject(source, `Track 5 source ${index}`);
    requireFields(
      source,
      ["source_id", "image_path", "authenticity_label", "split"],
      `Track 5 source ${index}`,
    );
    if (!CONTROLLED_SPLITS.has(source.split) || ![0, 1].includes(source.authenticity_label)) {
      throw new ContractError(`Track 5 source ${index} is outside the controlled binary partitions.`);
    }
    if (sourceById.has(source.source_id)) {
      throw new ContractError(`Track 5 manifest repeats source ${source.source_id}.`);
    }
    sourceById.set(source.source_id, source);
  }

  const seenVariants = new Set();
  const observationsBySource = new Map();
  for (const [index, observation] of manifest.observations.entries()) {
    requireObject(observation, `Track 5 observation ${index}`);
    requireFields(
      observation,
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
      ],
      `Track 5 observation ${index}`,
    );
    if (seenVariants.has(observation.variant_id)) {
      throw new ContractError(`Track 5 manifest repeats variant ${observation.variant_id}.`);
    }
    seenVariants.add(observation.variant_id);
    const source = sourceById.get(observation.source_id);
    if (source === undefined) {
      throw new ContractError(`Track 5 observation ${observation.variant_id} has no source.`);
    }
    if (
      source.image_path !== observation.image_path ||
      source.split !== observation.split ||
      source.authenticity_label !== observation.authenticity_label
    ) {
      throw new ContractError(`Track 5 observation ${observation.variant_id} disagrees with its source.`);
    }
    if (
      observation.transform_implementation_version !==
      manifest.corruption.transform_implementation_version
    ) {
      throw new ContractError(`Track 5 observation ${observation.variant_id} has a stale corruption version.`);
    }
    const grouped = observationsBySource.get(observation.source_id) ?? [];
    grouped.push({ index, observation });
    observationsBySource.set(observation.source_id, grouped);
  }

  const absoluteOutput = resolve(outputDirectory);
  const imageDirectory = resolve(absoluteOutput, "observations");
  await mkdir(imageDirectory, { recursive: true });
  const materialized = new Array(manifest.observations.length);
  await mapWithConcurrency(
    [...observationsBySource.values()],
    concurrency,
    async (group) => {
      const sourcePath = containedPath(datasetRoot, group[0].observation.image_path, "image_path");
      const decoded = await decodeSourceImage(sourcePath);
      for (const { index, observation } of group) {
        const corrupted = await applyCorruption(decoded, observation);
        const bytes = await sharp(corrupted.data, {
          raw: { width: corrupted.width, height: corrupted.height, channels: corrupted.channels },
        })
          .png({ compressionLevel: 9, adaptiveFiltering: false, palette: false })
          .toBuffer();
        const filename = `${sha256(observation.variant_id)}.png`;
        await writeAtomically(resolve(imageDirectory, filename), bytes);
        materialized[index] = Object.freeze({
          ...observation,
          materialized_image_path: `observations/${filename}`,
          materialized_sha256: sha256(bytes),
          materialized_encoding: "lossless-rgb-png-v1",
        });
      }
    },
  );

  return Object.freeze({
    ...manifest,
    materialization_schema_version: "track5-materialized-observations-v1",
    materialization: Object.freeze({
      shared_observation_preprocessing_version: manifest.corruption.preprocessing_version,
      corruption_version: manifest.corruption.transform_implementation_version,
      encoding: "lossless-rgb-png-v1",
      observation_count: materialized.length,
    }),
    observations: Object.freeze(materialized),
  });
}

export async function writeMaterializedManifest(path, manifest) {
  await mkdir(resolve(path, ".."), { recursive: true });
  await writeAtomically(path, `${JSON.stringify(manifest, null, 2)}\n`);
}
