import { createHash } from "node:crypto";
import { readFile, realpath } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";

import sharp from "sharp";

import { writeFileAtomically } from "./atomic-file.js";

import {
  ContractError,
  requireFields,
  requireLowercaseHex,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { applyCorruption, decodeSourceImage } from "./corruption-harness.js";
import {
  ensureManagedDirectory,
  managedOutputPath,
  resolveManagedOutputRoot,
} from "./managed-output.js";

const CONTROLLED_SPLITS = new Set([
  "expert-training",
  "fusion-training",
  "internal-validation",
  "sealed-internal-test",
]);

function pathEscapesRoot(root, path) {
  const relation = relative(root, path);
  return relation === ".." || relation.startsWith(`..${sep}`) || isAbsolute(relation);
}

async function containedPath(root, path, field) {
  requireNonemptyString(path, field, "Track 5 materialization");
  if (isAbsolute(path)) {
    throw new ContractError(`Track 5 materialization.${field} must be relative.`);
  }
  const absoluteRoot = resolve(root);
  const absolutePath = resolve(absoluteRoot, path);
  if (pathEscapesRoot(absoluteRoot, absolutePath)) {
    throw new ContractError(`Track 5 materialization.${field} escapes the dataset root.`);
  }
  let resolvedRoot;
  let resolvedPath;
  try {
    [resolvedRoot, resolvedPath] = await Promise.all([
      realpath(absoluteRoot),
      realpath(absolutePath),
    ]);
  } catch {
    throw new ContractError(
      `Track 5 materialization.${field} could not be resolved within the dataset root.`,
    );
  }
  if (pathEscapesRoot(resolvedRoot, resolvedPath)) {
    throw new ContractError(`Track 5 materialization.${field} escapes the dataset root.`);
  }
  return resolvedPath;
}

function sha256(value) {
  return createHash("sha256").update(value).digest("hex");
}

function validateDeclaredRuntime(corruption) {
  const hasSharpVersion = Object.hasOwn(corruption, "sharp_version");
  const hasLibvipsVersion = Object.hasOwn(corruption, "libvips_version");
  if (hasSharpVersion !== hasLibvipsVersion) {
    throw new ContractError(
      "Track 5 manifest.corruption must declare sharp_version and libvips_version together.",
    );
  }
  if (!hasSharpVersion) return;
  for (const [field, actual] of [
    ["sharp_version", sharp.versions.sharp],
    ["libvips_version", sharp.versions.vips],
  ]) {
    requireNonemptyString(corruption[field], field, "Track 5 manifest.corruption");
    if (corruption[field] !== actual) {
      throw new ContractError(
        `Track 5 manifest.corruption.${field} is incompatible: ` +
          `declared ${corruption[field]}, runtime ${actual}.`,
      );
    }
  }
}

function validateDeclaredGeometry(record, contractName) {
  const hasWidth = Object.hasOwn(record, "width");
  const hasHeight = Object.hasOwn(record, "height");
  if (hasWidth !== hasHeight) {
    throw new ContractError(`${contractName} must declare width and height together.`);
  }
  if (!hasWidth) return false;
  for (const field of ["width", "height"]) {
    requireNonnegativeInteger(record[field], field, contractName);
    if (record[field] === 0) {
      throw new ContractError(`${contractName}.${field} must be positive.`);
    }
  }
  return true;
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

async function verifiedPinnedSourceBytes(sourcePath, source) {
  if (!Object.hasOwn(source, "exact_sha256")) return undefined;
  let bytes;
  try {
    bytes = await readFile(sourcePath);
  } catch (error) {
    throw new ContractError(
      `Track 5 source ${source.source_id} could not be read before decode: ${error.message}`,
    );
  }
  if (Object.hasOwn(source, "byte_length") && bytes.length !== source.byte_length) {
    throw new ContractError(
      `Track 5 source ${source.source_id} does not match its pinned byte length ` +
        `(received ${bytes.length}, expected ${source.byte_length}).`,
    );
  }
  if (sha256(bytes) !== source.exact_sha256) {
    throw new ContractError(
      `Track 5 source ${source.source_id} does not match its pinned SHA-256.`,
    );
  }
  return bytes;
}

async function writeAtomically(path, bytes) {
  await writeFileAtomically(path, bytes);
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
  validateDeclaredRuntime(manifest.corruption);

  const sourceById = new Map();
  for (const [index, source] of manifest.sources.entries()) {
    const contractName = `Track 5 source ${index}`;
    requireObject(source, contractName);
    requireFields(
      source,
      ["source_id", "image_path", "authenticity_label", "split"],
      contractName,
    );
    validatePinnedSourceMetadata(source, contractName);
    validateDeclaredGeometry(source, contractName);
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
    const contractName = `Track 5 observation ${index}`;
    requireObject(observation, contractName);
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
      contractName,
    );
    const observationHasPinnedHash = validatePinnedSourceMetadata(observation, contractName);
    const observationHasGeometry = validateDeclaredGeometry(observation, contractName);
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
      observationHasGeometry &&
      Object.hasOwn(source, "width") &&
      (observation.width !== source.width || observation.height !== source.height)
    ) {
      throw new ContractError(
        `Track 5 observation ${observation.variant_id} has incompatible native dimensions.`,
      );
    }
    if (
      observationHasPinnedHash &&
      (observation.exact_sha256 !== source.exact_sha256 ||
        (Object.hasOwn(observation, "byte_length") &&
          observation.byte_length !== source.byte_length))
    ) {
      throw new ContractError(
        `Track 5 observation ${observation.variant_id} has incompatible pinned source bytes.`,
      );
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

  const absoluteOutput = await resolveManagedOutputRoot(
    outputDirectory,
    "Track 5 materialization root",
  );
  const imageDirectory = await ensureManagedDirectory(
    absoluteOutput,
    "observations",
    "Track 5 observations directory",
  );
  const materialized = new Array(manifest.observations.length);
  await mapWithConcurrency(
    [...observationsBySource.values()],
    concurrency,
    async (group) => {
      const sourcePath = await containedPath(
        datasetRoot,
        group[0].observation.image_path,
        "image_path",
      );
      const source = sourceById.get(group[0].observation.source_id);
      const pinnedBytes = await verifiedPinnedSourceBytes(sourcePath, source);
      const decoded = await decodeSourceImage(
        pinnedBytes ?? sourcePath,
        pinnedBytes === undefined
          ? undefined
          : `verified source bytes for Track 5 source ${JSON.stringify(source.source_id)}`,
      );
      for (const { index, observation } of group) {
        const corrupted = await applyCorruption(decoded, observation);
        const expectedWidth = observation.width ?? source.width;
        const expectedHeight = observation.height ?? source.height;
        if (
          corrupted.channels !== 3 ||
          (expectedWidth !== undefined &&
            (corrupted.width !== expectedWidth || corrupted.height !== expectedHeight))
        ) {
          const declaredGeometry =
            expectedWidth === undefined ? "three-channel RGB" : `${expectedWidth}x${expectedHeight}x3`;
          throw new ContractError(
            `Track 5 corruption result for ${observation.variant_id} disagrees with declared ` +
              `native RGB geometry ${declaredGeometry}; received ` +
              `${corrupted.width}x${corrupted.height}x${corrupted.channels}.`,
          );
        }
        const bytes = await sharp(corrupted.data, {
          raw: { width: corrupted.width, height: corrupted.height, channels: corrupted.channels },
        })
          .png({ compressionLevel: 9, adaptiveFiltering: false, palette: false })
          .toBuffer();
        const filename = `${sha256(observation.variant_id)}.png`;
        const imagePath = await managedOutputPath(
          absoluteOutput,
          `observations/${filename}`,
          `Track 5 observation ${observation.variant_id}`,
        );
        await writeAtomically(imagePath, bytes);
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
      sharp_version: sharp.versions.sharp,
      libvips_version: sharp.versions.vips,
      encoding: "lossless-rgb-png-v1",
      observation_count: materialized.length,
    }),
    observations: Object.freeze(materialized),
  });
}

export async function writeMaterializedManifest(path, manifest) {
  const root = await resolveManagedOutputRoot(
    dirname(resolve(path)),
    "Track 5 materialized manifest root",
  );
  const manifestPath = await managedOutputPath(
    root,
    basename(path),
    "Track 5 materialized manifest",
  );
  await writeAtomically(manifestPath, `${JSON.stringify(manifest, null, 2)}\n`);
}
