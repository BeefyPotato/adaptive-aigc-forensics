import { mkdir, rename, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

import sharp from "sharp";

import { buildTrack5Manifest } from "./track5-manifest.js";
import { loadAndInspectSourceInventory } from "./source-inventory.js";

async function writeJsonAtomically(path, value) {
  const temporaryPath = `${path}.tmp-${process.pid}`;
  await writeFile(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await rename(temporaryPath, path);
}

export async function runTrack5Manifest({
  artifactSchemaVersion = "artifact-v1",
  concurrency = 4,
  corruptionSeed = 23,
  datasetRevision,
  datasetRoot,
  inventoryPath,
  organizerHashes,
  outputDirectory,
  perceptualDistance = 4,
  preprocessingVersion = "shared-preprocessing-v1",
  splitPlan,
  splitSeed = 17,
  transformImplementationVersion = `track5-corruption-v1+sharp-${sharp.versions.sharp}`,
}) {
  const inventory = await loadAndInspectSourceInventory(inventoryPath, {
    datasetRoot,
    concurrency,
  });
  const manifest = buildTrack5Manifest(inventory, {
    artifactSchemaVersion,
    corruptionSeed,
    datasetRevision,
    organizerHashes,
    perceptualDistance,
    preprocessingVersion,
    ...(splitPlan === undefined ? {} : { splitPlan }),
    splitSeed,
    transformImplementationVersion,
  });
  const absoluteOutputDirectory = resolve(outputDirectory);
  await mkdir(absoluteOutputDirectory, { recursive: true });
  await writeJsonAtomically(join(absoluteOutputDirectory, "track5-manifest.json"), manifest);
  await writeJsonAtomically(
    join(absoluteOutputDirectory, "track5-leakage-audit.json"),
    manifest.leakage_audit,
  );
  return Object.freeze({
    outputDirectory: absoluteOutputDirectory,
    sourceCount: manifest.sources.length,
    observationCount: manifest.observations.length,
    auditStatus: manifest.leakage_audit.status,
  });
}
