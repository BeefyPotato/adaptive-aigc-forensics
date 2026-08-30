#!/usr/bin/env node
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

import {
  materializeTrack5Observations,
  writeMaterializedManifest,
} from "../src/materialized-observations.js";

function argumentsByName(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!name?.startsWith("--") || value === undefined) {
      throw new Error("Expected --manifest, --dataset-root, --output-dir, and optional --concurrency.");
    }
    values.set(name, value);
  }
  return values;
}

const values = argumentsByName(process.argv.slice(2));
for (const required of ["--manifest", "--dataset-root", "--output-dir"]) {
  if (!values.has(required)) throw new Error(`Missing ${required}.`);
}
const manifest = JSON.parse(await readFile(values.get("--manifest"), "utf8"));
const outputDirectory = resolve(values.get("--output-dir"));
const resolved = await materializeTrack5Observations(manifest, {
  concurrency: Number(values.get("--concurrency") ?? 4),
  datasetRoot: values.get("--dataset-root"),
  outputDirectory,
});
await writeMaterializedManifest(resolve(outputDirectory, "track5-materialized-manifest.json"), resolved);
process.stdout.write(`Materialized ${resolved.observations.length} observations under ${outputDirectory}.\n`);
