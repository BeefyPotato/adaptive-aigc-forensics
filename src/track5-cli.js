#!/usr/bin/env node

import { readFile } from "node:fs/promises";

import { ContractError } from "./contract-validation.js";
import { runTrack5Manifest } from "./track5-pipeline.js";

function usage() {
  return [
    "Usage: node ./src/track5-cli.js build-manifest",
    "--inventory <inventory.jsonl> --dataset-root <directory>",
    "--dataset-revision <immutable-revision> --output-dir <directory>",
    "[--split-plan <split-plan.json>] [--organizer-hashes <hashes.json>]",
    "[--split-seed <integer>] [--corruption-seed <integer>]",
    "[--perceptual-distance <0..64>] [--concurrency <positive-integer>]",
  ].join(" ");
}

function parseInteger(value, option, { minimum = 0, maximum } = {}) {
  if (!/^(?:0|[1-9]\d*)$/u.test(value ?? "")) {
    throw new ContractError(`${option} must be a non-negative integer.`);
  }
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < minimum || (maximum !== undefined && parsed > maximum)) {
    throw new ContractError(
      `${option} must be a safe integer from ${minimum}${maximum === undefined ? "" : ` through ${maximum}`}.`,
    );
  }
  return parsed;
}

function parseArguments(arguments_) {
  if (arguments_[0] !== "build-manifest") throw new ContractError(usage());
  const values = new Map();
  for (let index = 1; index < arguments_.length; index += 2) {
    const option = arguments_[index];
    const value = arguments_[index + 1];
    if (!option?.startsWith("--") || value === undefined) throw new ContractError(usage());
    if (values.has(option)) throw new ContractError(`Option ${option} was provided more than once.`);
    values.set(option, value);
  }
  const knownOptions = new Set([
    "--inventory",
    "--dataset-root",
    "--dataset-revision",
    "--output-dir",
    "--split-plan",
    "--organizer-hashes",
    "--split-seed",
    "--corruption-seed",
    "--perceptual-distance",
    "--concurrency",
  ]);
  for (const option of values.keys()) {
    if (!knownOptions.has(option)) throw new ContractError(`Unknown option ${option}. ${usage()}`);
  }
  for (const required of ["--inventory", "--dataset-root", "--dataset-revision", "--output-dir"]) {
    if (!values.has(required)) throw new ContractError(usage());
  }
  return values;
}

async function readJson(path, description) {
  let text;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    throw new ContractError(`${description} could not be read from ${path}: ${error.message}`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ContractError(`${description} at ${path} is invalid JSON: ${error.message}`);
  }
}

async function main() {
  const values = parseArguments(process.argv.slice(2));
  const splitPlan = values.has("--split-plan")
    ? await readJson(values.get("--split-plan"), "Track 5 split plan")
    : undefined;
  const organizerHashes = values.has("--organizer-hashes")
    ? await readJson(values.get("--organizer-hashes"), "organizer demonstration hashes")
    : undefined;
  const result = await runTrack5Manifest({
    inventoryPath: values.get("--inventory"),
    datasetRoot: values.get("--dataset-root"),
    datasetRevision: values.get("--dataset-revision"),
    outputDirectory: values.get("--output-dir"),
    splitPlan,
    organizerHashes,
    splitSeed: parseInteger(values.get("--split-seed") ?? "17", "--split-seed"),
    corruptionSeed: parseInteger(
      values.get("--corruption-seed") ?? "23",
      "--corruption-seed",
    ),
    perceptualDistance: parseInteger(
      values.get("--perceptual-distance") ?? "4",
      "--perceptual-distance",
      { maximum: 64 },
    ),
    concurrency: parseInteger(values.get("--concurrency") ?? "4", "--concurrency", {
      minimum: 1,
    }),
  });
  process.stdout.write(
    `Wrote ${result.sourceCount} source records and ${result.observationCount} observation records to ${result.outputDirectory}; leakage audit ${result.auditStatus}.\n`,
  );
}

try {
  await main();
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Track 5 manifest failed: ${message}\n`);
  process.exitCode = 1;
}
