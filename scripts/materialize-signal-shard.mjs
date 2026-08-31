#!/usr/bin/env node
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { basename, dirname, extname, resolve } from "node:path";

import { writeFileAtomically } from "../src/atomic-file.js";

import {
  materializeTrack5Observations,
  writeMaterializedManifest,
} from "../src/materialized-observations.js";
import {
  buildSignalExperimentPlan,
  validateSignalExperimentShard,
} from "../src/signal-experiment-plan.js";
import {
  ensureManagedDirectory,
  managedOutputPath,
  resolveManagedOutputRoot,
} from "../src/managed-output.js";

function argumentsByName(argv, allowed) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const value = argv[index + 1];
    if (!allowed.has(name) || value === undefined) {
      throw new Error(`Unexpected or incomplete argument ${JSON.stringify(name)}.`);
    }
    if (values.has(name)) throw new Error(`Argument ${name} was provided more than once.`);
    values.set(name, value);
  }
  return values;
}

function requireArguments(values, names) {
  for (const name of names) {
    if (!values.has(name)) throw new Error(`Missing ${name}.`);
  }
}

async function writeAtomically(path, value) {
  await writeFileAtomically(path, value);
}

async function writeJsonAtomically(path, value) {
  await writeAtomically(path, `${JSON.stringify(value, null, 2)}\n`);
}

function shardFilename({ count, index, phase }) {
  return `${phase}-${String(index).padStart(5, "0")}-of-${String(count).padStart(5, "0")}.json`;
}

async function planCommand(argv) {
  const values = argumentsByName(
    argv,
    new Set([
      "--manifest",
      "--experiment-profile",
      "--training-count",
      "--training-seed",
      "--validation-source-count",
      "--validation-seed",
      "--raw-byte-budget",
      "--output",
    ]),
  );
  requireArguments(values, [
    "--manifest",
    "--training-count",
    "--training-seed",
    "--raw-byte-budget",
    "--output",
  ]);
  const manifestBytes = await readFile(values.get("--manifest"));
  const manifest = JSON.parse(manifestBytes.toString("utf8"));
  const plan = buildSignalExperimentPlan(manifest, {
    experimentProfile: values.get("--experiment-profile") ?? "custom-v1",
    parentRecipeManifestSha256: createHash("sha256").update(manifestBytes).digest("hex"),
    rawByteBudget: Number(values.get("--raw-byte-budget")),
    trainingCount: Number(values.get("--training-count")),
    trainingSeed: Number(values.get("--training-seed")),
    validationSourceCount:
      values.get("--validation-source-count") === undefined ||
      values.get("--validation-source-count") === "all"
        ? null
        : Number(values.get("--validation-source-count")),
    validationSeed: Number(values.get("--validation-seed") ?? 61),
  });
  const requestedOutputPath = resolve(values.get("--output"));
  const outputRoot = await resolveManagedOutputRoot(
    dirname(requestedOutputPath),
    "signal plan output root",
  );
  const outputPath = await managedOutputPath(
    outputRoot,
    basename(requestedOutputPath),
    "signal plan",
  );
  const extension = extname(outputPath);
  const shardDirectory = await ensureManagedDirectory(
    outputRoot,
    `${basename(outputPath, extension)}.shards`,
    "signal plan shard directory",
  );
  for (const phase of plan.phases) {
    for (const shard of phase.shards) {
      const filename = shardFilename(shard);
      await writeJsonAtomically(
        await managedOutputPath(
          outputRoot,
          `${basename(shardDirectory)}/${filename}`,
          `signal plan shard ${filename}`,
        ),
        shard,
      );
    }
  }
  // The full plan is the commit marker: publish it only after every referenced shard exists.
  await writeJsonAtomically(outputPath, plan);
  process.stdout.write(
    `${JSON.stringify({
      command: "plan",
      acceptance_scope: plan.acceptance_scope,
      experiment_profile: plan.experiment_profile,
      parent_recipe_manifest_sha256: plan.parent_recipe_manifest_sha256,
      phase_shard_counts: Object.fromEntries(
        plan.phases.map(({ phase, shards }) => [phase, shards.length]),
      ),
      plan_path: outputPath,
      plan_sha256: plan.plan_sha256,
      shard_directory: shardDirectory,
      validation_source_count: plan.validation_source_count,
    })}\n`,
  );
}

async function materializeCommand(argv) {
  const values = argumentsByName(
    argv,
    new Set([
      "--shard-plan",
      "--expected-plan-sha256",
      "--expected-shard-sha256",
      "--phase",
      "--index",
      "--dataset-root",
      "--output-dir",
      "--concurrency",
    ]),
  );
  requireArguments(values, [
    "--shard-plan",
    "--expected-plan-sha256",
    "--expected-shard-sha256",
    "--phase",
    "--index",
    "--dataset-root",
    "--output-dir",
  ]);
  const shard = JSON.parse(await readFile(values.get("--shard-plan"), "utf8"));
  const validated = validateSignalExperimentShard(shard, {
    expectedIndex: Number(values.get("--index")),
    expectedPhase: values.get("--phase"),
    expectedPlanSha256: values.get("--expected-plan-sha256"),
    expectedShardSha256: values.get("--expected-shard-sha256"),
  });
  const outputDirectory = await resolveManagedOutputRoot(
    values.get("--output-dir"),
    "signal materialization output root",
  );
  const materialized = await materializeTrack5Observations(validated.recipeManifest, {
    concurrency: Number(values.get("--concurrency") ?? 4),
    datasetRoot: values.get("--dataset-root"),
    outputDirectory,
  });
  const resolved = Object.freeze({
    ...materialized,
    signal_shard_provenance: Object.freeze({
      parent_recipe_manifest_sha256: shard.parent_recipe_manifest_sha256,
      plan_sha256: shard.plan_sha256,
      shard_sha256: shard.shard_sha256,
      phase: shard.phase,
      index: shard.index,
      count: shard.count,
      variant_set_digest: shard.variant_set_digest,
      raw_byte_budget: shard.raw_byte_budget,
      raw_byte_estimate: shard.raw_byte_estimate,
    }),
  });
  const materializedManifestPath = await managedOutputPath(
    outputDirectory,
    "track5-materialized-manifest.json",
    "signal materialized manifest",
  );
  await writeMaterializedManifest(materializedManifestPath, resolved);
  process.stdout.write(
    `${JSON.stringify({
      command: "materialize",
      index: shard.index,
      materialized_manifest_path: materializedManifestPath,
      observation_count: resolved.observations.length,
      phase: shard.phase,
      plan_sha256: shard.plan_sha256,
      shard_sha256: shard.shard_sha256,
    })}\n`,
  );
}

const [command, ...argv] = process.argv.slice(2);
if (command === "plan") {
  await planCommand(argv);
} else if (command === "materialize") {
  await materializeCommand(argv);
} else {
  throw new Error("Expected command plan or materialize.");
}
