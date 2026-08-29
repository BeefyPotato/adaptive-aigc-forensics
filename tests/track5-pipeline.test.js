import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { runTrack5Manifest } from "../src/track5-pipeline.js";

const inventoryPath = new URL("../fixtures/track5/inventory.jsonl", import.meta.url);
const datasetRoot = new URL("../fixtures/track5/", import.meta.url);
const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const splitPlan = [{ split: "expert-training", datasetSplit: "train", perClass: 1 }];

test("Track 5 fixture pipeline writes deterministic manifest and leakage audit artifacts", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-manifest-test-"));
  const firstOutput = join(temporaryRoot, "first");
  const secondOutput = join(temporaryRoot, "second");
  try {
    const options = {
      inventoryPath,
      datasetRoot,
      datasetRevision: "track5-controlled-fixture-v1",
      outputDirectory: firstOutput,
      splitPlan,
      splitSeed: 17,
      corruptionSeed: 23,
      perceptualDistance: 0,
    };
    const first = await runTrack5Manifest(options);
    const second = await runTrack5Manifest({ ...options, outputDirectory: secondOutput });

    assert.equal(first.sourceCount, 2);
    assert.equal(first.observationCount, 40);
    assert.equal(first.auditStatus, "passed");
    assert.deepEqual(
      await readFile(join(firstOutput, "track5-manifest.json"), "utf8"),
      await readFile(join(secondOutput, "track5-manifest.json"), "utf8"),
    );
    const audit = JSON.parse(
      await readFile(join(firstOutput, "track5-leakage-audit.json"), "utf8"),
    );
    assert.equal(audit.status, "passed");
    assert.equal(audit.organizer_demonstration.status, "not-available");
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("production split defaults refuse an incomplete source inventory", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-manifest-incomplete-"));
  try {
    await assert.rejects(
      () =>
        runTrack5Manifest({
          inventoryPath,
          datasetRoot,
          datasetRevision: "track5-controlled-fixture-v1",
          outputDirectory: temporaryRoot,
          perceptualDistance: 0,
        }),
      /required through expert-training/i,
    );
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("documented Track 5 CLI runs the controlled fixture", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-cli-test-"));
  try {
    const completed = spawnSync(
      process.execPath,
      [
        "./src/track5-cli.js",
        "build-manifest",
        "--inventory",
        "./fixtures/track5/inventory.jsonl",
        "--dataset-root",
        "./fixtures/track5",
        "--dataset-revision",
        "track5-controlled-fixture-v1",
        "--split-plan",
        "./fixtures/track5/split-plan.json",
        "--perceptual-distance",
        "0",
        "--output-dir",
        temporaryRoot,
      ],
      { cwd: repositoryRoot, encoding: "utf8" },
    );
    assert.equal(completed.status, 0, completed.stderr);
    assert.match(completed.stdout, /2 source records and 40 observation records/u);
    const manifest = JSON.parse(
      await readFile(join(temporaryRoot, "track5-manifest.json"), "utf8"),
    );
    assert.equal(manifest.manifest_schema_version, "track5-manifest-v1");
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});
