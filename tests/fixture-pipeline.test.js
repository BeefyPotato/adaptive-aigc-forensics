import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

import { prepareExpertInputs } from "../src/images.js";
import { runFixture } from "../src/pipeline.js";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const fixtureConfig = join(repositoryRoot, "fixtures", "experiment", "config.json");

test("shared preprocessing gives both experts the same observation", () => {
  const imagePath = join(
    repositoryRoot,
    "fixtures",
    "experiment",
    "images",
    "authentic.ppm",
  );
  const inputs = prepareExpertInputs(imagePath, { sigma: 0.02 }, 23);

  assert.equal(inputs.rgb.length, inputs.luminance.length);
  assert.ok(inputs.rgb.length > 0);
  for (let index = 0; index < inputs.rgb.length; index += 1) {
    const [red, green, blue] = inputs.rgb[index];
    const expected = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    assert.ok(Math.abs(inputs.luminance[index] - expected) <= Number.EPSILON);
  }
});

test("fixture pipeline repeats outputs and enforces prediction schema", () => {
  const directory = mkdtempSync(join(tmpdir(), "aigc-pipeline-"));
  const first = join(directory, "first");
  const second = join(directory, "second");
  runFixture(fixtureConfig, first);
  runFixture(fixtureConfig, second);

  for (const name of [
    "resolved_manifest.json",
    "cache.json",
    "predictions.json",
    "metrics.json",
  ]) {
    assert.deepEqual(readFileSync(join(first, name)), readFileSync(join(second, name)), name);
  }

  const predictions = JSON.parse(readFileSync(join(first, "predictions.json"), "utf8"));
  assert.deepEqual(
    predictions.map(({ image_path: imagePath }) => imagePath),
    predictions.map(({ image_path: imagePath }) => imagePath).toSorted(),
  );
  for (const record of predictions) {
    assert.deepEqual(Object.keys(record).toSorted(), ["image_path", "pred"]);
    assert.equal(typeof record.pred, "number");
    assert.ok(Number.isFinite(record.pred));
    assert.ok(record.pred >= 0 && record.pred <= 1);
  }
});

test("documented CLI runs the fixture from a clean checkout", () => {
  const outputDirectory = mkdtempSync(join(tmpdir(), "aigc-cli-"));
  const completed = spawnSync(
    process.execPath,
    [
      "./src/cli.js",
      "run-fixture",
      "--config",
      fixtureConfig,
      "--output-dir",
      outputDirectory,
    ],
    { cwd: repositoryRoot, encoding: "utf8" },
  );

  assert.equal(completed.status, 0, completed.stderr);
  assert.doesNotThrow(() => readFileSync(join(outputDirectory, "predictions.json")));
});
