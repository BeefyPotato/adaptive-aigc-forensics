import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import * as sourceInventoryModule from "../src/source-inventory.js";
import {
  inspectSourceInventory,
  loadAndInspectSourceInventory,
} from "../src/source-inventory.js";

const inventoryPath = new URL("../fixtures/track5/inventory.jsonl", import.meta.url);
const datasetRoot = new URL("../fixtures/track5/", import.meta.url);

test("bounded worker pool starts work concurrently without exceeding its limit", async () => {
  assert.equal(typeof sourceInventoryModule.forEachWithConcurrency, "function");
  let release;
  const gate = new Promise((resolveGate) => {
    release = resolveGate;
  });
  let active = 0;
  let maximumActive = 0;
  const started = [];
  const completed = [];

  const running = sourceInventoryModule.forEachWithConcurrency(
    [1, 2, 3, 4],
    2,
    async (value) => {
      active += 1;
      maximumActive = Math.max(maximumActive, active);
      started.push(value);
      await gate;
      completed.push(value);
      active -= 1;
    },
  );
  await new Promise((resolveImmediate) => setImmediate(resolveImmediate));

  assert.deepEqual(started, [1, 2]);
  release();
  await running;
  assert.equal(maximumActive, 2);
  assert.deepEqual(completed.toSorted((left, right) => left - right), [1, 2, 3, 4]);
});

test("source inventory inspection records stable file and perceptual hashes", async () => {
  const inspected = await loadAndInspectSourceInventory(inventoryPath, {
    datasetRoot,
    concurrency: 2,
  });
  const repeated = await loadAndInspectSourceInventory(inventoryPath, {
    datasetRoot,
    concurrency: 1,
  });

  assert.deepEqual(inspected, repeated);
  assert.equal(inspected.length, 2);
  assert.deepEqual(
    inspected.map(({ img_id: imageId }) => imageId),
    ["fixture-authentic", "fixture-synthetic"],
  );
  for (const record of inspected) {
    const file = await readFile(new URL(record.image_path, datasetRoot));
    assert.equal(record.exact_sha256, createHash("sha256").update(file).digest("hex"));
    assert.match(record.perceptual_hash, /^[0-9a-f]{16}$/u);
    assert.equal(record.width, 16);
    assert.equal(record.height, 12);
    assert.deepEqual(Object.keys(record.provenance).toSorted(), [
      "license",
      "source_dataset",
      "source_reference",
    ]);
  }
  assert.notEqual(inspected[0].perceptual_hash, inspected[1].perceptual_hash);
});

test("source inventory paths cannot escape the declared dataset root", async () => {
  await assert.rejects(
    () =>
      inspectSourceInventory(
        [
          {
            img_id: "escape",
            image_path: "../experiment/images/authentic.ppm",
            label: 0,
            dataset_split: "train",
            provenance: {
              source_dataset: "fixture",
              source_reference: "escape",
              license: "CC0-1.0",
            },
          },
        ],
        { datasetRoot },
      ),
    /dataset root/i,
  );
});
