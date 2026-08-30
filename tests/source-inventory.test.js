import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import * as sourceInventoryModule from "../src/source-inventory.js";
import {
  inspectSourceInventory,
  loadAndInspectSourceInventory,
} from "../src/source-inventory.js";

const inventoryPath = new URL("../fixtures/track5/inventory.jsonl", import.meta.url);
const datasetRoot = new URL("../fixtures/track5/", import.meta.url);
const candidateMetadataPath = new URL(
  "../metadata/sid-set-candidate-pool-v1.json",
  import.meta.url,
);
const candidateRecordsPath = new URL(
  "../metadata/sid-set-candidates-v1.jsonl",
  import.meta.url,
);

test("pinned candidate pool validates its revision, counts, identities, and hashes", () => {
  assert.equal(typeof sourceInventoryModule.validatePinnedCandidatePool, "function");
  const metadata = {
    candidate_pool_schema_version: "sid-set-candidate-pool-v1",
    dataset: "saberzl/SID_Set",
    dataset_revision: "dc03ead57929879319ce30a82bfcfb8d317b10bd",
    split_seed: 17,
    reserve_per_class: 1,
    source_count: 2,
    bucket_counts: {
      "train:class-0": 1,
      "validation:class-1": 1,
    },
    records_path: "sid-set-candidates-v1.jsonl",
    license: "CC-BY-4.0",
  };
  const records = [
    {
      candidate_schema_version: "sid-set-candidate-v1",
      img_id: "authentic-train-00001",
      image_path: "train/authentic/authentic-train-00001.jpg",
      label: 0,
      dataset_split: "train",
      row_index: 101,
      byte_length: 3,
      exact_sha256: "a".repeat(64),
      provenance: {
        source_dataset: "SID_Set",
        source_reference:
          "saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd:train:101:authentic-train-00001",
        license: "CC-BY-4.0",
      },
    },
    {
      candidate_schema_version: "sid-set-candidate-v1",
      img_id: "full-synthetic-validation-00001",
      image_path:
        "validation/full-synthetic/full-synthetic-validation-00001.jpg",
      label: 1,
      dataset_split: "validation",
      row_index: 201,
      byte_length: 4,
      exact_sha256: "b".repeat(64),
      provenance: {
        source_dataset: "SID_Set",
        source_reference:
          "saberzl/SID_Set@dc03ead57929879319ce30a82bfcfb8d317b10bd:validation:201:full-synthetic-validation-00001",
        license: "CC-BY-4.0",
      },
    },
  ];

  const validated = sourceInventoryModule.validatePinnedCandidatePool(
    metadata,
    records,
    {
      dataset: "saberzl/SID_Set",
      datasetRevision: "dc03ead57929879319ce30a82bfcfb8d317b10bd",
    },
  );

  assert.deepEqual(validated.metadata, metadata);
  assert.deepEqual(validated.records, records);
  assert.throws(
    () =>
      sourceInventoryModule.validatePinnedCandidatePool(
        { ...metadata, dataset_revision: "f".repeat(40) },
        records,
        {
          dataset: "saberzl/SID_Set",
          datasetRevision: "dc03ead57929879319ce30a82bfcfb8d317b10bd",
        },
      ),
    /dataset_revision.*expected/i,
  );
});

test("tracked SID_Set candidate pool satisfies its pinned contract", async () => {
  const metadata = JSON.parse(await readFile(candidateMetadataPath, "utf8"));
  const records = (await readFile(candidateRecordsPath, "utf8"))
    .split(/\r?\n/u)
    .filter((line) => line.trim() !== "")
    .map((line) => JSON.parse(line));
  const validated = sourceInventoryModule.validatePinnedCandidatePool(
    metadata,
    records,
    {
      dataset: "saberzl/SID_Set",
      datasetRevision: "dc03ead57929879319ce30a82bfcfb8d317b10bd",
    },
  );

  assert.equal(validated.records.length, 14_600);
  assert.deepEqual(validated.metadata.bucket_counts, {
    "train:class-0": 5_150,
    "train:class-1": 5_150,
    "validation:class-0": 2_150,
    "validation:class-1": 2_150,
  });
});

test("SID_Set rows page accepts only complete pinned candidate identities", () => {
  assert.equal(typeof sourceInventoryModule.validateSidSetRowsPage, "function");
  const candidate = {
    img_id: "authentic-train-00001",
    image_path: "train/authentic/authentic-train-00001.jpg",
    label: 0,
    dataset_split: "train",
    row_index: 0,
  };
  const page = {
    features: [
      { feature_idx: 0, name: "image", type: { _type: "Image" } },
      { feature_idx: 1, name: "img_id", type: { dtype: "string", _type: "Value" } },
      { feature_idx: 2, name: "label", type: { dtype: "int64", _type: "Value" } },
    ],
    rows: [
      {
        row_idx: 0,
        row: {
          image: {
            src: "https://datasets-server.huggingface.co/signed-image",
            height: 768,
            width: 1024,
          },
          img_id: candidate.img_id,
          label: candidate.label,
          height: 768,
          width: 1024,
        },
        truncated_cells: [],
      },
    ],
    num_rows_total: 1,
    num_rows_per_page: 100,
    partial: false,
  };

  const urls = sourceInventoryModule.validateSidSetRowsPage(page, {
    datasetSplit: "train",
    expectedCandidates: [candidate],
    expectedTotalRows: 1,
    offset: 0,
    pageSize: 100,
  });

  assert.equal(
    urls.get(0),
    "https://datasets-server.huggingface.co/signed-image",
  );
  assert.throws(
    () =>
      sourceInventoryModule.validateSidSetRowsPage(
        { ...page, partial: true },
        {
          datasetSplit: "train",
          expectedCandidates: [candidate],
          expectedTotalRows: 1,
          offset: 0,
          pageSize: 100,
        },
      ),
    /partial/i,
  );
  assert.throws(
    () =>
      sourceInventoryModule.validateSidSetRowsPage(
        {
          ...page,
          rows: [
            {
              ...page.rows[0],
              row: { ...page.rows[0].row, img_id: "changed-upstream-id" },
            },
          ],
        },
        {
          datasetSplit: "train",
          expectedCandidates: [candidate],
          expectedTotalRows: 1,
          offset: 0,
          pageSize: 100,
        },
      ),
    /img_id.*expected/i,
  );
});

test("SID_Set transport retries transient failures but not permanent responses", async () => {
  assert.equal(typeof sourceInventoryModule.fetchWithRetry, "function");
  let permanentAttempts = 0;
  await assert.rejects(
    sourceInventoryModule.fetchWithRetry("https://example.test/missing", {
      description: "missing candidate",
      fetchImplementation: async () => {
        permanentAttempts += 1;
        return new Response("missing", { status: 404 });
      },
      delayImplementation: async () => {},
      attempts: 4,
    }),
    /HTTP 404/i,
  );
  assert.equal(permanentAttempts, 1);

  let transientAttempts = 0;
  const delays = [];
  const response = await sourceInventoryModule.fetchWithRetry(
    "https://example.test/transient",
    {
      description: "transient candidate",
      fetchImplementation: async () => {
        transientAttempts += 1;
        return transientAttempts < 3
          ? new Response("busy", { status: 503 })
          : new Response("ready", { status: 200 });
      },
      delayImplementation: async (milliseconds) => delays.push(milliseconds),
      attempts: 4,
    },
  );
  assert.equal(await response.text(), "ready");
  assert.equal(transientAttempts, 3);
  assert.deepEqual(delays, [1_000, 2_000]);
});

test("candidate file reuse requires its pinned byte length and SHA-256", async () => {
  assert.equal(typeof sourceInventoryModule.candidateFileMatches, "function");
  const temporaryRoot = await mkdtemp(join(tmpdir(), "sid-set-candidate-"));
  const candidatePath = join(temporaryRoot, "candidate.jpg");
  const bytes = Buffer.from("pinned candidate bytes", "utf8");
  const candidate = {
    byte_length: bytes.length,
    exact_sha256: createHash("sha256").update(bytes).digest("hex"),
  };
  try {
    assert.equal(
      await sourceInventoryModule.candidateFileMatches(candidatePath, candidate),
      false,
    );
    await writeFile(candidatePath, bytes);
    assert.equal(
      await sourceInventoryModule.candidateFileMatches(candidatePath, candidate),
      true,
    );
    await writeFile(candidatePath, Buffer.alloc(bytes.length, 0));
    assert.equal(
      await sourceInventoryModule.candidateFileMatches(candidatePath, candidate),
      false,
    );
    await writeFile(candidatePath, Buffer.from("short", "utf8"));
    assert.equal(
      await sourceInventoryModule.candidateFileMatches(candidatePath, candidate),
      false,
    );
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

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
