import { createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import {
  selectTrack5CandidateInventoryRecords,
  TRACK5_SPLIT_PLAN,
} from "../src/track5-manifest.js";
import { forEachWithConcurrency } from "../src/source-inventory.js";

const DATASET = "saberzl/SID_Set";
const REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd";
const SPLIT_SEED = 17;
const RESERVE_PER_CLASS = 150;
const PAGE_SIZE = 100;
const METADATA_CONCURRENCY = 2;
const DOWNLOAD_GROUP_CONCURRENCY = 4;
const DOWNLOAD_CONCURRENCY = 2;
const OUTPUT_ROOT = resolve("datasets/sid-set");
const IMAGE_ROOT = resolve(OUTPUT_ROOT, "images");
const PRODUCTION_SOURCE_COUNT = TRACK5_SPLIT_PLAN.reduce(
  (count, allocation) => count + allocation.perClass * 2,
  0,
);

const splitDefinitions = [
  { name: "train", rows: 210_000 },
  { name: "validation", rows: 30_000 },
];

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

async function fetchWithRetry(url, description, attempts = 12) {
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(url);
      if (response.ok) return response;
      lastError = new Error(`${description} returned HTTP ${response.status}.`);
      if (response.status !== 429 && response.status < 500) throw lastError;
      const retryAfter = Number(response.headers.get("retry-after"));
      if (Number.isFinite(retryAfter) && retryAfter > 0) {
        await delay(Math.min(120_000, retryAfter * 1_000));
      }
    } catch (error) {
      lastError = error;
    }
    if (attempt < attempts) await delay(Math.min(60_000, 1_000 * 2 ** (attempt - 1)));
  }
  throw new Error(`${description} failed after ${attempts} attempts: ${lastError.message}`);
}

async function fetchJson(url, description) {
  const response = await fetchWithRetry(url, description);
  return response.json();
}

async function cachedMetadataPage(datasetSplit, page, url, description) {
  const cachePath = resolve(OUTPUT_ROOT, "metadata-cache", datasetSplit, `${page}.json`);
  try {
    return JSON.parse(await readFile(cachePath, "utf8"));
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
  }
  const value = await fetchJson(url, description);
  await mkdir(dirname(cachePath), { recursive: true });
  const temporary = `${cachePath}.part`;
  await writeFile(temporary, `${JSON.stringify(value)}\n`, "utf8");
  await rename(temporary, cachePath);
  return value;
}

function metadataRecord(datasetSplit, apiRow) {
  const row = apiRow.row;
  if (
    typeof row?.img_id !== "string" ||
    !Number.isSafeInteger(row.label) ||
    typeof row.image?.src !== "string"
  ) {
    throw new Error(`${datasetSplit} row ${apiRow.row_idx} has incomplete SID_Set metadata.`);
  }
  const classDirectory = row.label === 0 ? "authentic" : "full-synthetic";
  const imagePath = `${datasetSplit}/${classDirectory}/${row.img_id}.jpg`;
  return {
    sourceId: `sid-set:${row.img_id}`,
    datasetSplit,
    rowIndex: apiRow.row_idx,
    imageUrl: row.image.src,
    inventory: {
      img_id: row.img_id,
      image_path: imagePath,
      label: row.label,
      dataset_split: datasetSplit,
      width: row.width,
      height: row.height,
      provenance: {
        source_dataset: "SID_Set",
        source_reference: `${DATASET}@${REVISION}:${datasetSplit}:${apiRow.row_idx}:${row.img_id}`,
        license: "CC-BY-4.0",
      },
    },
  };
}

function rowsUrl(datasetSplit, offset, length = PAGE_SIZE) {
  return (
    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}` +
    `&config=default&split=${datasetSplit}&offset=${offset}&length=${length}`
  );
}

async function scanSplit({ name, rows }) {
  const pageCount = Math.ceil(rows / PAGE_SIZE);
  const records = [];
  let nextPage = 0;
  let completedPages = 0;

  async function worker() {
    while (nextPage < pageCount) {
      const page = nextPage;
      nextPage += 1;
      const offset = page * PAGE_SIZE;
      const url = rowsUrl(name, offset);
      const result = await cachedMetadataPage(
        name,
        page,
        url,
        `${name} metadata page ${page + 1}/${pageCount}`,
      );
      if (result.num_rows_total !== rows) {
        throw new Error(
          `${name} reports ${result.num_rows_total} rows; pinned contract expects ${rows}.`,
        );
      }
      for (const apiRow of result.rows) {
        if (apiRow.row.label === 0 || apiRow.row.label === 1) {
          records.push(metadataRecord(name, apiRow));
        }
      }
      completedPages += 1;
      if (completedPages % 100 === 0 || completedPages === pageCount) {
        process.stdout.write(
          `Scanned ${name} metadata: ${completedPages}/${pageCount} pages.\n`,
        );
      }
    }
  }

  await Promise.all(Array.from({ length: METADATA_CONCURRENCY }, () => worker()));
  return records;
}

async function existingNonemptyFile(path) {
  try {
    return (await stat(path)).size > 0;
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
}

async function downloadImage(record, attempts = 12) {
  const target = resolve(IMAGE_ROOT, record.inventory.image_path);
  if (await existingNonemptyFile(target)) return "reused";
  await mkdir(dirname(target), { recursive: true });
  const temporary = `${target}.part`;
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    await rm(temporary, { force: true });
    try {
      const response = await fetchWithRetry(record.imageUrl, `image ${record.sourceId}`);
      if (response.body === null) throw new Error(`image ${record.sourceId} returned no body.`);
      await pipeline(
        Readable.fromWeb(response.body),
        createWriteStream(temporary, { flags: "wx" }),
      );
      if (!(await existingNonemptyFile(temporary))) {
        throw new Error(`image ${record.sourceId} downloaded as an empty file.`);
      }
      await rename(temporary, target);
      return "downloaded";
    } catch (error) {
      lastError = error;
      await rm(temporary, { force: true });
      if (attempt < attempts) {
        await delay(Math.min(60_000, 1_000 * 2 ** (attempt - 1)));
      }
    }
  }
  throw new Error(
    `image ${record.sourceId} transfer failed after ${attempts} attempts: ${lastError.message}`,
  );
}

async function downloadSelected(records) {
  let complete = 0;
  let downloaded = 0;
  let reused = 0;
  const groups = new Map();
  for (const record of records) {
    const page = Math.floor(record.rowIndex / PAGE_SIZE);
    const key = `${record.datasetSplit}:${page}`;
    const group = groups.get(key) ?? { datasetSplit: record.datasetSplit, page, records: [] };
    group.records.push(record);
    groups.set(key, group);
  }
  const orderedGroups = [...groups.values()].toSorted(
    (left, right) =>
      left.datasetSplit.localeCompare(right.datasetSplit) || left.page - right.page,
  );

  await forEachWithConcurrency(
    orderedGroups,
    DOWNLOAD_GROUP_CONCURRENCY,
    async (group) => {
      const pending = [];
      for (const record of group.records) {
        const target = resolve(IMAGE_ROOT, record.inventory.image_path);
        if (await existingNonemptyFile(target)) {
          reused += 1;
          complete += 1;
        } else {
          pending.push(record);
        }
      }
      if (pending.length > 0) {
        const offset = group.page * PAGE_SIZE;
        const freshPage = await fetchJson(
          rowsUrl(group.datasetSplit, offset),
          `${group.datasetSplit} download page ${group.page + 1}`,
        );
        const urlsByRowIndex = new Map(
          freshPage.rows.map((apiRow) => [apiRow.row_idx, apiRow.row.image?.src]),
        );
        for (const record of pending) {
          record.imageUrl = urlsByRowIndex.get(record.rowIndex);
          if (typeof record.imageUrl !== "string") {
            throw new Error(
              `Fresh URL missing for ${record.datasetSplit} row ${record.rowIndex}.`,
            );
          }
        }
        await forEachWithConcurrency(
          pending,
          DOWNLOAD_CONCURRENCY,
          async (record) => {
            const status = await downloadImage(record);
            if (status === "downloaded") downloaded += 1;
            else reused += 1;
            complete += 1;
          },
        );
      }
      if (complete % 100 < group.records.length || complete === records.length) {
        process.stdout.write(
          `Images: ${complete}/${records.length} complete (${downloaded} downloaded, ${reused} reused).\n`,
        );
      }
    },
  );
}

async function main() {
  await mkdir(OUTPUT_ROOT, { recursive: true });
  const datasetInfo = await fetchJson(
    `https://huggingface.co/api/datasets/${DATASET}`,
    "SID_Set repository metadata",
  );
  if (datasetInfo.sha !== REVISION) {
    throw new Error(
      `SID_Set currently resolves to ${datasetInfo.sha}; expected pinned revision ${REVISION}.`,
    );
  }

  const scanned = [];
  for (const definition of splitDefinitions) {
    for (const record of await scanSplit(definition)) {
      scanned.push(record);
    }
  }

  const validationSourceIds = new Set(
    scanned
      .filter(({ datasetSplit }) => datasetSplit === "validation")
      .map(({ sourceId }) => sourceId),
  );
  const excludedTrainingDuplicates = scanned.filter(
    ({ datasetSplit, sourceId }) =>
      datasetSplit === "train" && validationSourceIds.has(sourceId),
  ).length;
  process.stdout.write(
    `Excluded ${excludedTrainingDuplicates} training row(s) whose stable source ID also occurs in validation.\n`,
  );

  const byInventoryKey = new Map();
  for (const record of scanned) {
    const key = `${record.datasetSplit}\0${record.inventory.img_id}`;
    if (byInventoryKey.has(key)) {
      throw new Error(`SID_Set repeats source identity ${record.sourceId} within one split.`);
    }
    byInventoryKey.set(key, record);
  }

  const candidateInventory = selectTrack5CandidateInventoryRecords(
    scanned.map(({ inventory }) => inventory),
    {
      reservePerClass: RESERVE_PER_CLASS,
      splitSeed: SPLIT_SEED,
    },
  );
  const selected = candidateInventory.map((inventory) =>
    byInventoryKey.get(`${inventory.dataset_split}\0${inventory.img_id}`),
  );
  if (selected.some((record) => record === undefined)) {
    throw new Error("Candidate source metadata could not be resolved to a download URL.");
  }
  await downloadSelected(selected);

  const inventoryLines = selected
    .map(({ inventory }) => {
      const { exact_sha256, height, perceptual_hash, width, ...uninspected } = inventory;
      void exact_sha256;
      void height;
      void perceptual_hash;
      void width;
      return JSON.stringify(uninspected);
    })
    .join("\n");
  await writeFile(resolve(OUTPUT_ROOT, "inventory.jsonl"), `${inventoryLines}\n`, "utf8");
  await writeFile(
    resolve(OUTPUT_ROOT, "DOWNLOAD.json"),
    `${JSON.stringify(
      {
        downloaded_at: new Date().toISOString(),
        dataset: DATASET,
        dataset_revision: REVISION,
        split_seed: SPLIT_SEED,
        candidate_selection_contract_version: "sid-set-candidate-selection-v1",
        production_selection_contract_version: "track5-source-selection-v2",
        production_source_count: PRODUCTION_SOURCE_COUNT,
        reserve_per_class: RESERVE_PER_CLASS,
        candidate_source_count: selected.length,
        inventory: "inventory.jsonl",
        image_root: "images",
        license: "CC-BY-4.0",
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  process.stdout.write(
    `SID_Set candidate pool ready: ${selected.length} images for ${PRODUCTION_SOURCE_COUNT}-source production selection.\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
