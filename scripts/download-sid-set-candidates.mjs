import { createWriteStream } from "node:fs";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";

import {
  candidateFileMatches,
  fetchWithRetry,
  forEachWithConcurrency,
  validatePinnedCandidatePool,
  validateSidSetRowsPage,
} from "../src/source-inventory.js";
import { TRACK5_SPLIT_PLAN } from "../src/track5-manifest.js";

const DATASET = "saberzl/SID_Set";
const REVISION = "dc03ead57929879319ce30a82bfcfb8d317b10bd";
const PAGE_SIZE = 100;
const DOWNLOAD_GROUP_CONCURRENCY = 4;
const DOWNLOAD_CONCURRENCY = 2;
const CANDIDATE_METADATA_PATH = resolve(
  "metadata/sid-set-candidate-pool-v1.json",
);
const CANDIDATE_RECORDS_PATH = resolve(
  "metadata/sid-set-candidates-v1.jsonl",
);
const OUTPUT_ROOT = resolve("datasets/sid-set");
const IMAGE_ROOT = resolve(OUTPUT_ROOT, "images");
const PRODUCTION_SOURCE_COUNT = TRACK5_SPLIT_PLAN.reduce(
  (count, allocation) => count + allocation.perClass * 2,
  0,
);
const SPLIT_ROW_COUNTS = Object.freeze({ train: 210_000, validation: 30_000 });

function rowsUrl(datasetSplit, offset) {
  return (
    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}` +
    `&config=default&split=${datasetSplit}&offset=${offset}&length=${PAGE_SIZE}`
  );
}

function parseJsonLines(text, description) {
  const records = [];
  for (const [index, line] of text.split(/\r?\n/u).entries()) {
    if (line.trim() === "") continue;
    try {
      records.push(JSON.parse(line));
    } catch (error) {
      throw new Error(`${description} line ${index + 1} is invalid JSON: ${error.message}`);
    }
  }
  return records;
}

async function loadCandidatePool() {
  const metadata = JSON.parse(await readFile(CANDIDATE_METADATA_PATH, "utf8"));
  if (metadata.records_path !== "sid-set-candidates-v1.jsonl") {
    throw new Error(
      `Candidate pool records_path received ${metadata.records_path}; expected sid-set-candidates-v1.jsonl.`,
    );
  }
  const records = parseJsonLines(
    await readFile(CANDIDATE_RECORDS_PATH, "utf8"),
    "SID_Set candidate pool",
  );
  return validatePinnedCandidatePool(metadata, records, {
    dataset: DATASET,
    datasetRevision: REVISION,
  });
}

async function fetchCandidatePage(group) {
  const offset = group.page * PAGE_SIZE;
  const response = await fetchWithRetry(rowsUrl(group.datasetSplit, offset), {
    description: `${group.datasetSplit} candidate page ${group.page + 1}`,
  });
  let page;
  try {
    page = await response.json();
  } catch (error) {
    throw new Error(
      `${group.datasetSplit} candidate page ${group.page + 1} returned invalid JSON: ${error.message}`,
    );
  }
  return validateSidSetRowsPage(page, {
    datasetSplit: group.datasetSplit,
    expectedCandidates: group.records,
    expectedTotalRows: SPLIT_ROW_COUNTS[group.datasetSplit],
    offset,
    pageSize: PAGE_SIZE,
  });
}

async function downloadCandidate(candidate, imageUrl) {
  const target = resolve(IMAGE_ROOT, candidate.image_path);
  if (await candidateFileMatches(target, candidate)) return "reused";

  await mkdir(dirname(target), { recursive: true });
  const temporary = `${target}.${process.pid}.part`;
  await rm(temporary, { force: true });
  try {
    const response = await fetchWithRetry(imageUrl, {
      description: `image sid-set:${candidate.img_id}`,
    });
    if (response.body === null) {
      throw new Error(`image sid-set:${candidate.img_id} returned no body.`);
    }
    await pipeline(
      Readable.fromWeb(response.body),
      createWriteStream(temporary, { flags: "wx" }),
    );
    if (!(await candidateFileMatches(temporary, candidate))) {
      throw new Error(
        `image sid-set:${candidate.img_id} does not match its pinned byte length and SHA-256.`,
      );
    }
    await rm(target, { force: true });
    await rename(temporary, target);
    return "downloaded";
  } catch (error) {
    await rm(temporary, { force: true });
    throw error;
  }
}

async function prepareCandidates(records) {
  let complete = 0;
  let downloaded = 0;
  let reused = 0;
  const pendingGroups = new Map();

  await forEachWithConcurrency(records, 8, async (candidate) => {
    const target = resolve(IMAGE_ROOT, candidate.image_path);
    if (await candidateFileMatches(target, candidate)) {
      reused += 1;
      complete += 1;
      return;
    }
    const page = Math.floor(candidate.row_index / PAGE_SIZE);
    const key = `${candidate.dataset_split}:${page}`;
    const group = pendingGroups.get(key) ?? {
      datasetSplit: candidate.dataset_split,
      page,
      records: [],
    };
    group.records.push(candidate);
    pendingGroups.set(key, group);
  });

  const orderedGroups = [...pendingGroups.values()].toSorted(
    (left, right) =>
      left.datasetSplit.localeCompare(right.datasetSplit) || left.page - right.page,
  );
  if (orderedGroups.length === 0) {
    process.stdout.write(`Verified ${reused}/${records.length} existing candidates.\n`);
    return { downloaded, reused };
  }

  await forEachWithConcurrency(
    orderedGroups,
    DOWNLOAD_GROUP_CONCURRENCY,
    async (group) => {
      const urlsByRowIndex = await fetchCandidatePage(group);
      await forEachWithConcurrency(
        group.records,
        DOWNLOAD_CONCURRENCY,
        async (candidate) => {
          const status = await downloadCandidate(
            candidate,
            urlsByRowIndex.get(candidate.row_index),
          );
          if (status === "downloaded") downloaded += 1;
          else reused += 1;
          complete += 1;
          if (complete % 100 === 0 || complete === records.length) {
            process.stdout.write(
              `Candidates: ${complete}/${records.length} complete ` +
                `(${downloaded} downloaded, ${reused} verified).\n`,
            );
          }
        },
      );
    },
  );
  return { downloaded, reused };
}

function toInventoryRecord(candidate) {
  const {
    byte_length: byteLength,
    candidate_schema_version: candidateSchemaVersion,
    exact_sha256: exactSha256,
    row_index: rowIndex,
    ...inventory
  } = candidate;
  void byteLength;
  void candidateSchemaVersion;
  void exactSha256;
  void rowIndex;
  return inventory;
}

async function main() {
  const candidatePool = await loadCandidatePool();
  await mkdir(OUTPUT_ROOT, { recursive: true });
  const transfer = await prepareCandidates(candidatePool.records);

  const inventoryLines = candidatePool.records
    .map((candidate) => JSON.stringify(toInventoryRecord(candidate)))
    .join("\n");
  await writeFile(
    resolve(OUTPUT_ROOT, "inventory.jsonl"),
    `${inventoryLines}\n`,
    "utf8",
  );
  await writeFile(
    resolve(OUTPUT_ROOT, "DOWNLOAD.json"),
    `${JSON.stringify(
      {
        downloaded_at: new Date().toISOString(),
        dataset: DATASET,
        dataset_revision: REVISION,
        candidate_pool_schema_version:
          candidatePool.metadata.candidate_pool_schema_version,
        production_selection_contract_version: "track5-source-selection-v2",
        production_source_count: PRODUCTION_SOURCE_COUNT,
        reserve_per_class: candidatePool.metadata.reserve_per_class,
        candidate_source_count: candidatePool.records.length,
        downloaded_this_run: transfer.downloaded,
        verified_existing_this_run: transfer.reused,
        inventory: "inventory.jsonl",
        image_root: "images",
        license: candidatePool.metadata.license,
      },
      null,
      2,
    )}\n`,
    "utf8",
  );
  process.stdout.write(
    `SID_Set candidate pool ready: ${candidatePool.records.length} verified images ` +
      `for ${PRODUCTION_SOURCE_COUNT}-source production selection.\n`,
  );
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
});
