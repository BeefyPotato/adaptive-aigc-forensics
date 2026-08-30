import { createHash } from "node:crypto";
import { createReadStream } from "node:fs";
import { readFile, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import {
  ContractError,
  requireFields,
  requireLowercaseHex,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { validateSourceInventoryRecord } from "./source-inventory-contract.js";

function toFilePath(value) {
  return value instanceof URL ? fileURLToPath(value) : value;
}

function delay(milliseconds) {
  return new Promise((resolveDelay) => setTimeout(resolveDelay, milliseconds));
}

export async function fetchWithRetry(
  url,
  {
    description = "request",
    fetchImplementation = globalThis.fetch,
    delayImplementation = delay,
    attempts = 12,
    requestTimeoutMilliseconds = 60_000,
    totalTimeoutMilliseconds = 10 * 60_000,
  } = {},
) {
  requireNonemptyString(url, "url", "SID_Set transport options");
  requireNonemptyString(description, "description", "SID_Set transport options");
  if (typeof fetchImplementation !== "function") {
    throw new ContractError(
      "SID_Set transport options.fetchImplementation must be a function.",
    );
  }
  if (typeof delayImplementation !== "function") {
    throw new ContractError(
      "SID_Set transport options.delayImplementation must be a function.",
    );
  }
  for (const [field, value] of [
    ["attempts", attempts],
    ["requestTimeoutMilliseconds", requestTimeoutMilliseconds],
    ["totalTimeoutMilliseconds", totalTimeoutMilliseconds],
  ]) {
    if (!Number.isSafeInteger(value) || value <= 0) {
      throw new ContractError(
        `SID_Set transport options.${field} must be a positive safe integer.`,
      );
    }
  }

  const deadline = Date.now() + totalTimeoutMilliseconds;
  let lastError;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) break;
    const controller = new AbortController();
    const timeout = setTimeout(
      () => controller.abort(new Error(`${description} timed out.`)),
      Math.min(requestTimeoutMilliseconds, remaining),
    );
    let response;
    try {
      response = await fetchImplementation(url, { signal: controller.signal });
    } catch (error) {
      lastError = error;
    } finally {
      clearTimeout(timeout);
    }

    let retryAfterMilliseconds;
    if (response !== undefined) {
      if (response.ok) return response;
      lastError = new Error(`${description} returned HTTP ${response.status}.`);
      if (response.status !== 429 && response.status < 500) {
        await response.body?.cancel();
        throw new ContractError(lastError.message);
      }
      const retryAfterSeconds = Number(response.headers.get("retry-after"));
      if (Number.isFinite(retryAfterSeconds) && retryAfterSeconds > 0) {
        retryAfterMilliseconds = retryAfterSeconds * 1_000;
      }
      await response.body?.cancel();
    }

    if (attempt < attempts) {
      const remainingAfterRequest = deadline - Date.now();
      if (remainingAfterRequest <= 0) break;
      const backoff = retryAfterMilliseconds ?? 1_000 * 2 ** (attempt - 1);
      await delayImplementation(Math.min(60_000, backoff, remainingAfterRequest));
    }
  }
  throw new ContractError(
    `${description} failed after at most ${attempts} attempts: ${lastError?.message ?? "overall timeout exceeded"}`,
  );
}

export async function candidateFileMatches(path, candidate) {
  const candidatePath = toFilePath(path);
  let fileStat;
  try {
    fileStat = await stat(candidatePath);
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
  if (!fileStat.isFile() || fileStat.size !== candidate.byte_length) return false;

  const hash = createHash("sha256");
  try {
    for await (const chunk of createReadStream(candidatePath)) hash.update(chunk);
  } catch (error) {
    if (error.code === "ENOENT") return false;
    throw error;
  }
  return hash.digest("hex") === candidate.exact_sha256;
}

export function validatePinnedCandidatePool(
  metadata,
  records,
  { dataset, datasetRevision },
) {
  requireObject(metadata, "SID_Set candidate pool metadata");
  requireFields(
    metadata,
    [
      "candidate_pool_schema_version",
      "dataset",
      "dataset_revision",
      "split_seed",
      "reserve_per_class",
      "source_count",
      "bucket_counts",
      "records_path",
      "license",
    ],
    "SID_Set candidate pool metadata",
  );
  if (metadata.candidate_pool_schema_version !== "sid-set-candidate-pool-v1") {
    throw new ContractError(
      "SID_Set candidate pool metadata.candidate_pool_schema_version must be sid-set-candidate-pool-v1.",
    );
  }
  requireNonemptyString(dataset, "dataset", "SID_Set candidate pool options");
  requireNonemptyString(
    datasetRevision,
    "datasetRevision",
    "SID_Set candidate pool options",
  );
  if (metadata.dataset !== dataset) {
    throw new ContractError(
      `SID_Set candidate pool metadata.dataset received ${metadata.dataset}; expected ${dataset}.`,
    );
  }
  if (metadata.dataset_revision !== datasetRevision) {
    throw new ContractError(
      `SID_Set candidate pool metadata.dataset_revision received ${metadata.dataset_revision}; expected ${datasetRevision}.`,
    );
  }
  for (const field of ["split_seed", "reserve_per_class", "source_count"]) {
    requireNonnegativeInteger(
      metadata[field],
      field,
      "SID_Set candidate pool metadata",
    );
  }
  requireObject(metadata.bucket_counts, "SID_Set candidate pool metadata.bucket_counts");
  requireNonemptyString(
    metadata.records_path,
    "records_path",
    "SID_Set candidate pool metadata",
  );
  requireNonemptyString(metadata.license, "license", "SID_Set candidate pool metadata");
  if (!Array.isArray(records)) {
    throw new ContractError("SID_Set candidate pool records must be an array.");
  }
  if (records.length !== metadata.source_count) {
    throw new ContractError(
      `SID_Set candidate pool has ${records.length} records; metadata declares ${metadata.source_count}.`,
    );
  }

  const seenSourceIds = new Set();
  const seenPaths = new Set();
  const actualBucketCounts = {};
  const normalizedRecords = records.map((record, index) => {
    const contractName = validateSourceInventoryRecord(record, index);
    requireFields(
      record,
      [
        "candidate_schema_version",
        "row_index",
        "byte_length",
        "exact_sha256",
      ],
      contractName,
    );
    if (record.candidate_schema_version !== "sid-set-candidate-v1") {
      throw new ContractError(
        `${contractName}.candidate_schema_version must be sid-set-candidate-v1.`,
      );
    }
    if (record.label !== 0 && record.label !== 1) {
      throw new ContractError(`${contractName}.label must be 0 or 1.`);
    }
    requireNonnegativeInteger(record.row_index, "row_index", contractName);
    requireNonnegativeInteger(record.byte_length, "byte_length", contractName);
    if (record.byte_length === 0) {
      throw new ContractError(`${contractName}.byte_length must be positive.`);
    }
    requireLowercaseHex(record.exact_sha256, "exact_sha256", 64, contractName);
    const sourceId = `sid-set:${record.img_id}`;
    if (seenSourceIds.has(sourceId)) {
      throw new ContractError(`SID_Set candidate pool repeats source identity ${sourceId}.`);
    }
    if (seenPaths.has(record.image_path)) {
      throw new ContractError(
        `SID_Set candidate pool repeats image path ${record.image_path}.`,
      );
    }
    const expectedReference =
      `${dataset}@${datasetRevision}:${record.dataset_split}:` +
      `${record.row_index}:${record.img_id}`;
    if (record.provenance.source_reference !== expectedReference) {
      throw new ContractError(
        `${contractName}.provenance.source_reference received ${record.provenance.source_reference}; expected ${expectedReference}.`,
      );
    }
    seenSourceIds.add(sourceId);
    seenPaths.add(record.image_path);
    const bucket = `${record.dataset_split}:class-${record.label}`;
    actualBucketCounts[bucket] = (actualBucketCounts[bucket] ?? 0) + 1;
    return Object.freeze({
      ...record,
      provenance: Object.freeze({ ...record.provenance }),
    });
  });

  const expectedBuckets = Object.entries(metadata.bucket_counts).toSorted();
  const actualBuckets = Object.entries(actualBucketCounts).toSorted();
  if (JSON.stringify(actualBuckets) !== JSON.stringify(expectedBuckets)) {
    throw new ContractError(
      "SID_Set candidate pool bucket counts do not match metadata.bucket_counts.",
    );
  }
  return Object.freeze({
    metadata: Object.freeze({
      ...metadata,
      bucket_counts: Object.freeze({ ...metadata.bucket_counts }),
    }),
    records: Object.freeze(normalizedRecords),
  });
}

export function validateSidSetRowsPage(
  page,
  {
    datasetSplit,
    expectedCandidates,
    expectedTotalRows,
    offset,
    pageSize,
  },
) {
  requireObject(page, "SID_Set rows page");
  requireFields(
    page,
    [
      "features",
      "rows",
      "num_rows_total",
      "num_rows_per_page",
      "partial",
    ],
    "SID_Set rows page",
  );
  if (datasetSplit !== "train" && datasetSplit !== "validation") {
    throw new ContractError("SID_Set rows page datasetSplit must be train or validation.");
  }
  if (!Array.isArray(expectedCandidates)) {
    throw new ContractError("SID_Set rows page expectedCandidates must be an array.");
  }
  for (const [field, value] of [
    ["expectedTotalRows", expectedTotalRows],
    ["offset", offset],
    ["pageSize", pageSize],
  ]) {
    requireNonnegativeInteger(value, field, "SID_Set rows page options");
  }
  if (pageSize === 0) {
    throw new ContractError("SID_Set rows page options.pageSize must be positive.");
  }
  if (page.partial !== false) {
    throw new ContractError("SID_Set rows page must declare partial as false.");
  }
  if (page.num_rows_total !== expectedTotalRows) {
    throw new ContractError(
      `SID_Set ${datasetSplit} rows page reports ${page.num_rows_total} total rows; expected ${expectedTotalRows}.`,
    );
  }
  if (page.num_rows_per_page !== pageSize) {
    throw new ContractError(
      `SID_Set rows page reports ${page.num_rows_per_page} rows per page; expected ${pageSize}.`,
    );
  }
  if (!Array.isArray(page.rows)) {
    throw new ContractError("SID_Set rows page.rows must be an array.");
  }
  const expectedPageLength = Math.min(pageSize, expectedTotalRows - offset);
  if (expectedPageLength < 0 || page.rows.length !== expectedPageLength) {
    throw new ContractError(
      `SID_Set rows page contains ${page.rows.length} rows; expected ${Math.max(0, expectedPageLength)} at offset ${offset}.`,
    );
  }

  const rowsByIndex = new Map();
  for (const [pageIndex, apiRow] of page.rows.entries()) {
    requireObject(apiRow, `SID_Set rows page row ${pageIndex}`);
    const expectedRowIndex = offset + pageIndex;
    if (apiRow.row_idx !== expectedRowIndex) {
      throw new ContractError(
        `SID_Set rows page row ${pageIndex}.row_idx received ${apiRow.row_idx}; expected ${expectedRowIndex}.`,
      );
    }
    requireObject(apiRow.row, `SID_Set rows page row ${pageIndex}.row`);
    if (!Array.isArray(apiRow.truncated_cells) || apiRow.truncated_cells.length > 0) {
      throw new ContractError(
        `SID_Set rows page row ${pageIndex}.truncated_cells must be an empty array.`,
      );
    }
    rowsByIndex.set(apiRow.row_idx, apiRow.row);
  }

  const urlsByRowIndex = new Map();
  for (const candidate of expectedCandidates) {
    if (candidate.dataset_split !== datasetSplit) {
      throw new ContractError(
        `SID_Set candidate ${candidate.img_id} belongs to ${candidate.dataset_split}; expected ${datasetSplit}.`,
      );
    }
    const row = rowsByIndex.get(candidate.row_index);
    if (row === undefined) {
      throw new ContractError(
        `SID_Set rows page is missing pinned row ${candidate.row_index}.`,
      );
    }
    if (row.img_id !== candidate.img_id) {
      throw new ContractError(
        `SID_Set row ${candidate.row_index}.img_id received ${row.img_id}; expected ${candidate.img_id}.`,
      );
    }
    if (row.label !== candidate.label) {
      throw new ContractError(
        `SID_Set row ${candidate.row_index}.label received ${row.label}; expected ${candidate.label}.`,
      );
    }
    requireObject(row.image, `SID_Set row ${candidate.row_index}.image`);
    requireNonemptyString(
      row.image.src,
      "src",
      `SID_Set row ${candidate.row_index}.image`,
    );
    urlsByRowIndex.set(candidate.row_index, row.image.src);
  }
  return urlsByRowIndex;
}

function resolveImagePath(datasetRoot, imagePath, contractName) {
  const root = resolve(toFilePath(datasetRoot));
  const resolvedImage = resolve(root, imagePath);
  const relativeImage = relative(root, resolvedImage);
  if (
    relativeImage === "" ||
    relativeImage === ".." ||
    relativeImage.startsWith(`..${process.platform === "win32" ? "\\" : "/"}`) ||
    isAbsolute(relativeImage)
  ) {
    throw new ContractError(`${contractName}.image_path must stay within the dataset root.`);
  }
  return { resolvedImage, relativeImage: relativeImage.replaceAll("\\", "/") };
}

function perceptualDifferenceHash(pixels, { channels, width, height }) {
  if (channels !== 1 || width !== 9 || height !== 8) {
    throw new ContractError(
      `Perceptual hash decoder returned ${width}x${height} with ${channels} channels; expected 9x8 grayscale.`,
    );
  }
  let bits = 0n;
  for (let row = 0; row < 8; row += 1) {
    for (let column = 0; column < 8; column += 1) {
      bits <<= 1n;
      const offset = row * 9 + column;
      if (pixels[offset] > pixels[offset + 1]) bits |= 1n;
    }
  }
  return bits.toString(16).padStart(16, "0");
}

export async function forEachWithConcurrency(values, concurrency, operation) {
  if (!Array.isArray(values)) {
    throw new ContractError("Concurrent worker values must be an array.");
  }
  if (!Number.isSafeInteger(concurrency) || concurrency <= 0) {
    throw new ContractError("Concurrent worker limit must be a positive safe integer.");
  }
  if (typeof operation !== "function") {
    throw new ContractError("Concurrent worker operation must be a function.");
  }
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < values.length) {
      const index = nextIndex;
      nextIndex += 1;
      await operation(values[index], index);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, values.length) }, () => worker()),
  );
}

async function inspectRecord(record, index, datasetRoot) {
  const contractName = validateSourceInventoryRecord(record, index);
  const { resolvedImage, relativeImage } = resolveImagePath(
    datasetRoot,
    record.image_path,
    contractName,
  );
  let file;
  try {
    file = await readFile(resolvedImage);
  } catch (error) {
    throw new ContractError(
      `${contractName}.image_path could not be read (${relativeImage}): ${error.message}`,
    );
  }

  let metadata;
  let hashPixels;
  try {
    metadata = await sharp(file, { failOn: "error" }).metadata();
    hashPixels = await sharp(file, { failOn: "error" })
      .autoOrient()
      .resize(9, 8, { fit: "fill", kernel: "lanczos3" })
      .greyscale()
      .raw()
      .toBuffer({ resolveWithObject: true });
  } catch (error) {
    throw new ContractError(
      `${contractName}.image_path is not a decodable image (${relativeImage}): ${error.message}`,
    );
  }
  if (!Number.isSafeInteger(metadata.width) || !Number.isSafeInteger(metadata.height)) {
    throw new ContractError(`${contractName}.image_path has no finite image dimensions.`);
  }
  const swapsDimensions = [5, 6, 7, 8].includes(metadata.orientation);
  const width = swapsDimensions ? metadata.height : metadata.width;
  const height = swapsDimensions ? metadata.width : metadata.height;

  return Object.freeze({
    ...record,
    image_path: relativeImage,
    width,
    height,
    exact_sha256: createHash("sha256").update(file).digest("hex"),
    perceptual_hash: perceptualDifferenceHash(hashPixels.data, hashPixels.info),
    provenance: Object.freeze({ ...record.provenance }),
  });
}

export async function inspectSourceInventory(
  records,
  { datasetRoot, concurrency = 4 },
) {
  if (!Array.isArray(records)) {
    throw new ContractError("SID_Set source inventory must be an array.");
  }
  requireNonemptyString(toFilePath(datasetRoot), "datasetRoot", "Source inventory options");
  if (!Number.isSafeInteger(concurrency) || concurrency <= 0) {
    throw new ContractError("Source inventory options.concurrency must be a positive safe integer.");
  }
  const inspected = new Array(records.length);
  await forEachWithConcurrency(records, concurrency, async (record, index) => {
    inspected[index] = await inspectRecord(record, index, datasetRoot);
  });
  return Object.freeze(inspected);
}

export async function loadAndInspectSourceInventory(
  inventoryPath,
  options,
) {
  let text;
  try {
    text = await readFile(inventoryPath, "utf8");
  } catch (error) {
    throw new ContractError(`Source inventory could not be read: ${error.message}`);
  }
  const records = [];
  for (const [index, line] of text.split(/\r?\n/u).entries()) {
    if (line.trim() === "") continue;
    try {
      records.push(JSON.parse(line));
    } catch (error) {
      throw new ContractError(`Source inventory line ${index + 1} is invalid JSON: ${error.message}`);
    }
  }
  if (records.length === 0) throw new ContractError("Source inventory contains no records.");
  return inspectSourceInventory(records, options);
}
