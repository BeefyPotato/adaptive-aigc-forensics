import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import sharp from "sharp";

import {
  ContractError,
  requireNonemptyString,
} from "./contract-validation.js";
import { validateSourceInventoryRecord } from "./source-inventory-contract.js";

function toFilePath(value) {
  return value instanceof URL ? fileURLToPath(value) : value;
}

function resolveImagePath(datasetRoot, imagePath, contractName) {
  requireNonemptyString(imagePath, "image_path", contractName);
  const root = resolve(toFilePath(datasetRoot));
  if (isAbsolute(imagePath)) {
    throw new ContractError(`${contractName}.image_path must stay relative to the dataset root.`);
  }
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
  let nextIndex = 0;
  async function worker() {
    while (nextIndex < records.length) {
      const index = nextIndex;
      nextIndex += 1;
      inspected[index] = await inspectRecord(records[index], index, datasetRoot);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, records.length) }, () => worker()),
  );
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
