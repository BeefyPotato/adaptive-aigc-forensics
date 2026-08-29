import { isAbsolute } from "node:path";

import {
  ContractError,
  requireFields,
  requireNonemptyString,
  requireObject,
} from "./contract-validation.js";

export function requireRelativeImagePath(imagePath, contractName) {
  requireNonemptyString(imagePath, "image_path", contractName);
  const normalizedPath = imagePath.replaceAll("\\", "/");
  if (
    isAbsolute(imagePath) ||
    /^[a-z]:\//iu.test(normalizedPath) ||
    normalizedPath.startsWith("//") ||
    normalizedPath.split("/").includes("..")
  ) {
    throw new ContractError(`${contractName}.image_path must stay relative to the dataset root.`);
  }
}

export function validateSourceInventoryRecord(record, index) {
  const contractName = `SID_Set source inventory record ${index}`;
  requireObject(record, contractName);
  requireFields(
    record,
    ["img_id", "image_path", "label", "dataset_split", "provenance"],
    contractName,
  );
  requireNonemptyString(record.img_id, "img_id", contractName);
  requireRelativeImagePath(record.image_path, contractName);
  if (record.label !== 0 && record.label !== 1 && record.label !== 2) {
    throw new ContractError(`${contractName}.label must be 0, 1, or 2.`);
  }
  if (record.dataset_split !== "train" && record.dataset_split !== "validation") {
    throw new ContractError(`${contractName}.dataset_split must be train or validation.`);
  }
  requireObject(record.provenance, `${contractName}.provenance`);
  for (const field of ["source_dataset", "source_reference", "license"]) {
    requireNonemptyString(record.provenance[field], field, `${contractName}.provenance`);
  }
  return contractName;
}
