import { readFileSync } from "node:fs";

export class ContractError extends Error {
  constructor(message) {
    super(message);
    this.name = "ContractError";
  }
}

export function requireObject(value, contractName) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    throw new ContractError(`${contractName} must be a JSON object.`);
  }
  return value;
}

export function requireFields(value, fields, contractName) {
  const missing = fields.filter((field) => !(field in value));
  if (missing.length > 0) {
    throw new ContractError(`${contractName} is missing required field(s): ${missing.join(", ")}.`);
  }
}

export function requireNonemptyString(value, field, contractName) {
  if (typeof value !== "string" || value.length === 0) {
    throw new ContractError(`${contractName}.${field} must be a non-empty string.`);
  }
}

export function requireNonnegativeInteger(value, field, contractName) {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new ContractError(`${contractName}.${field} must be a non-negative safe integer.`);
  }
}

export function requirePositiveFiniteNumber(value, field, contractName) {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ContractError(`${contractName}.${field} must be a positive finite number.`);
  }
}

export function assertCompatible(actual, expected, field, contractName) {
  if (actual !== expected) {
    throw new ContractError(
      `${contractName}.${field} is incompatible: received ${JSON.stringify(actual)}; expected ${JSON.stringify(expected)}.`,
    );
  }
}

export function readJson(path, contractName) {
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new ContractError(`${contractName} could not be read from ${path}: ${error.message}`);
  }
  try {
    return JSON.parse(text);
  } catch (error) {
    throw new ContractError(`${contractName} at ${path} is not valid JSON: ${error.message}`);
  }
}
