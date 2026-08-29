#!/usr/bin/env node

import { ContractError } from "./contracts.js";
import { runFixture } from "./pipeline.js";

function usage() {
  return "Usage: node ./src/cli.js run-fixture --config <config.json> --output-dir <directory>";
}

function parseArguments(arguments_) {
  if (arguments_[0] !== "run-fixture") {
    throw new ContractError(usage());
  }
  const values = new Map();
  for (let index = 1; index < arguments_.length; index += 2) {
    const option = arguments_[index];
    const value = arguments_[index + 1];
    if (!option?.startsWith("--") || value === undefined) {
      throw new ContractError(usage());
    }
    if (values.has(option)) {
      throw new ContractError(`Option ${option} was provided more than once.`);
    }
    values.set(option, value);
  }
  for (const option of values.keys()) {
    if (option !== "--config" && option !== "--output-dir") {
      throw new ContractError(`Unknown option ${option}. ${usage()}`);
    }
  }
  if (!values.has("--config") || !values.has("--output-dir")) {
    throw new ContractError(usage());
  }
  return { configPath: values.get("--config"), outputDirectory: values.get("--output-dir") };
}

try {
  const { configPath, outputDirectory } = parseArguments(process.argv.slice(2));
  const result = runFixture(configPath, outputDirectory);
  process.stdout.write(
    `Wrote ${result.predictionCount} deterministic prediction records to ${result.outputDirectory} (numeric tolerance ${result.numericTolerance}).\n`,
  );
} catch (error) {
  const message = error instanceof Error ? error.message : String(error);
  process.stderr.write(`Fixture failed: ${message}\n`);
  process.exitCode = 1;
}
