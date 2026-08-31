import { parentPort, workerData } from "node:worker_threads";

import { applySeededRgbNoiseBytesSequentially } from "./seeded-rgb-noise.js";

if (parentPort === null) {
  throw new Error("Seeded RGB noise worker requires a parent port.");
}

const output = applySeededRgbNoiseBytesSequentially(Buffer.from(workerData.input), {
  seed: workerData.seed,
  sigma: workerData.sigma,
  startIndex: workerData.startIndex,
});
const transferableOutput = Uint8Array.from(output);
parentPort.postMessage(transferableOutput.buffer, [transferableOutput.buffer]);
