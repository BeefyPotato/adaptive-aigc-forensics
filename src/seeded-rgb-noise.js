import { Worker } from "node:worker_threads";

import { deterministicStandardNormal } from "./deterministic-random.js";

export const PARALLEL_NOISE_MINIMUM_CHANNELS = 262_144;

function validateInputs(input, { seed, sigma, startIndex = 0, workerCount = 1 }) {
  if (!Buffer.isBuffer(input)) {
    throw new TypeError("Seeded RGB noise input must be a Buffer.");
  }
  for (const [name, value] of [["seed", seed], ["startIndex", startIndex]]) {
    if (!Number.isSafeInteger(value) || value < 0) {
      throw new TypeError(`Seeded RGB noise ${name} must be a non-negative safe integer.`);
    }
  }
  if (!Number.isFinite(sigma) || sigma < 0) {
    throw new TypeError("Seeded RGB noise sigma must be finite and non-negative.");
  }
  if (!Number.isSafeInteger(workerCount) || workerCount <= 0) {
    throw new TypeError("Seeded RGB noise workerCount must be a positive safe integer.");
  }
  if (!Number.isSafeInteger(startIndex + input.length)) {
    throw new TypeError("Seeded RGB noise global channel range exceeds safe integer bounds.");
  }
}

export function applySeededRgbNoiseBytesSequentially(
  input,
  { seed, sigma, startIndex = 0 },
) {
  validateInputs(input, { seed, sigma, startIndex, workerCount: 1 });
  const output = Buffer.allocUnsafe(input.length);
  for (let offset = 0; offset < input.length; offset += 1) {
    const globalChannelIndex = startIndex + offset;
    const noise = deterministicStandardNormal(seed, globalChannelIndex) * sigma * 255;
    output[offset] = Math.max(0, Math.min(255, Math.round(input[offset] + noise)));
  }
  return output;
}

function runNoiseWorker(input, { seed, sigma, startIndex }) {
  const ownedInput = Uint8Array.from(input);
  const expectedLength = ownedInput.byteLength;
  const worker = new Worker(new URL("./seeded-rgb-noise-worker.js", import.meta.url), {
    workerData: {
      input: ownedInput.buffer,
      seed,
      sigma,
      startIndex,
    },
    transferList: [ownedInput.buffer],
  });
  return new Promise((resolve, reject) => {
    let completed = false;
    let received;
    const fail = (error) => {
      if (completed) return;
      completed = true;
      reject(error);
    };
    worker.once("message", (message) => {
      if (!(message instanceof ArrayBuffer) || message.byteLength !== expectedLength) {
        void worker.terminate();
        fail(new Error("Seeded RGB noise worker returned an incompatible byte range."));
        return;
      }
      received = Buffer.from(message);
    });
    worker.once("error", fail);
    worker.once("exit", (code) => {
      if (completed) return;
      if (code !== 0) {
        fail(new Error(`Seeded RGB noise worker exited with code ${code}.`));
      } else if (received === undefined) {
        fail(new Error("Seeded RGB noise worker exited without returning bytes."));
      } else {
        completed = true;
        resolve(received);
      }
    });
  });
}

export async function applySeededRgbNoiseBytes(
  input,
  { seed, sigma, workerCount = 1 },
) {
  validateInputs(input, { seed, sigma, workerCount });
  if (workerCount === 1 || input.length < PARALLEL_NOISE_MINIMUM_CHANNELS) {
    return applySeededRgbNoiseBytesSequentially(input, { seed, sigma });
  }

  const effectiveWorkerCount = Math.min(workerCount, input.length);
  const chunkSize = Math.ceil(input.length / effectiveWorkerCount);
  const ranges = [];
  for (let startIndex = 0; startIndex < input.length; startIndex += chunkSize) {
    const endIndex = Math.min(input.length, startIndex + chunkSize);
    ranges.push({ startIndex, endIndex });
  }
  const chunks = await Promise.all(
    ranges.map(({ startIndex, endIndex }) =>
      runNoiseWorker(input.subarray(startIndex, endIndex), {
        seed,
        sigma,
        startIndex,
      }),
    ),
  );
  const output = Buffer.allocUnsafe(input.length);
  for (let index = 0; index < chunks.length; index += 1) {
    chunks[index].copy(output, ranges[index].startIndex);
  }
  return output;
}
