import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";

import { applyCorruption } from "../src/corruption-harness.js";
import {
  PARALLEL_NOISE_MINIMUM_CHANNELS,
  applySeededRgbNoiseBytes,
  applySeededRgbNoiseBytesSequentially,
} from "../src/seeded-rgb-noise.js";

function patternedBytes(length) {
  return Buffer.from(Array.from({ length }, (_, index) => (index * 37 + 11) % 256));
}

test("sequential seeded RGB noise preserves the finalized corruption bytes", () => {
  const output = applySeededRgbNoiseBytesSequentially(patternedBytes(64), {
    seed: 23,
    sigma: 0.1,
  });

  assert.equal(
    createHash("sha256").update(output).digest("hex"),
    "d8c7b935afb217126f5113e2552896784bd2f04d2eae4928ecca00fe0844805a",
  );
  assert.deepEqual(
    [...output],
    [
      0, 11, 98, 118, 148, 250, 246, 10, 67, 87, 81, 179, 224, 210, 0, 0,
      113, 119, 199, 206, 197, 10, 43, 66, 149, 168, 183, 254, 45, 0, 78, 52,
      155, 203, 242, 45, 65, 129, 187, 187, 224, 255, 16, 77, 93, 104, 225, 181,
      226, 0, 56, 125, 163, 172, 222, 237, 58, 50, 87, 128, 213, 236, 7, 43,
    ],
  );
});

test("parallel seeded RGB noise is byte-identical across uneven worker boundaries", async () => {
  const input = patternedBytes(PARALLEL_NOISE_MINIMUM_CHANNELS + 17);
  const sequential = applySeededRgbNoiseBytesSequentially(input, {
    seed: 61,
    sigma: 0.05,
  });
  const parallel = await applySeededRgbNoiseBytes(input, {
    seed: 61,
    sigma: 0.05,
    workerCount: 5,
  });

  assert.deepEqual(parallel, sequential);
});

test("corruption output is independent of noise worker concurrency", async () => {
  const width = 257;
  const height = Math.ceil(PARALLEL_NOISE_MINIMUM_CHANNELS / (width * 3)) + 1;
  const observation = {
    data: patternedBytes(width * height * 3),
    width,
    height,
    channels: 3,
  };
  const variant = {
    condition_family: "noise",
    corruption_parameters: { sigma: 0.1, color_space: "rgb-0-1" },
    corruption_seed: 23,
  };

  const sequential = await applyCorruption(observation, variant, { noiseWorkerCount: 1 });
  const parallel = await applyCorruption(observation, variant, { noiseWorkerCount: 4 });

  assert.deepEqual(parallel, sequential);
});
