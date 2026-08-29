import assert from "node:assert/strict";
import { dirname, join, resolve } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  applyCorruption,
  decodeSourceImage,
} from "../src/corruption-harness.js";

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const sourceImagePath = join(
  repositoryRoot,
  "fixtures",
  "track5",
  "images",
  "checker.svg",
);

test("JPEG corruption performs a 4:2:0 encode/decode round trip at the declared quality", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const corrupted = await applyCorruption(source, {
    condition_family: "jpeg",
    corruption_parameters: { quality: 70, chroma_subsampling: "4:2:0" },
    corruption_seed: 23,
  });

  assert.equal(corrupted.width, source.width);
  assert.equal(corrupted.height, source.height);
  assert.equal(corrupted.channels, 3);
  assert.equal(corrupted.transform_details.intermediate_format, "jpeg");
  assert.equal(corrupted.transform_details.quality, 70);
  assert.equal(corrupted.transform_details.chroma_subsampling, "4:2:0");
  assert.deepEqual(corrupted.transform_details.sampling_factors, ["2x2", "1x1", "1x1"]);
  assert.notDeepEqual(corrupted.data, source.data);
});

function totalVariation(observation) {
  let variation = 0;
  for (let row = 0; row < observation.height; row += 1) {
    for (let column = 0; column < observation.width; column += 1) {
      const offset = (row * observation.width + column) * observation.channels;
      if (column + 1 < observation.width) {
        const rightOffset = offset + observation.channels;
        for (let channel = 0; channel < 3; channel += 1) {
          variation += Math.abs(observation.data[offset + channel] - observation.data[rightOffset + channel]);
        }
      }
      if (row + 1 < observation.height) {
        const belowOffset = offset + observation.width * observation.channels;
        for (let channel = 0; channel < 3; channel += 1) {
          variation += Math.abs(observation.data[offset + channel] - observation.data[belowOffset + channel]);
        }
      }
    }
  }
  return variation;
}

test("blur uses the declared sigma deterministically and reduces local variation", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const variant = {
    condition_family: "blur",
    corruption_parameters: { sigma: 1 },
    corruption_seed: 23,
  };
  const first = await applyCorruption(source, variant);
  const repeated = await applyCorruption(source, variant);

  assert.deepEqual(first.data, repeated.data);
  assert.equal(first.transform_details.sigma, 1);
  assert.equal(first.width, source.width);
  assert.equal(first.height, source.height);
  assert.ok(totalVariation(first) < totalVariation(source));
});

test("resize antialiases downscaling and restores the original dimensions with cubic interpolation", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const resized = await applyCorruption(source, {
    condition_family: "resize",
    corruption_parameters: {
      factor: 0.25,
      down_kernel: "lanczos3",
      up_kernel: "cubic",
    },
    corruption_seed: 23,
  });

  assert.equal(resized.width, source.width);
  assert.equal(resized.height, source.height);
  assert.equal(resized.transform_details.down_width, 4);
  assert.equal(resized.transform_details.down_height, 3);
  assert.equal(resized.transform_details.down_kernel, "lanczos3");
  assert.equal(resized.transform_details.up_kernel, "cubic");
  assert.notDeepEqual(resized.data, source.data);
});

test("noise uses the declared RGB sigma and seed with deterministic clamping", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const variant = {
    condition_family: "noise",
    corruption_parameters: { sigma: 0.1, color_space: "rgb-0-1" },
    corruption_seed: 23,
  };
  const first = await applyCorruption(source, variant);
  const repeated = await applyCorruption(source, variant);
  const differentSeed = await applyCorruption(source, { ...variant, corruption_seed: 24 });

  assert.deepEqual(first.data, repeated.data);
  assert.notDeepEqual(first.data, differentSeed.data);
  assert.notDeepEqual(first.data, source.data);
  assert.ok(first.data.every((channel) => channel >= 0 && channel <= 255));
  assert.equal(first.transform_details.sigma, 0.1);
  assert.equal(first.transform_details.seed, 23);
  assert.equal(first.transform_details.color_space, "rgb-0-1");

  const edgePixels = await applyCorruption(
    { data: Buffer.from([0, 0, 0, 255, 255, 255]), width: 2, height: 1, channels: 3 },
    { ...variant, corruption_seed: 0 },
  );
  assert.ok(edgePixels.data.includes(0));
  assert.ok(edgePixels.data.includes(255));
});

test("color corruption applies one declared property and factor at a time", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  for (const property of ["brightness", "contrast", "saturation"]) {
    for (const factor of [0.8, 1.2]) {
      const corrupted = await applyCorruption(source, {
        condition_family: "color",
        corruption_parameters: { property, factor },
        corruption_seed: 23,
      });
      assert.equal(corrupted.width, source.width);
      assert.equal(corrupted.height, source.height);
      assert.equal(corrupted.transform_details.property, property);
      assert.equal(corrupted.transform_details.factor, factor);
      assert.deepEqual(Object.keys(corrupted.transform_details).toSorted(), [
        "factor",
        "operation",
        "property",
      ]);
      assert.notDeepEqual(corrupted.data, source.data, `${property}-${factor}`);
    }
  }
});

test("center crop retains 80 percent geometry and restores the source dimensions", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const cropped = await applyCorruption(source, {
    condition_family: "crop",
    corruption_parameters: {
      retained_fraction: 0.8,
      position: "center",
      restoration_kernel: "cubic",
    },
    corruption_seed: 23,
  });

  assert.equal(cropped.width, 16);
  assert.equal(cropped.height, 12);
  assert.deepEqual(cropped.transform_details, {
    operation: "center-crop-round-trip",
    retained_fraction: 0.8,
    crop_left: 1,
    crop_top: 1,
    crop_width: 13,
    crop_height: 10,
    restoration_kernel: "cubic",
  });
  assert.notDeepEqual(cropped.data, source.data);
});

test("clean observations preserve the decoded RGB image", async () => {
  const source = await decodeSourceImage(sourceImagePath);
  const clean = await applyCorruption(source, {
    condition_family: "clean",
    corruption_parameters: {},
    corruption_seed: 23,
  });

  assert.notEqual(clean.data, source.data);
  assert.deepEqual(clean.data, source.data);
  assert.deepEqual(clean.transform_details, { operation: "clean" });
});
