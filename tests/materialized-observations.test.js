import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import { materializeTrack5Observations } from "../src/materialized-observations.js";

test("materialized observations preserve identity and contain actual deterministic corruptions", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "track5-materialized-"));
  try {
    const manifest = {
      manifest_schema_version: "track5-manifest-v1",
      corruption: {
        preprocessing_version: "shared-preprocessing-v1",
        transform_implementation_version: "track5-corruption-test-v1",
      },
      sources: [
        {
          source_id: "source-gradient",
          image_path: "images/gradient.svg",
          authenticity_label: 0,
          split: "internal-validation",
        },
      ],
      observations: [
        {
          source_id: "source-gradient",
          variant_id: "variant-clean",
          image_path: "images/gradient.svg",
          authenticity_label: 0,
          split: "internal-validation",
          condition_family: "clean",
          severity: "clean",
          corruption_parameters: {},
          corruption_seed: 23,
          transform_implementation_version: "track5-corruption-test-v1",
        },
        {
          source_id: "source-gradient",
          variant_id: "variant-noise",
          image_path: "images/gradient.svg",
          authenticity_label: 0,
          split: "internal-validation",
          condition_family: "noise",
          severity: "0.10",
          corruption_parameters: { sigma: 0.1, color_space: "rgb-0-1" },
          corruption_seed: 23,
          transform_implementation_version: "track5-corruption-test-v1",
        },
      ],
    };

    const resolved = await materializeTrack5Observations(manifest, {
      datasetRoot: resolve("fixtures/track5"),
      outputDirectory,
    });

    assert.equal(resolved.materialization_schema_version, "track5-materialized-observations-v1");
    assert.deepEqual(
      resolved.observations.map(({ variant_id: variantId }) => variantId),
      ["variant-clean", "variant-noise"],
    );
    const clean = await readFile(join(outputDirectory, resolved.observations[0].materialized_image_path));
    const noisy = await readFile(join(outputDirectory, resolved.observations[1].materialized_image_path));
    assert.notDeepEqual(clean, noisy);

    const repeatedDirectory = join(outputDirectory, "repeated");
    const repeated = await materializeTrack5Observations(manifest, {
      datasetRoot: resolve("fixtures/track5"),
      outputDirectory: repeatedDirectory,
    });
    const repeatedNoise = await readFile(
      join(repeatedDirectory, repeated.observations[1].materialized_image_path),
    );
    assert.deepEqual(noisy, repeatedNoise);
  } finally {
    await rm(outputDirectory, { recursive: true, force: true });
  }
});

test("materialization rejects source paths outside the declared dataset root", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "track5-materialized-escape-"));
  try {
    const manifest = {
      manifest_schema_version: "track5-manifest-v1",
      corruption: {
        preprocessing_version: "shared-preprocessing-v1",
        transform_implementation_version: "track5-corruption-test-v1",
      },
      sources: [
        {
          source_id: "escape",
          image_path: "../escape.png",
          authenticity_label: 0,
          split: "expert-training",
        },
      ],
      observations: [
        {
          source_id: "escape",
          variant_id: "escape-clean",
          image_path: "../escape.png",
          authenticity_label: 0,
          split: "expert-training",
          condition_family: "clean",
          severity: "clean",
          corruption_parameters: {},
          corruption_seed: 23,
          transform_implementation_version: "track5-corruption-test-v1",
        },
      ],
    };
    await assert.rejects(
      materializeTrack5Observations(manifest, {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory,
      }),
      /escapes the dataset root/u,
    );
  } finally {
    await rm(outputDirectory, { recursive: true, force: true });
  }
});
