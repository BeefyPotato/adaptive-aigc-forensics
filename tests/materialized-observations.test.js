import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { link, mkdir, mkdtemp, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";

import sharp from "sharp";

import { materializeTrack5Observations } from "../src/materialized-observations.js";

function cleanManifest(sourceOverrides = {}, observationOverrides = {}, corruptionOverrides = {}) {
  const source = {
    source_id: "source-gradient",
    image_path: "images/gradient.svg",
    authenticity_label: 0,
    split: "internal-validation",
    width: 16,
    height: 12,
    ...sourceOverrides,
  };
  return {
    manifest_schema_version: "track5-manifest-v1",
    corruption: {
      preprocessing_version: "shared-preprocessing-v1",
      transform_implementation_version: "track5-corruption-test-v1",
      sharp_version: sharp.versions.sharp,
      libvips_version: sharp.versions.vips,
      ...corruptionOverrides,
    },
    sources: [source],
    observations: [
      {
        source_id: source.source_id,
        variant_id: "variant-clean",
        image_path: source.image_path,
        authenticity_label: source.authenticity_label,
        split: source.split,
        width: source.width,
        height: source.height,
        condition_family: "clean",
        severity: "clean",
        corruption_parameters: {},
        corruption_seed: 23,
        transform_implementation_version: "track5-corruption-test-v1",
        ...observationOverrides,
      },
    ],
  };
}

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
    assert.equal(resolved.materialization.sharp_version, sharp.versions.sharp);
    assert.equal(resolved.materialization.libvips_version, sharp.versions.vips);
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

test("materialization rejects paths whose resolved target escapes through a directory junction", async (t) => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-materialized-junction-"));
  try {
    const datasetRoot = join(temporaryRoot, "dataset");
    const outsideRoot = join(temporaryRoot, "outside");
    const outputDirectory = join(temporaryRoot, "output");
    await mkdir(datasetRoot, { recursive: true });
    await mkdir(outsideRoot, { recursive: true });
    await writeFile(join(outsideRoot, "gradient.svg"), "<svg xmlns=\"http://www.w3.org/2000/svg\"/>");
    try {
      await symlink(outsideRoot, join(datasetRoot, "linked-outside"), "junction");
    } catch (error) {
      if (["EACCES", "ENOTSUP", "EPERM"].includes(error.code)) {
        t.skip(`directory junctions are unavailable in this environment: ${error.code}`);
        return;
      }
      throw error;
    }

    const imagePath = "linked-outside/gradient.svg";
    await assert.rejects(
      materializeTrack5Observations(
        cleanManifest(
          { image_path: imagePath },
          { image_path: imagePath },
        ),
        { datasetRoot, outputDirectory },
      ),
      /escapes the dataset root/u,
    );
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("decode failures for pinned source bytes never disclose the source payload", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-materialized-decode-"));
  try {
    const datasetRoot = join(temporaryRoot, "dataset");
    const outputDirectory = join(temporaryRoot, "output");
    const imageDirectory = join(datasetRoot, "images");
    const imagePath = "images/corrupt.bin";
    const secretPayload = "ISSUE6-PRIVATE-PIXEL-BYTES-MUST-NOT-APPEAR-IN-ERRORS";
    const sourceBytes = Buffer.from(secretPayload, "utf8");
    await mkdir(imageDirectory, { recursive: true });
    await writeFile(join(datasetRoot, imagePath), sourceBytes);

    const exactSha256 = createHash("sha256").update(sourceBytes).digest("hex");
    await assert.rejects(
      materializeTrack5Observations(
        cleanManifest(
          {
            source_id: "source-corrupt",
            image_path: imagePath,
            byte_length: sourceBytes.length,
            exact_sha256: exactSha256,
          },
          { source_id: "source-corrupt", image_path: imagePath },
        ),
        { datasetRoot, outputDirectory },
      ),
      (error) => {
        assert.match(error.message, /source-corrupt/u);
        assert.match(error.message, /verified source bytes/u);
        assert.doesNotMatch(error.message, new RegExp(secretPayload, "u"));
        return true;
      },
    );
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("atomic materialization never reuses a preplanted predictable temp file", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-materialized-atomic-"));
  try {
    const outputDirectory = join(temporaryRoot, "output");
    const imageDirectory = join(outputDirectory, "observations");
    await mkdir(imageDirectory, { recursive: true });
    const sentinel = join(temporaryRoot, "outside-sentinel.txt");
    const sentinelBytes = Buffer.from("must remain unchanged", "utf8");
    await writeFile(sentinel, sentinelBytes);
    const filename = `${createHash("sha256").update("variant-clean").digest("hex")}.png`;
    const legacyTemporary = join(imageDirectory, `${filename}.tmp-${process.pid}`);
    await link(sentinel, legacyTemporary);

    await materializeTrack5Observations(cleanManifest(), {
      datasetRoot: resolve("fixtures/track5"),
      outputDirectory,
    });

    assert.deepEqual(await readFile(sentinel), sentinelBytes);
    assert.deepEqual(await readFile(legacyTemporary), sentinelBytes);
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("materialization rejects a redirected managed observations directory", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "track5-materialized-output-link-"));
  try {
    const outputDirectory = join(temporaryRoot, "output");
    const outsideDirectory = join(temporaryRoot, "outside");
    await mkdir(outputDirectory, { recursive: true });
    await mkdir(outsideDirectory, { recursive: true });
    const sentinel = join(outsideDirectory, "must-survive.txt");
    await writeFile(sentinel, "sentinel");
    await symlink(outsideDirectory, join(outputDirectory, "observations"), "junction");

    await assert.rejects(
      materializeTrack5Observations(cleanManifest(), {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory,
      }),
      /managed output.*redirected|observations.*symlink|observations.*junction/iu,
    );
    assert.equal(await readFile(sentinel, "utf8"), "sentinel");
  } finally {
    await rm(temporaryRoot, { recursive: true, force: true });
  }
});

test("materialization verifies every available source-byte pin before decoding", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "track5-materialized-pinned-"));
  try {
    const sourceBytes = await readFile(resolve("fixtures/track5/images/gradient.svg"));
    const exactSha256 = createHash("sha256").update(sourceBytes).digest("hex");
    const pinned = {
      byte_length: sourceBytes.length,
      exact_sha256: exactSha256,
    };

    const resolved = await materializeTrack5Observations(cleanManifest(pinned), {
      datasetRoot: resolve("fixtures/track5"),
      outputDirectory: join(outputDirectory, "valid"),
    });
    assert.equal(resolved.observations.length, 1);

    const hashOnlyResolved = await materializeTrack5Observations(
      cleanManifest({ exact_sha256: exactSha256 }),
      {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory: join(outputDirectory, "valid-hash-only"),
      },
    );
    assert.equal(hashOnlyResolved.observations.length, 1);
    const hashOnlyObservationResolved = await materializeTrack5Observations(
      cleanManifest({ exact_sha256: exactSha256 }, { exact_sha256: exactSha256 }),
      {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory: join(outputDirectory, "valid-hash-only-observation"),
      },
    );
    assert.equal(hashOnlyObservationResolved.observations[0].exact_sha256, exactSha256);

    const invalidSources = [
      [{ byte_length: sourceBytes.length }, /byte_length requires exact_sha256/i],
      [{ exact_sha256: "not-a-sha" }, /lowercase 64-digit hex/i],
      [{ ...pinned, byte_length: sourceBytes.length + 1 }, /pinned byte length/i],
      [{ ...pinned, exact_sha256: "0".repeat(64) }, /pinned SHA-256/i],
    ];
    for (const [index, [sourceOverrides, expectedError]] of invalidSources.entries()) {
      await assert.rejects(
        materializeTrack5Observations(cleanManifest(sourceOverrides), {
          datasetRoot: resolve("fixtures/track5"),
          outputDirectory: join(outputDirectory, `invalid-${index}`),
        }),
        expectedError,
      );
    }

    const resolvedWithObservationPins = await materializeTrack5Observations(
      cleanManifest(pinned, pinned),
      {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory: join(outputDirectory, "valid-observation-pins"),
      },
    );
    assert.equal(resolvedWithObservationPins.observations[0].byte_length, sourceBytes.length);
    assert.equal(resolvedWithObservationPins.observations[0].exact_sha256, exactSha256);

    await assert.rejects(
      materializeTrack5Observations(
        cleanManifest(pinned, { byte_length: sourceBytes.length }),
        {
          datasetRoot: resolve("fixtures/track5"),
          outputDirectory: join(outputDirectory, "incomplete-observation-pins"),
        },
      ),
      /byte_length requires exact_sha256/i,
    );
    await assert.rejects(
      materializeTrack5Observations(
        cleanManifest(pinned, {
          byte_length: sourceBytes.length + 1,
          exact_sha256: exactSha256,
        }),
        {
          datasetRoot: resolve("fixtures/track5"),
          outputDirectory: join(outputDirectory, "disagreeing-observation-pins"),
        },
      ),
      /incompatible pinned source bytes/i,
    );
  } finally {
    await rm(outputDirectory, { recursive: true, force: true });
  }
});

test("materialization rejects stale declared Sharp runtimes before reading source pixels", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "track5-materialized-runtime-"));
  try {
    const invalidSource = {
      image_path: "images/does-not-exist.png",
    };
    const staleDeclarations = [
      [{ sharp_version: "0.0.0-stale" }, /sharp_version.*incompatible/i],
      [{ libvips_version: "0.0.0-stale" }, /libvips_version.*incompatible/i],
    ];
    for (const [index, [corruptionOverrides, expectedError]] of staleDeclarations.entries()) {
      await assert.rejects(
        materializeTrack5Observations(
          cleanManifest(invalidSource, { image_path: invalidSource.image_path }, corruptionOverrides),
          {
            datasetRoot: resolve("fixtures/track5"),
            outputDirectory: join(outputDirectory, `stale-${index}`),
          },
        ),
        expectedError,
      );
    }
    const incompleteRuntime = cleanManifest();
    delete incompleteRuntime.corruption.libvips_version;
    await assert.rejects(
      materializeTrack5Observations(incompleteRuntime, {
        datasetRoot: resolve("fixtures/track5"),
        outputDirectory: join(outputDirectory, "incomplete"),
      }),
      /declare sharp_version and libvips_version together/i,
    );
  } finally {
    await rm(outputDirectory, { recursive: true, force: true });
  }
});

test("materialization rejects corrupted pixels outside declared native RGB geometry", async () => {
  const outputDirectory = await mkdtemp(join(tmpdir(), "track5-materialized-geometry-"));
  try {
    await assert.rejects(
      materializeTrack5Observations(
        cleanManifest({ width: 15 }, { width: 15 }),
        {
          datasetRoot: resolve("fixtures/track5"),
          outputDirectory,
        },
      ),
      /corruption result.*declared.*15x12.*received.*16x12/i,
    );
  } finally {
    await rm(outputDirectory, { recursive: true, force: true });
  }
});
