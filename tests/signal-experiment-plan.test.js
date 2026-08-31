import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { access, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

import {
  buildSignalExperimentPlan,
  validateSignalExperimentShard,
} from "../src/signal-experiment-plan.js";
import { buildObservationRecords } from "../src/track5-manifest.js";

const PARENT_MANIFEST_SHA256 = "a".repeat(64);
const MATERIALIZABLE_SOURCE_RAW_BYTES = 16 * 12 * 3 * 20;

function recipeManifest({ validationSourcesPerClass = 1 } = {}) {
  const sources = [
    "expert-training",
    "fusion-training",
    "internal-validation",
    "sealed-internal-test",
  ].flatMap((split) =>
    [0, 1].flatMap((authenticityLabel) =>
      Array.from(
        { length: split === "internal-validation" ? validationSourcesPerClass : 1 },
        (_, index) => {
          const suffix = validationSourcesPerClass === 1 || split !== "internal-validation"
            ? ""
            : `-${index}`;
          return {
            source_id: `${split}-class-${authenticityLabel}${suffix}`,
            image_path: `images/${split}-class-${authenticityLabel}${suffix}.png`,
            authenticity_label: authenticityLabel,
            split,
            width: 8,
            height: 6,
          };
        },
      ),
    ),
  );
  const corruption = {
    root_seed: 23,
    preprocessing_version: "shared-preprocessing-test-v1",
    artifact_schema_version: "artifact-test-v1",
    transform_implementation_version: "track5-corruption-test-v1",
    condition_count_per_source: 20,
  };
  return {
    manifest_schema_version: "track5-manifest-v1",
    source_contract_version: "track5-source-v1",
    observation_contract_version: "track5-observation-v1",
    condition_matrix_version: "track5-condition-matrix-v1",
    sampler_contract_version: "track5-balanced-sampler-v1",
    corruption,
    organizer_demonstration_policy: { usage: "evaluation-only" },
    leakage_audit: { status: "passed" },
    sources,
    observations: buildObservationRecords(sources, {
      artifactSchemaVersion: corruption.artifact_schema_version,
      corruptionSeed: corruption.root_seed,
      preprocessingVersion: corruption.preprocessing_version,
      transformImplementationVersion: corruption.transform_implementation_version,
    }),
  };
}

function materializableRecipeManifest() {
  const manifest = recipeManifest();
  const imagePathBySource = new Map(
    manifest.sources.map((source) => [
      source.source_id,
      source.authenticity_label === 0 ? "images/checker.svg" : "images/gradient.svg",
    ]),
  );
  return {
    ...manifest,
    sources: manifest.sources.map((source) => ({
      ...source,
      image_path: imagePathBySource.get(source.source_id),
      width: 16,
      height: 12,
    })),
    observations: manifest.observations.map((observation) => ({
      ...observation,
      image_path: imagePathBySource.get(observation.source_id),
      width: 16,
      height: 12,
    })),
  };
}

test("signal plan deterministically selects only balanced expert training and complete validation", () => {
  const manifest = recipeManifest();
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  };

  const plan = buildSignalExperimentPlan(manifest, options);
  const repeated = buildSignalExperimentPlan(
    {
      ...manifest,
      sources: manifest.sources.toReversed(),
      observations: manifest.observations.toReversed(),
    },
    options,
  );

  assert.deepEqual(plan, repeated);
  assert.equal(plan.plan_schema_version, "signal-experiment-plan-v1");
  assert.equal(plan.experiment_profile, "custom-v1");
  assert.equal(plan.acceptance_scope, "non-acceptance");
  assert.equal(plan.validation_source_count, 2);
  assert.equal(plan.validation_seed, 61);
  assert.equal(plan.parent_recipe_manifest_sha256, PARENT_MANIFEST_SHA256);
  assert.match(plan.plan_sha256, /^[0-9a-f]{64}$/u);
  assert.deepEqual(
    plan.phases.map(({ phase }) => phase),
    ["expert-training", "internal-validation"],
  );

  const training = plan.phases[0];
  const validation = plan.phases[1];
  assert.equal(
    training.shards.flatMap(({ records }) => records).reduce(
      (total, { sample_weight: sampleWeight }) => total + sampleWeight,
      0,
    ),
    168,
  );
  assert.equal(validation.shards.flatMap(({ records }) => records).length, 40);
  assert.ok(
    validation.shards
      .flatMap(({ records }) => records)
      .every(({ sample_weight: sampleWeight }) => sampleWeight === 1),
  );
  assert.equal(training.shards.length, 2);
  assert.equal(validation.shards.length, 2);

  for (const phase of plan.phases) {
    for (const [index, shard] of phase.shards.entries()) {
      assert.equal(shard.parent_recipe_manifest_sha256, PARENT_MANIFEST_SHA256);
      assert.equal(shard.plan_sha256, plan.plan_sha256);
      assert.equal(shard.phase, phase.phase);
      assert.equal(shard.index, index);
      assert.equal(shard.count, phase.shards.length);
      assert.equal(shard.raw_byte_estimate, options.rawByteBudget);
      assert.equal(new Set(shard.records.map(({ source_id: sourceId }) => sourceId)).size, 1);
      assert.match(shard.variant_set_digest, /^[0-9a-f]{64}$/u);
      assert.ok(shard.records.every(({ split }) => split === phase.phase));
    }
  }
});

test("signal hackathon planning selects a deterministic class-balanced whole-source validation subset", () => {
  const manifest = recipeManifest({ validationSourcesPerClass: 8 });
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20 * 4,
    trainingCount: 168,
    trainingSeed: 61,
    validationSourceCount: 8,
    validationSeed: 73,
    experimentProfile: "custom-v1",
  };

  const plan = buildSignalExperimentPlan(manifest, options);
  const repeated = buildSignalExperimentPlan(
    {
      ...manifest,
      sources: manifest.sources.toReversed(),
      observations: manifest.observations.toReversed(),
    },
    options,
  );
  const validation = plan.phases
    .find(({ phase }) => phase === "internal-validation")
    .shards.flatMap(({ records }) => records);
  const selectedSources = new Map();
  for (const record of validation) {
    selectedSources.set(record.source_id, record.authenticity_label);
  }

  assert.deepEqual(plan, repeated);
  assert.equal(plan.validation_source_count, 8);
  assert.equal(validation.length, 160);
  assert.deepEqual(
    Object.fromEntries(
      [0, 1].map((label) => [
        label,
        [...selectedSources.values()].filter((candidate) => candidate === label).length,
      ]),
    ),
    { 0: 4, 1: 4 },
  );
  for (const sourceId of selectedSources.keys()) {
    assert.equal(validation.filter(({ source_id: candidate }) => candidate === sourceId).length, 20);
  }

  const differentSeed = buildSignalExperimentPlan(manifest, {
    ...options,
    validationSeed: 74,
  });
  assert.notEqual(differentSeed.plan_sha256, plan.plan_sha256);
});

test("named signal profiles reject counts that could mislabel time-boxed or full acceptance artifacts", () => {
  const manifest = recipeManifest();
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  };

  assert.throws(
    () => buildSignalExperimentPlan(manifest, {
      ...options,
      experimentProfile: "hackathon-v1",
      validationSourceCount: 400,
    }),
    /hackathon-v1.*8,064|8,064.*hackathon-v1/i,
  );
  assert.throws(
    () => buildSignalExperimentPlan(manifest, {
      ...options,
      experimentProfile: "issue-6-full-v1",
    }),
    /issue-6-full-v1.*40,320|40,320.*issue-6-full-v1/i,
  );
  assert.throws(
    () => buildSignalExperimentPlan(manifest, {
      ...options,
      experimentProfile: "unknown-profile",
    }),
    /experimentProfile.*supported/i,
  );
});

test("plan CLI hashes the recipe once and writes standalone bounded shard plans", async () => {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "signal-plan-cli-"));
  try {
    const manifestPath = join(temporaryDirectory, "track5-manifest.json");
    const outputPath = join(temporaryDirectory, "signal-plan.json");
    const manifestText = `${JSON.stringify(recipeManifest(), null, 2)}\n`;
    await writeFile(manifestPath, manifestText);

    const result = spawnSync(
      process.execPath,
      [
        "scripts/materialize-signal-shard.mjs",
        "plan",
        "--manifest",
        manifestPath,
        "--training-count",
        "168",
        "--training-seed",
        "61",
        "--raw-byte-budget",
        String(8 * 6 * 3 * 20),
        "--output",
        outputPath,
      ],
      { cwd: process.cwd(), encoding: "utf8" },
    );

    assert.equal(result.status, 0, result.stderr);
    const summary = JSON.parse(result.stdout);
    const plan = JSON.parse(await readFile(outputPath, "utf8"));
    assert.equal(
      plan.parent_recipe_manifest_sha256,
      createHash("sha256").update(manifestText).digest("hex"),
    );
    assert.equal(summary.plan_sha256, plan.plan_sha256);
    assert.deepEqual(summary.phase_shard_counts, {
      "expert-training": 2,
      "internal-validation": 2,
    });
    const firstShardPath = join(
      summary.shard_directory,
      "expert-training-00000-of-00002.json",
    );
    const firstShard = JSON.parse(await readFile(firstShardPath, "utf8"));
    assert.equal(firstShard.plan_sha256, plan.plan_sha256);
    assert.equal(firstShard.phase, "expert-training");
    assert.equal(firstShard.index, 0);
    assert.ok(firstShard.records.length > 0);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
});

test("materialize CLI resolves one standalone shard through the canonical materializer", async () => {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "signal-materialize-cli-"));
  try {
    const manifestPath = join(temporaryDirectory, "track5-manifest.json");
    const planPath = join(temporaryDirectory, "signal-plan.json");
    await writeFile(manifestPath, `${JSON.stringify(materializableRecipeManifest(), null, 2)}\n`);
    const planning = spawnSync(
      process.execPath,
      [
        "scripts/materialize-signal-shard.mjs",
        "plan",
        "--manifest",
        manifestPath,
        "--training-count",
        "168",
        "--training-seed",
        "61",
        "--raw-byte-budget",
        String(MATERIALIZABLE_SOURCE_RAW_BYTES),
        "--output",
        planPath,
      ],
      { cwd: process.cwd(), encoding: "utf8" },
    );
    assert.equal(planning.status, 0, planning.stderr);
    const planSummary = JSON.parse(planning.stdout);
    const shardPath = join(
      planSummary.shard_directory,
      "expert-training-00000-of-00002.json",
    );
    const shard = JSON.parse(await readFile(shardPath, "utf8"));
    const outputDirectory = join(temporaryDirectory, "materialized");

    const materialization = spawnSync(
      process.execPath,
      [
        "scripts/materialize-signal-shard.mjs",
        "materialize",
        "--shard-plan",
        shardPath,
        "--expected-plan-sha256",
        planSummary.plan_sha256,
        "--expected-shard-sha256",
        shard.shard_sha256,
        "--phase",
        "expert-training",
        "--index",
        "0",
        "--dataset-root",
        "fixtures/track5",
        "--output-dir",
        outputDirectory,
        "--concurrency",
        "1",
      ],
      { cwd: process.cwd(), encoding: "utf8" },
    );

    assert.equal(materialization.status, 0, materialization.stderr);
    const summary = JSON.parse(materialization.stdout);
    const resolved = JSON.parse(await readFile(summary.materialized_manifest_path, "utf8"));
    assert.equal(resolved.materialization_schema_version, "track5-materialized-observations-v1");
    assert.deepEqual(resolved.signal_shard_provenance, {
      parent_recipe_manifest_sha256: resolved.parent_recipe_manifest_sha256,
      plan_sha256: planSummary.plan_sha256,
      shard_sha256: shard.shard_sha256,
      phase: "expert-training",
      index: 0,
      count: 2,
      variant_set_digest: resolved.signal_shard_provenance.variant_set_digest,
      raw_byte_budget: MATERIALIZABLE_SOURCE_RAW_BYTES,
      raw_byte_estimate: MATERIALIZABLE_SOURCE_RAW_BYTES,
    });
    assert.ok(resolved.observations.every(({ sample_weight: weight }) => weight > 0));
    assert.ok(
      resolved.observations.every(({ materialized_image_path: imagePath }) =>
        /^observations\/[0-9a-f]{64}\.png$/u.test(imagePath),
      ),
    );
    assert.equal(summary.observation_count, resolved.observations.length);
    await access(join(outputDirectory, resolved.observations[0].materialized_image_path));
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
});

test("signal planning fails closed on invalid options and recipe relationships", () => {
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  };
  const build = (manifest, overrides = {}) =>
    buildSignalExperimentPlan(manifest, { ...options, ...overrides });

  assert.throws(() => build(recipeManifest(), { rawByteBudget: 0 }), /positive safe integer/i);
  assert.throws(() => build(recipeManifest(), { rawByteBudget: 1 }), /exceeding budget/i);
  assert.throws(() => build(recipeManifest(), { trainingCount: 1 }), /divisible/i);

  const invalidSplit = structuredClone(recipeManifest());
  invalidSplit.sources[0].split = "organizer-demonstration";
  assert.throws(() => build(invalidSplit), /invalid split/i);

  const staleSampler = structuredClone(recipeManifest());
  staleSampler.sampler_contract_version = "track5-balanced-sampler-stale";
  assert.throws(() => build(staleSampler), /sampler contract version/i);

  const duplicateSource = structuredClone(recipeManifest());
  duplicateSource.sources.push({ ...duplicateSource.sources[0] });
  assert.throws(() => build(duplicateSource), /repeats source/i);

  const duplicateVariant = structuredClone(recipeManifest());
  duplicateVariant.observations.push({ ...duplicateVariant.observations[0] });
  assert.throws(() => build(duplicateVariant), /repeats variant/i);

  const missingVariant = structuredClone(recipeManifest());
  missingVariant.observations.pop();
  assert.throws(() => build(missingVariant), /is missing/i);

  const wrongSource = structuredClone(recipeManifest());
  wrongSource.observations[0].source_id = wrongSource.sources[1].source_id;
  assert.throws(() => build(wrongSource), /disagrees with source/i);
});

test("signal planning rejects organizer demonstration metadata at every record level", () => {
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  };
  const mutations = [
    {
      name: "source dataset",
      mutate(manifest) {
        manifest.sources.find(({ split }) => split === "internal-validation").dataset =
          "organizer-demonstration";
      },
    },
    {
      name: "source usage",
      mutate(manifest) {
        manifest.sources.find(({ split }) => split === "internal-validation").usage =
          "evaluation-only";
      },
    },
    {
      name: "observation dataset",
      mutate(manifest) {
        manifest.observations.find(({ split }) => split === "internal-validation").dataset =
          "organizer-demonstration";
      },
    },
    {
      name: "observation usage",
      mutate(manifest) {
        manifest.observations.find(({ split }) => split === "internal-validation").usage =
          "evaluation-only";
      },
    },
  ];

  for (const { name, mutate } of mutations) {
    const manifest = structuredClone(recipeManifest());
    mutate(manifest);
    assert.throws(
      () => buildSignalExperimentPlan(manifest, options),
      /organizer demonstration.*prohibited|evaluation-only.*prohibited/i,
      name,
    );
  }
});

test("signal plans preserve available source pins and reject incomplete or disagreeing records", () => {
  const options = {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  };
  const pinnedSourceId = "internal-validation-class-0";

  const incomplete = structuredClone(recipeManifest());
  incomplete.sources.find(({ source_id: sourceId }) => sourceId === pinnedSourceId).byte_length =
    123;
  assert.throws(
    () => buildSignalExperimentPlan(incomplete, options),
    /byte_length requires exact_sha256/i,
  );

  const disagreeing = structuredClone(recipeManifest());
  const disagreeingSource = disagreeing.sources.find(
    ({ source_id: sourceId }) => sourceId === pinnedSourceId,
  );
  disagreeingSource.byte_length = 123;
  disagreeingSource.exact_sha256 = "c".repeat(64);
  const disagreeingObservation = disagreeing.observations.find(
    ({ source_id: sourceId }) => sourceId === pinnedSourceId,
  );
  disagreeingObservation.byte_length = 124;
  disagreeingObservation.exact_sha256 = "c".repeat(64);
  assert.throws(
    () => buildSignalExperimentPlan(disagreeing, options),
    /pinned source bytes.*disagree/i,
  );

  const valid = structuredClone(recipeManifest());
  const validSource = valid.sources.find(
    ({ source_id: sourceId }) => sourceId === pinnedSourceId,
  );
  validSource.byte_length = 123;
  validSource.exact_sha256 = "c".repeat(64);
  const plan = buildSignalExperimentPlan(valid, options);
  const plannedSource = plan.phases
    .flatMap(({ shards }) => shards)
    .flatMap(({ sources }) => sources)
    .find(({ source_id: sourceId }) => sourceId === pinnedSourceId);
  assert.equal(plannedSource.byte_length, 123);
  assert.equal(plannedSource.exact_sha256, "c".repeat(64));

  const hashOnly = structuredClone(recipeManifest());
  const hashOnlySource = hashOnly.sources.find(
    ({ source_id: sourceId }) => sourceId === pinnedSourceId,
  );
  hashOnlySource.exact_sha256 = "d".repeat(64);
  const hashOnlyPlan = buildSignalExperimentPlan(hashOnly, options);
  const plannedHashOnlySource = hashOnlyPlan.phases
    .flatMap(({ shards }) => shards)
    .flatMap(({ sources }) => sources)
    .find(({ source_id: sourceId }) => sourceId === pinnedSourceId);
  assert.equal(plannedHashOnlySource.exact_sha256, "d".repeat(64));
  assert.equal(Object.hasOwn(plannedHashOnlySource, "byte_length"), false);
});

test("standalone shard validation rejects stale identity, bounds, and record membership", () => {
  const plan = buildSignalExperimentPlan(recipeManifest(), {
    parentRecipeManifestSha256: PARENT_MANIFEST_SHA256,
    rawByteBudget: 8 * 6 * 3 * 20,
    trainingCount: 168,
    trainingSeed: 61,
  });
  const shard = plan.phases[0].shards[0];
  const expected = {
    expectedIndex: shard.index,
    expectedPhase: shard.phase,
    expectedPlanSha256: plan.plan_sha256,
    expectedShardSha256: shard.shard_sha256,
  };
  assert.doesNotThrow(() => validateSignalExperimentShard(shard, expected));
  assert.throws(
    () => validateSignalExperimentShard(shard, { ...expected, expectedIndex: 1 }),
    /does not match requested/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(shard, {
        ...expected,
        expectedPhase: "internal-validation",
      }),
    /does not match requested/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(shard, {
        ...expected,
        expectedShardSha256: "b".repeat(64),
      }),
    /SHA-256 does not match/i,
  );
  assert.throws(
    () => validateSignalExperimentShard({ ...shard, count: 0 }, expected),
    /positive safe integer/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(
        { ...shard, records: shard.records.slice(1) },
        expected,
      ),
    /raw-byte estimate|variant-set digest|content does not match/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(
        {
          ...shard,
          sources: shard.sources.map((source, index) =>
            index === 0 ? { ...source, dataset: "organizer-demonstration" } : source,
          ),
        },
        expected,
      ),
    /organizer demonstration.*prohibited/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(
        {
          ...shard,
          records: shard.records.map((record, index) =>
            index === 0 ? { ...record, usage: "evaluation-only" } : record,
          ),
        },
        expected,
    ),
    /evaluation-only.*prohibited/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(
        {
          ...shard,
          sources: shard.sources.map((source, index) =>
            index === 0 ? { ...source, byte_length: 123 } : source,
          ),
        },
        expected,
      ),
    /byte_length requires exact_sha256/i,
  );
  assert.throws(
    () =>
      validateSignalExperimentShard(
        {
          ...shard,
          sources: shard.sources.map((source, index) =>
            index === 0
              ? { ...source, byte_length: 123, exact_sha256: "c".repeat(64) }
              : source,
          ),
          records: shard.records.map((record, index) =>
            index === 0
              ? { ...record, byte_length: 124, exact_sha256: "c".repeat(64) }
              : record,
          ),
        },
        expected,
      ),
    /pinned source bytes disagree/i,
  );
});

test("materialize CLI rejects a recipe path that escapes the dataset root", async () => {
  const temporaryDirectory = await mkdtemp(join(tmpdir(), "signal-materialize-escape-"));
  try {
    const manifest = materializableRecipeManifest();
    manifest.sources[0].image_path = "../outside.png";
    for (const observation of manifest.observations) {
      if (observation.source_id === manifest.sources[0].source_id) {
        observation.image_path = "../outside.png";
      }
    }
    const manifestPath = join(temporaryDirectory, "track5-manifest.json");
    const planPath = join(temporaryDirectory, "signal-plan.json");
    await writeFile(manifestPath, `${JSON.stringify(manifest)}\n`);
    const planning = spawnSync(
      process.execPath,
      [
        "scripts/materialize-signal-shard.mjs",
        "plan",
        "--manifest",
        manifestPath,
        "--training-count",
        "168",
        "--training-seed",
        "61",
        "--raw-byte-budget",
        String(MATERIALIZABLE_SOURCE_RAW_BYTES),
        "--output",
        planPath,
      ],
      { cwd: process.cwd(), encoding: "utf8" },
    );
    assert.equal(planning.status, 0, planning.stderr);
    const planSummary = JSON.parse(planning.stdout);
    const shardPath = join(
      planSummary.shard_directory,
      "expert-training-00000-of-00002.json",
    );
    const shard = JSON.parse(await readFile(shardPath, "utf8"));
    const materialization = spawnSync(
      process.execPath,
      [
        "scripts/materialize-signal-shard.mjs",
        "materialize",
        "--shard-plan",
        shardPath,
        "--expected-plan-sha256",
        planSummary.plan_sha256,
        "--expected-shard-sha256",
        shard.shard_sha256,
        "--phase",
        shard.phase,
        "--index",
        String(shard.index),
        "--dataset-root",
        "fixtures/track5",
        "--output-dir",
        join(temporaryDirectory, "materialized"),
      ],
      { cwd: process.cwd(), encoding: "utf8" },
    );

    assert.notEqual(materialization.status, 0);
    assert.match(materialization.stderr, /escapes the dataset root/i);
  } finally {
    await rm(temporaryDirectory, { recursive: true, force: true });
  }
});
