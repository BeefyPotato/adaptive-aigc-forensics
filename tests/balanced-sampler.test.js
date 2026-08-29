import assert from "node:assert/strict";
import test from "node:test";

import { sampleBalancedTrainingObservations } from "../src/balanced-sampler.js";
import { buildObservationRecords, selectTrack5Sources } from "../src/track5-manifest.js";

function inventoryRecord(label, index) {
  const imgId = `class-${label}-${index}`;
  const uniqueNumber = label * 100 + index;
  return {
    img_id: imgId,
    image_path: `train/${imgId}.png`,
    label,
    dataset_split: "train",
    width: 32,
    height: 24,
    exact_sha256: uniqueNumber.toString(16).padStart(64, "0"),
    perceptual_hash: uniqueNumber.toString(16).padStart(16, "0"),
    provenance: {
      source_dataset: "SID_Set",
      source_reference: imgId,
      license: "CC-BY-4.0",
    },
  };
}

function trainingManifest() {
  const inventory = [];
  for (const label of [0, 1]) {
    for (let index = 1; index <= 3; index += 1) inventory.push(inventoryRecord(label, index));
  }
  const sources = selectTrack5Sources(inventory, {
    datasetRevision: "fixture-revision",
    splitSeed: 7,
    splitPlan: [{ split: "expert-training", datasetSplit: "train", perClass: 3 }],
  });
  const observations = buildObservationRecords(sources, {
    artifactSchemaVersion: "artifact-v1",
    corruptionSeed: 11,
    preprocessingVersion: "preprocessing-v1",
    transformImplementationVersion: "track5-corruption-v1+sharp-0.35.4",
  });
  return { sources, observations };
}

function countsBy(values, key) {
  const counts = {};
  for (const value of values) counts[key(value)] = (counts[key(value)] ?? 0) + 1;
  return counts;
}

test("training sampler balances class, source, family, and severity deterministically", () => {
  const manifest = trainingManifest();
  const options = { split: "expert-training", count: 504, seed: 29 };
  const sampled = sampleBalancedTrainingObservations(manifest, options);
  const repeated = sampleBalancedTrainingObservations(manifest, options);
  const differentSeed = sampleBalancedTrainingObservations(manifest, { ...options, seed: 30 });

  assert.deepEqual(sampled, repeated);
  assert.notDeepEqual(sampled, differentSeed);
  assert.deepEqual(countsBy(sampled, ({ authenticity_label: label }) => label), {
    0: 252,
    1: 252,
  });
  assert.deepEqual(countsBy(sampled, ({ condition_family: family }) => family), {
    blur: 72,
    clean: 72,
    color: 72,
    crop: 72,
    jpeg: 72,
    noise: 72,
    resize: 72,
  });

  for (const label of [0, 1]) {
    const classSamples = sampled.filter(({ authenticity_label: actual }) => actual === label);
    assert.deepEqual(Object.values(countsBy(classSamples, ({ source_id: sourceId }) => sourceId)), [
      84,
      84,
      84,
    ]);
    for (const family of ["clean", "jpeg", "blur", "resize", "noise", "color", "crop"]) {
      const severityCounts = Object.values(
        countsBy(
          classSamples.filter(({ condition_family: actual }) => actual === family),
          ({ severity }) => severity,
        ),
      );
      assert.equal(new Set(severityCounts).size, 1);
    }
  }
});

test("training sampler rejects validation and organizer demonstration inputs", () => {
  const manifest = trainingManifest();
  assert.throws(
    () =>
      sampleBalancedTrainingObservations(manifest, {
        split: "sealed-internal-test",
        count: 504,
        seed: 29,
      }),
    /training split/i,
  );

  const organizerSources = manifest.sources.map((source, index) =>
    index === 0
      ? { ...source, dataset: "organizer-demonstration", usage: "evaluation-only" }
      : source,
  );
  assert.throws(
    () =>
      sampleBalancedTrainingObservations(
        { sources: organizerSources, observations: manifest.observations },
        { split: "expert-training", count: 504, seed: 29 },
      ),
    /organizer demonstration/i,
  );

  const mislabeledObservations = manifest.observations.map((observation, index) =>
    index === 0
      ? { ...observation, authenticity_label: 1 - observation.authenticity_label }
      : observation,
  );
  assert.throws(
    () =>
      sampleBalancedTrainingObservations(
        { sources: manifest.sources, observations: mislabeledObservations },
        { split: "expert-training", count: 504, seed: 29 },
      ),
    /authenticity_label.*source/i,
  );
});
