import assert from "node:assert/strict";
import test from "node:test";

import {
  assertLeakageAuditPassed,
  auditTrack5Sources,
  buildObservationRecords,
  buildTrack5Manifest,
  selectTrack5Sources,
} from "../src/track5-manifest.js";

function inventoryRecord({ datasetSplit, index, label }) {
  const className = label === 0 ? "authentic" : label === 1 ? "full-synthetic" : "tampered";
  const imgId = `${className}-${datasetSplit}-${String(index).padStart(5, "0")}`;
  const uniqueNumber =
    (datasetSplit === "validation" ? 30_000 : 0) + label * 10_000 + index + 1;
  return {
    img_id: imgId,
    image_path: `${datasetSplit}/${className}/${imgId}.jpg`,
    label,
    dataset_split: datasetSplit,
    width: 1024,
    height: 768,
    exact_sha256: uniqueNumber.toString(16).padStart(64, "0"),
    perceptual_hash: uniqueNumber.toString(16).padStart(16, "0"),
    provenance: {
      source_dataset: label === 0 ? "OpenImages V7" : "SID_Set",
      source_reference: imgId,
      license: "CC-BY-4.0",
    },
  };
}

function productionSizedInventory() {
  const records = [];
  for (const datasetSplit of ["train", "validation"]) {
    const perClass = datasetSplit === "train" ? 5_001 : 2_001;
    for (const label of [0, 1]) {
      for (let index = 0; index < perClass; index += 1) {
        records.push(inventoryRecord({ datasetSplit, index, label }));
      }
    }
  }
  for (let index = 0; index < 10; index += 1) {
    records.push(inventoryRecord({ datasetSplit: "train", index, label: 2 }));
  }
  return records;
}

test("Track 5 sources are class-balanced, source-disjoint, and selected before variants", () => {
  const inventory = productionSizedInventory();
  const selected = selectTrack5Sources(inventory, {
    datasetRevision: "saberzl/SID_Set@dc03ead",
    splitSeed: 17,
  });
  const repeated = selectTrack5Sources(inventory.toReversed(), {
    datasetRevision: "saberzl/SID_Set@dc03ead",
    splitSeed: 17,
  });

  assert.equal(selected.length, 14_000);
  assert.deepEqual(
    selected.map(({ source_id: sourceId }) => sourceId),
    repeated.map(({ source_id: sourceId }) => sourceId),
  );
  assert.equal(new Set(selected.map(({ source_id: sourceId }) => sourceId)).size, 14_000);
  assert.ok(selected.every(({ authenticity_label: label }) => label === 0 || label === 1));

  const expectedCounts = {
    "expert-training:0": 4_000,
    "expert-training:1": 4_000,
    "fusion-training:0": 1_000,
    "fusion-training:1": 1_000,
    "internal-validation:0": 1_000,
    "internal-validation:1": 1_000,
    "sealed-internal-test:0": 1_000,
    "sealed-internal-test:1": 1_000,
  };
  const actualCounts = Object.fromEntries(Object.keys(expectedCounts).map((key) => [key, 0]));
  for (const source of selected) {
    actualCounts[`${source.split}:${source.authenticity_label}`] += 1;
    const expectedDatasetSplit =
      source.split === "expert-training" || source.split === "fusion-training"
        ? "train"
        : "validation";
    assert.equal(source.dataset_split, expectedDatasetSplit);
    assert.equal(source.dataset_revision, "saberzl/SID_Set@dc03ead");
  }
  assert.deepEqual(actualCounts, expectedCounts);
});

test("every selected source receives the complete deterministic Track 5 condition matrix", () => {
  const selectedSources = selectTrack5Sources(
    [
      inventoryRecord({ datasetSplit: "train", index: 1, label: 0 }),
      inventoryRecord({ datasetSplit: "train", index: 1, label: 1 }),
    ],
    {
      datasetRevision: "saberzl/SID_Set@dc03ead",
      splitSeed: 17,
      splitPlan: [{ split: "expert-training", datasetSplit: "train", perClass: 1 }],
    },
  );
  const options = {
    artifactSchemaVersion: "artifact-v1",
    corruptionSeed: 23,
    preprocessingVersion: "shared-preprocessing-v1",
    transformImplementationVersion: "track5-corruption-v1+sharp-0.35.4",
  };
  const observations = buildObservationRecords(selectedSources, options);
  const repeated = buildObservationRecords(selectedSources, options);

  assert.equal(observations.length, 40);
  assert.deepEqual(observations, repeated);
  assert.equal(new Set(observations.map(({ variant_id: variantId }) => variantId)).size, 40);
  assert.ok(
    observations.every(
      (observation) =>
        observation.width === 1024 &&
        observation.height === 768 &&
        observation.transform_implementation_version === options.transformImplementationVersion,
    ),
  );

  const expectedFamilySeverities = {
    clean: ["clean"],
    jpeg: ["quality-30", "quality-50", "quality-70", "quality-90"],
    blur: ["sigma-0.5", "sigma-1", "sigma-2"],
    resize: ["factor-0.25", "factor-0.5"],
    noise: ["sigma-0.02", "sigma-0.05", "sigma-0.1"],
    color: [
      "brightness-0.8",
      "brightness-1.2",
      "contrast-0.8",
      "contrast-1.2",
      "saturation-0.8",
      "saturation-1.2",
    ],
    crop: ["center-0.8"],
  };
  for (const source of selectedSources) {
    const sourceObservations = observations.filter(
      ({ source_id: sourceId }) => sourceId === source.source_id,
    );
    for (const [family, severities] of Object.entries(expectedFamilySeverities)) {
      assert.deepEqual(
        sourceObservations
          .filter(({ condition_family: conditionFamily }) => conditionFamily === family)
          .map(({ severity }) => severity)
          .toSorted(),
        severities,
      );
    }
  }

  const authenticMatrix = observations
    .filter(({ authenticity_label: label }) => label === 0)
    .map(({ condition_family: family, severity }) => `${family}:${severity}`);
  const syntheticMatrix = observations
    .filter(({ authenticity_label: label }) => label === 1)
    .map(({ condition_family: family, severity }) => `${family}:${severity}`);
  assert.deepEqual(authenticMatrix, syntheticMatrix);
});

test("leakage audit detects cross-partition and organizer exact or perceptual overlap", () => {
  const selectedSources = selectTrack5Sources(
    [
      inventoryRecord({ datasetSplit: "train", index: 1, label: 0 }),
      inventoryRecord({ datasetSplit: "train", index: 2, label: 0 }),
      inventoryRecord({ datasetSplit: "train", index: 1, label: 1 }),
      inventoryRecord({ datasetSplit: "train", index: 2, label: 1 }),
    ],
    {
      datasetRevision: "saberzl/SID_Set@dc03ead",
      splitSeed: 17,
      splitPlan: [
        { split: "expert-training", datasetSplit: "train", perClass: 1 },
        { split: "fusion-training", datasetSplit: "train", perClass: 1 },
      ],
    },
  );
  const cleanAudit = auditTrack5Sources(selectedSources, { perceptualDistance: 0 });
  assert.equal(cleanAudit.status, "passed");
  assert.equal(cleanAudit.organizer_demonstration.status, "not-available");
  assert.deepEqual(cleanAudit.organizer_demonstration.prohibited_uses, [
    "training",
    "calibration",
    "model-selection",
    "threshold-fitting",
  ]);
  assert.throws(
    () =>
      auditTrack5Sources(selectedSources, {
        organizerHashes: [],
        perceptualDistance: 0,
      }),
    /non-empty array/i,
  );

  const expertSource = selectedSources.find(({ split }) => split === "expert-training");
  const fusionSource = selectedSources.find(({ split }) => split === "fusion-training");
  const contaminated = selectedSources.map((source) => {
    if (source.source_id !== fusionSource.source_id) return source;
    return {
      ...source,
      exact_sha256: expertSource.exact_sha256,
      perceptual_hash: `${(BigInt(`0x${expertSource.perceptual_hash}`) ^ 1n)
        .toString(16)
        .padStart(16, "0")}`,
    };
  });
  const failedAudit = auditTrack5Sources(contaminated, {
    organizerHashes: [
      {
        image_id: "coco-val2017-example",
        collection: "COCO val2017",
        exact_sha256: selectedSources[1].exact_sha256,
        perceptual_hash: selectedSources[1].perceptual_hash,
      },
    ],
    perceptualDistance: 1,
  });

  assert.equal(failedAudit.status, "failed");
  assert.ok(failedAudit.cross_partition_exact.length > 0);
  assert.ok(failedAudit.cross_partition_perceptual.length > 0);
  assert.ok(failedAudit.organizer_demonstration.overlaps.length > 0);
  assert.throws(() => assertLeakageAuditPassed(failedAudit), /leakage audit failed/i);
});

test("versioned Track 5 manifest captures selection, runtime, provenance, and audit metadata", () => {
  const inventory = [
    inventoryRecord({ datasetSplit: "train", index: 1, label: 0 }),
    inventoryRecord({ datasetSplit: "train", index: 1, label: 1 }),
  ];
  const options = {
    artifactSchemaVersion: "artifact-v1",
    corruptionSeed: 23,
    datasetRevision: "saberzl/SID_Set@dc03ead",
    perceptualDistance: 0,
    preprocessingVersion: "shared-preprocessing-v1",
    splitPlan: [{ split: "expert-training", datasetSplit: "train", perClass: 1 }],
    splitSeed: 17,
    transformImplementationVersion: "track5-corruption-v1+sharp-0.35.4",
  };
  const manifest = buildTrack5Manifest(inventory, options);
  const repeated = buildTrack5Manifest(inventory.toReversed(), options);

  assert.deepEqual(manifest, repeated);
  assert.equal(manifest.manifest_schema_version, "track5-manifest-v1");
  assert.equal(manifest.selection.source_count, 2);
  assert.equal(manifest.selection.partition_unit, "source-image");
  assert.deepEqual(manifest.selection.split_counts, {
    "expert-training:class-0": 1,
    "expert-training:class-1": 1,
  });
  assert.equal(manifest.corruption.condition_count_per_source, 20);
  assert.equal(manifest.corruption.sharp_version, "0.35.4");
  assert.match(manifest.corruption.libvips_version, /^\d+\.\d+\.\d+$/u);
  assert.equal(manifest.observations.length, 40);
  assert.equal(manifest.leakage_audit.status, "passed");
  assert.equal(manifest.organizer_demonstration_policy.usage, "evaluation-only");
  assert.ok(
    manifest.sources.every(
      ({ provenance }) =>
        provenance.source_dataset && provenance.source_reference && provenance.license,
    ),
  );
});
