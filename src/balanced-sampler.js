import { createHash } from "node:crypto";

import {
  ContractError,
  requireFields,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { TRACK5_CONDITION_MATRIX } from "./track5-manifest.js";

const TRAINING_SPLITS = new Set(["expert-training", "fusion-training"]);

function compareText(left, right) {
  return left < right ? -1 : left > right ? 1 : 0;
}

function rank(seed, ...parts) {
  return createHash("sha256").update([seed, ...parts].join("\0"), "utf8").digest("hex");
}

function conditionHierarchy() {
  const families = new Map();
  for (const { family, severity } of TRACK5_CONDITION_MATRIX) {
    const severities = families.get(family) ?? [];
    severities.push(severity);
    families.set(family, severities);
  }
  return families;
}

function validateManifest(manifest, split) {
  requireObject(manifest, "Track 5 sampler manifest");
  requireFields(manifest, ["sources", "observations"], "Track 5 sampler manifest");
  if (!Array.isArray(manifest.sources) || !Array.isArray(manifest.observations)) {
    throw new ContractError("Track 5 sampler manifest sources and observations must be arrays.");
  }
  const sources = manifest.sources.filter((source) => source.split === split);
  if (sources.length === 0) {
    throw new ContractError(`Track 5 sampler found no sources for training split ${split}.`);
  }
  const sourceIds = new Set();
  const sourceById = new Map();
  for (const [index, source] of sources.entries()) {
    const contractName = `Track 5 sampler source ${index}`;
    requireObject(source, contractName);
    requireFields(source, ["source_id", "authenticity_label", "split"], contractName);
    requireNonemptyString(source.source_id, "source_id", contractName);
    if (sourceIds.has(source.source_id)) {
      throw new ContractError(`Track 5 sampler repeats source identity ${source.source_id}.`);
    }
    sourceIds.add(source.source_id);
    sourceById.set(source.source_id, source);
    if (source.authenticity_label !== 0 && source.authenticity_label !== 1) {
      throw new ContractError(`${contractName}.authenticity_label must be 0 or 1.`);
    }
    if (
      source.dataset === "organizer-demonstration" ||
      source.usage === "evaluation-only"
    ) {
      throw new ContractError(
        `Organizer demonstration source ${source.source_id} is prohibited from training.`,
      );
    }
  }
  return { sources, sourceById, sourceIds };
}

function indexObservations(observations, sourceById, sourceIds, split) {
  const byCondition = new Map();
  for (const [index, observation] of observations.entries()) {
    if (observation.split !== split || !sourceIds.has(observation.source_id)) continue;
    const contractName = `Track 5 sampler observation ${index}`;
    requireObject(observation, contractName);
    requireFields(
      observation,
      [
        "variant_id",
        "source_id",
        "authenticity_label",
        "condition_family",
        "severity",
      ],
      contractName,
    );
    const source = sourceById.get(observation.source_id);
    if (observation.authenticity_label !== source.authenticity_label) {
      throw new ContractError(
        `${contractName}.authenticity_label does not match source ${source.source_id}.`,
      );
    }
    const key = [
      observation.source_id,
      observation.condition_family,
      observation.severity,
    ].join("\0");
    if (byCondition.has(key)) {
      throw new ContractError(
        `Track 5 sampler has multiple observations for ${observation.source_id}, ${observation.condition_family}, ${observation.severity}.`,
      );
    }
    byCondition.set(key, observation);
  }
  return byCondition;
}

export function sampleBalancedTrainingObservations(
  manifest,
  { split, count, seed },
) {
  requireNonemptyString(split, "split", "Track 5 sampler options");
  if (!TRAINING_SPLITS.has(split)) {
    throw new ContractError(
      `Track 5 sampler split must be a training split: ${[...TRAINING_SPLITS].join(" or ")}.`,
    );
  }
  requireNonnegativeInteger(count, "count", "Track 5 sampler options");
  requireNonnegativeInteger(seed, "seed", "Track 5 sampler options");
  if (count === 0) throw new ContractError("Track 5 sampler count must be greater than zero.");

  const families = conditionHierarchy();
  const stratumCount = 2 * families.size;
  if (count % stratumCount !== 0) {
    throw new ContractError(
      `Track 5 sampler count must be divisible by ${stratumCount} for equal class and family allocation.`,
    );
  }
  const perClassFamily = count / stratumCount;
  for (const [family, severities] of families) {
    if (perClassFamily % severities.length !== 0) {
      throw new ContractError(
        `Track 5 sampler allocates ${perClassFamily} observations per class/${family}; this must be divisible by its ${severities.length} severities.`,
      );
    }
  }

  const { sources, sourceById, sourceIds } = validateManifest(manifest, split);
  const sourcesByClass = new Map(
    [0, 1].map((label) => [
      label,
      sources.filter(({ authenticity_label: actual }) => actual === label),
    ]),
  );
  for (const [label, classSources] of sourcesByClass) {
    if (classSources.length === 0) {
      throw new ContractError(`Track 5 sampler found no class-${label} sources in ${split}.`);
    }
  }
  const observations = indexObservations(manifest.observations, sourceById, sourceIds, split);
  const sourceCounts = new Map(sources.map(({ source_id: sourceId }) => [sourceId, 0]));
  const sampled = [];

  for (const label of [0, 1]) {
    const classSources = sourcesByClass.get(label);
    for (const [family, severities] of families) {
      const perSeverity = perClassFamily / severities.length;
      for (const severity of severities) {
        for (let draw = 0; draw < perSeverity; draw += 1) {
          const source = classSources.toSorted((left, right) => {
            const countDifference =
              sourceCounts.get(left.source_id) - sourceCounts.get(right.source_id);
            if (countDifference !== 0) return countDifference;
            return compareText(
              rank(seed, label, family, severity, draw, left.source_id),
              rank(seed, label, family, severity, draw, right.source_id),
            );
          })[0];
          const key = [source.source_id, family, severity].join("\0");
          const observation = observations.get(key);
          if (observation === undefined) {
            throw new ContractError(
              `Track 5 sampler is missing ${family}/${severity} for ${source.source_id}.`,
            );
          }
          sampled.push(observation);
          sourceCounts.set(source.source_id, sourceCounts.get(source.source_id) + 1);
        }
      }
    }
  }

  return Object.freeze(
    sampled
      .map((observation, index) => ({
        observation,
        order: rank(seed, "output", index, observation.variant_id),
      }))
      .toSorted((left, right) => compareText(left.order, right.order))
      .map(({ observation }) => observation),
  );
}
