import { ContractError } from "./contract-validation.js";

const METRIC_VERSION = "metric-v1";

function mean(values) {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

export function evaluateFixture(evaluatedRecords, config, modelBundle) {
  if (evaluatedRecords.length === 0) {
    throw new ContractError("fixture evaluation requires at least one prediction record.");
  }
  const threshold = 0.5;
  const correct = evaluatedRecords.filter(
    ({ authenticity_label: label, pred }) => Number(pred >= threshold) === label,
  ).length;
  return {
    metric_schema_version: METRIC_VERSION,
    artifact_schema_version: config.artifactSchemaVersion,
    prediction_count: evaluatedRecords.length,
    threshold,
    accuracy: correct / evaluatedRecords.length,
    brier_score: mean(
      evaluatedRecords.map(({ authenticity_label: label, pred }) => (pred - label) ** 2),
    ),
    numeric_tolerance: modelBundle.numericTolerance,
    records: evaluatedRecords.map(({ variant_id: variantId, authenticity_label: label, pred }) => ({
      variant_id: variantId,
      authenticity_label: label,
      pred,
    })),
  };
}
