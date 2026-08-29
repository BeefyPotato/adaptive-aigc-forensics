import { canonicalJson } from "./contracts.js";
import { ContractError } from "./contract-validation.js";

export const TRACK5_CONDITION_MATRIX = Object.freeze([
  Object.freeze({ family: "clean", severity: "clean", parameters: Object.freeze({}) }),
  ...[90, 70, 50, 30].map((quality) =>
    Object.freeze({
      family: "jpeg",
      severity: `quality-${quality}`,
      parameters: Object.freeze({ quality, chroma_subsampling: "4:2:0" }),
    }),
  ),
  ...[0.5, 1, 2].map((sigma) =>
    Object.freeze({
      family: "blur",
      severity: `sigma-${sigma}`,
      parameters: Object.freeze({ sigma }),
    }),
  ),
  ...[0.5, 0.25].map((factor) =>
    Object.freeze({
      family: "resize",
      severity: `factor-${factor}`,
      parameters: Object.freeze({ factor, down_kernel: "lanczos3", up_kernel: "cubic" }),
    }),
  ),
  ...[0.02, 0.05, 0.1].map((sigma) =>
    Object.freeze({
      family: "noise",
      severity: `sigma-${sigma}`,
      parameters: Object.freeze({ sigma, color_space: "rgb-0-1" }),
    }),
  ),
  ...["brightness", "contrast", "saturation"].flatMap((property) =>
    [0.8, 1.2].map((factor) =>
      Object.freeze({
        family: "color",
        severity: `${property}-${factor}`,
        parameters: Object.freeze({ property, factor }),
      }),
    ),
  ),
  Object.freeze({
    family: "crop",
    severity: "center-0.8",
    parameters: Object.freeze({
      retained_fraction: 0.8,
      position: "center",
      restoration_kernel: "cubic",
    }),
  }),
]);

export function requireTrack5Condition(family, parameters) {
  const familyConditions = TRACK5_CONDITION_MATRIX.filter(
    ({ family: allowedFamily }) => allowedFamily === family,
  );
  if (familyConditions.length === 0) {
    throw new ContractError(
      `corruption variant.condition_family ${JSON.stringify(family)} is unsupported.`,
    );
  }
  const serializedParameters = canonicalJson(parameters);
  const condition = familyConditions.find(
    ({ parameters: allowedParameters }) =>
      canonicalJson(allowedParameters) === serializedParameters,
  );
  if (condition === undefined) {
    throw new ContractError(
      `${family} corruption parameters do not match a declared Track 5 condition.`,
    );
  }
  return condition;
}
