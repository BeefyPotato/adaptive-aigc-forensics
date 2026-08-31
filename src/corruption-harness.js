import sharp from "sharp";

import {
  ContractError,
  requireFields,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";
import { applySeededRgbNoiseBytes } from "./seeded-rgb-noise.js";
import { requireTrack5Condition } from "./track5-conditions.js";

function rawSharp(observation) {
  return sharp(observation.data, {
    raw: {
      width: observation.width,
      height: observation.height,
      channels: observation.channels,
    },
  });
}

async function rawObservation(pipeline, transformDetails) {
  const { data, info } = await pipeline
    .removeAlpha()
    .toColourspace("srgb")
    .raw()
    .toBuffer({ resolveWithObject: true });
  return Object.freeze({
    data,
    width: info.width,
    height: info.height,
    channels: info.channels,
    transform_details: Object.freeze(transformDetails),
  });
}

function validateObservation(observation) {
  requireObject(observation, "decoded source image");
  requireFields(observation, ["data", "width", "height", "channels"], "decoded source image");
  if (!Buffer.isBuffer(observation.data)) {
    throw new ContractError("decoded source image.data must be a Buffer.");
  }
  if (observation.channels !== 3) {
    throw new ContractError("decoded source image must contain exactly three RGB channels.");
  }
}

export async function decodeSourceImage(input, sourceDescription) {
  const description =
    sourceDescription ?? (Buffer.isBuffer(input) ? "verified source bytes" : input);
  try {
    return await rawObservation(sharp(input, { failOn: "error" }).autoOrient(), {
      operation: "decode",
    });
  } catch (error) {
    throw new ContractError(`source image could not be decoded from ${description}: ${error.message}`);
  }
}

async function applyJpegRoundTrip(observation, parameters) {
  const encoded = await rawSharp(observation)
    .jpeg({
      quality: parameters.quality,
      chromaSubsampling: parameters.chroma_subsampling,
      progressive: false,
    })
    .toBuffer();
  const metadata = await sharp(encoded).metadata();
  if (metadata.format !== "jpeg") {
    throw new ContractError(`JPEG corruption encoded an unexpected ${metadata.format} intermediate.`);
  }
  const samplingFactors = readJpegSamplingFactors(encoded);
  if (samplingFactors.join(",") !== "2x2,1x1,1x1") {
    throw new ContractError(
      `JPEG corruption encoded sampling factors ${samplingFactors.join(",")}; expected 2x2,1x1,1x1 for 4:2:0.`,
    );
  }
  return rawObservation(sharp(encoded), {
    operation: "jpeg-round-trip",
    intermediate_format: metadata.format,
    quality: parameters.quality,
    chroma_subsampling: parameters.chroma_subsampling,
    sampling_factors: Object.freeze(samplingFactors),
  });
}

function readJpegSamplingFactors(encoded) {
  if (encoded.length < 4 || encoded[0] !== 0xff || encoded[1] !== 0xd8) {
    throw new ContractError("JPEG corruption intermediate is missing the JPEG start marker.");
  }
  const startOfFrameMarkers = new Set([
    0xc0,
    0xc1,
    0xc2,
    0xc3,
    0xc5,
    0xc6,
    0xc7,
    0xc9,
    0xca,
    0xcb,
    0xcd,
    0xce,
    0xcf,
  ]);
  let offset = 2;
  while (offset + 3 < encoded.length) {
    if (encoded[offset] !== 0xff) {
      offset += 1;
      continue;
    }
    while (encoded[offset] === 0xff) offset += 1;
    const marker = encoded[offset];
    offset += 1;
    if (marker === 0xd8 || marker === 0xd9 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) {
      continue;
    }
    if (offset + 1 >= encoded.length) break;
    const segmentLength = encoded.readUInt16BE(offset);
    if (segmentLength < 2 || offset + segmentLength > encoded.length) break;
    if (startOfFrameMarkers.has(marker)) {
      const componentCount = encoded[offset + 7];
      const componentStart = offset + 8;
      if (componentCount !== 3 || componentStart + componentCount * 3 > offset + segmentLength) {
        throw new ContractError("JPEG corruption intermediate has an unexpected component layout.");
      }
      return Array.from({ length: componentCount }, (_, index) => {
        const sampling = encoded[componentStart + index * 3 + 1];
        return `${sampling >> 4}x${sampling & 0x0f}`;
      });
    }
    offset += segmentLength;
  }
  throw new ContractError("JPEG corruption intermediate has no supported start-of-frame marker.");
}

function gaussianKernel(sigma) {
  const radius = Math.ceil(3 * sigma);
  const size = radius * 2 + 1;
  const values = [];
  let total = 0;
  for (let row = -radius; row <= radius; row += 1) {
    for (let column = -radius; column <= radius; column += 1) {
      const value = Math.exp(-(row ** 2 + column ** 2) / (2 * sigma ** 2));
      values.push(value);
      total += value;
    }
  }
  return {
    size,
    values: values.map((value) => value / total),
  };
}

async function applyGaussianBlur(observation, parameters) {
  const kernel = gaussianKernel(parameters.sigma);
  return rawObservation(
    rawSharp(observation).convolve({
      width: kernel.size,
      height: kernel.size,
      kernel: kernel.values,
      scale: 1,
      offset: 0,
    }),
    {
      operation: "gaussian-blur",
      sigma: parameters.sigma,
      kernel_width: kernel.size,
      kernel_height: kernel.size,
    },
  );
}

async function applyResizeRoundTrip(observation, parameters) {
  const downWidth = Math.max(1, Math.round(observation.width * parameters.factor));
  const downHeight = Math.max(1, Math.round(observation.height * parameters.factor));
  const downsampled = await rawObservation(
    rawSharp(observation).resize({
      width: downWidth,
      height: downHeight,
      fit: "fill",
      kernel: parameters.down_kernel,
      fastShrinkOnLoad: false,
    }),
    { operation: "downscale" },
  );
  return rawObservation(
    rawSharp(downsampled).resize({
      width: observation.width,
      height: observation.height,
      fit: "fill",
      kernel: parameters.up_kernel,
    }),
    {
      operation: "resize-round-trip",
      factor: parameters.factor,
      down_width: downWidth,
      down_height: downHeight,
      down_kernel: parameters.down_kernel,
      up_kernel: parameters.up_kernel,
    },
  );
}

async function applySeededRgbNoise(observation, parameters, seed, executionOptions) {
  const data = await applySeededRgbNoiseBytes(observation.data, {
    seed,
    sigma: parameters.sigma,
    workerCount: executionOptions.noiseWorkerCount,
  });
  return Object.freeze({
    data,
    width: observation.width,
    height: observation.height,
    channels: observation.channels,
    transform_details: Object.freeze({
      operation: "seeded-rgb-noise",
      sigma: parameters.sigma,
      seed,
      color_space: parameters.color_space,
      clamped: true,
    }),
  });
}

async function applyAtomicColor(observation, parameters) {
  let pipeline = rawSharp(observation);
  if (parameters.property === "contrast") {
    pipeline = pipeline.linear(parameters.factor, 128 * (1 - parameters.factor));
  } else {
    pipeline = pipeline.modulate({ [parameters.property]: parameters.factor });
  }
  return rawObservation(pipeline, {
    operation: "atomic-color",
    property: parameters.property,
    factor: parameters.factor,
  });
}

async function applyCenterCropRoundTrip(observation, parameters) {
  const cropWidth = Math.max(1, Math.round(observation.width * parameters.retained_fraction));
  const cropHeight = Math.max(1, Math.round(observation.height * parameters.retained_fraction));
  const cropLeft = Math.floor((observation.width - cropWidth) / 2);
  const cropTop = Math.floor((observation.height - cropHeight) / 2);
  return rawObservation(
    rawSharp(observation)
      .extract({ left: cropLeft, top: cropTop, width: cropWidth, height: cropHeight })
      .resize({
        width: observation.width,
        height: observation.height,
        fit: "fill",
        kernel: parameters.restoration_kernel,
      }),
    {
      operation: "center-crop-round-trip",
      retained_fraction: parameters.retained_fraction,
      crop_left: cropLeft,
      crop_top: cropTop,
      crop_width: cropWidth,
      crop_height: cropHeight,
      restoration_kernel: parameters.restoration_kernel,
    },
  );
}

function applyClean(observation) {
  return Object.freeze({
    data: Buffer.from(observation.data),
    width: observation.width,
    height: observation.height,
    channels: observation.channels,
    transform_details: Object.freeze({ operation: "clean" }),
  });
}

const CORRUPTION_APPLICATORS = new Map([
  ["clean", applyClean],
  ["jpeg", applyJpegRoundTrip],
  ["blur", applyGaussianBlur],
  ["resize", applyResizeRoundTrip],
  ["noise", applySeededRgbNoise],
  ["color", applyAtomicColor],
  ["crop", applyCenterCropRoundTrip],
]);

export async function applyCorruption(
  observation,
  corruptionVariant,
  executionOptions = {},
) {
  validateObservation(observation);
  requireObject(executionOptions, "corruption execution options");
  requireObject(corruptionVariant, "corruption variant");
  requireFields(
    corruptionVariant,
    ["condition_family", "corruption_parameters", "corruption_seed"],
    "corruption variant",
  );
  requireObject(corruptionVariant.corruption_parameters, "corruption variant.corruption_parameters");
  requireNonnegativeInteger(
    corruptionVariant.corruption_seed,
    "corruption_seed",
    "corruption variant",
  );
  const noiseWorkerCount = executionOptions.noiseWorkerCount ?? 1;
  requireNonnegativeInteger(
    noiseWorkerCount,
    "noiseWorkerCount",
    "corruption execution options",
  );
  if (noiseWorkerCount === 0) {
    throw new ContractError(
      "corruption execution options.noiseWorkerCount must be positive.",
    );
  }

  requireTrack5Condition(
    corruptionVariant.condition_family,
    corruptionVariant.corruption_parameters,
  );
  const applicator = CORRUPTION_APPLICATORS.get(corruptionVariant.condition_family);
  if (applicator === undefined) {
    throw new ContractError(
      `corruption variant.condition_family ${JSON.stringify(corruptionVariant.condition_family)} has no implementation.`,
    );
  }
  return applicator(
    observation,
    corruptionVariant.corruption_parameters,
    corruptionVariant.corruption_seed,
    { noiseWorkerCount },
  );
}
