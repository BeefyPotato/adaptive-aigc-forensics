import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

import { ContractError } from "./contracts.js";

function decodeAsciiPortablePixmap(path) {
  let text;
  try {
    text = readFileSync(path, "utf8");
  } catch (error) {
    throw new ContractError(`source image could not be read from ${path}: ${error.message}`);
  }
  const tokens = text
    .replace(/#[^\r\n]*/g, " ")
    .trim()
    .split(/\s+/);
  if (tokens[0] !== "P3") {
    throw new ContractError(
      `fixture image ${path} uses ${JSON.stringify(tokens[0])}; shared-preprocessing-v1 supports checked-in P3 images only.`,
    );
  }
  const width = Number(tokens[1]);
  const height = Number(tokens[2]);
  const maximum = Number(tokens[3]);
  if (!Number.isInteger(width) || width <= 0 || !Number.isInteger(height) || height <= 0) {
    throw new ContractError(`fixture image ${path} must declare positive integer dimensions.`);
  }
  if (!Number.isInteger(maximum) || maximum <= 0 || maximum > 65535) {
    throw new ContractError(`fixture image ${path} must declare a channel maximum in [1, 65535].`);
  }
  const expectedChannels = width * height * 3;
  if (tokens.length !== expectedChannels + 4) {
    throw new ContractError(
      `fixture image ${path} has ${tokens.length - 4} channel values; expected ${expectedChannels}.`,
    );
  }
  const channels = tokens.slice(4).map((token, index) => {
    const channel = Number(token);
    if (!Number.isInteger(channel) || channel < 0 || channel > maximum) {
      throw new ContractError(
        `fixture image ${path} channel ${index} must be an integer in [0, ${maximum}].`,
      );
    }
    return channel / maximum;
  });
  const rgb = [];
  for (let index = 0; index < channels.length; index += 3) {
    rgb.push([channels[index], channels[index + 1], channels[index + 2]]);
  }
  return { width, height, rgb };
}

function deterministicStandardNormal(seed, channelIndex) {
  const digest = createHash("sha256").update(`${seed}:${channelIndex}`, "utf8").digest();
  const denominator = 2 ** 32 + 1;
  const firstUniform = (digest.readUInt32BE(0) + 1) / denominator;
  const secondUniform = (digest.readUInt32BE(4) + 1) / denominator;
  return Math.sqrt(-2 * Math.log(firstUniform)) * Math.cos(2 * Math.PI * secondUniform);
}

function applyGaussianNoise(rgb, sigma, seed) {
  if (typeof sigma !== "number" || !Number.isFinite(sigma) || sigma < 0) {
    throw new ContractError("noise corruption sigma must be a finite number greater than or equal to zero.");
  }
  return rgb.map((pixel, pixelIndex) =>
    pixel.map((channel, channelIndex) => {
      const normal = deterministicStandardNormal(seed, pixelIndex * 3 + channelIndex);
      return Math.max(0, Math.min(1, channel + sigma * normal));
    }),
  );
}

export function prepareExpertInputs(path, corruptionParameters, corruptionSeed) {
  const decoded = decodeAsciiPortablePixmap(path);
  const rgb = applyGaussianNoise(decoded.rgb, corruptionParameters.sigma, corruptionSeed).map(
    (pixel) => Object.freeze(pixel),
  );
  const luminance = rgb.map(
    ([red, green, blue]) => 0.2126 * red + 0.7152 * green + 0.0722 * blue,
  );
  return Object.freeze({
    width: decoded.width,
    height: decoded.height,
    rgb: Object.freeze(rgb),
    luminance: Object.freeze(luminance),
  });
}
