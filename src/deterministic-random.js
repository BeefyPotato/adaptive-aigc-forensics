import { createHash } from "node:crypto";

export function deterministicHexRank(seed, ...parts) {
  return createHash("sha256").update([seed, ...parts].join("\0"), "utf8").digest("hex");
}

export function deterministicStandardNormal(seed, channelIndex) {
  const digest = createHash("sha256").update(`${seed}:${channelIndex}`, "utf8").digest();
  const denominator = 2 ** 32 + 1;
  const firstUniform = (digest.readUInt32BE(0) + 1) / denominator;
  const secondUniform = (digest.readUInt32BE(4) + 1) / denominator;
  return Math.sqrt(-2 * Math.log(firstUniform)) * Math.cos(2 * Math.PI * secondUniform);
}
