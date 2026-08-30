import {
  compareText,
  ContractError,
  requireFields,
  requireLowercaseHex,
  requireNonemptyString,
  requireNonnegativeInteger,
  requireObject,
} from "./contract-validation.js";

export const ORGANIZER_DEMONSTRATION_POLICY = Object.freeze({
  usage: "evaluation-only",
  prohibited_uses: Object.freeze([
    "training",
    "calibration",
    "model-selection",
    "threshold-fitting",
  ]),
});

function hammingDistance(left, right) {
  let difference = left ^ right;
  let distance = 0;
  while (difference !== 0n) {
    difference &= difference - 1n;
    distance += 1;
  }
  return distance;
}

class PerceptualHashIndex {
  root;

  add(source) {
    const hash = BigInt(`0x${source.perceptual_hash}`);
    if (this.root === undefined) {
      this.root = { hash, sources: [source], children: new Map() };
      return;
    }

    let node = this.root;
    while (true) {
      const distance = hammingDistance(hash, node.hash);
      if (distance === 0) {
        node.sources.push(source);
        return;
      }
      const child = node.children.get(distance);
      if (child === undefined) {
        node.children.set(distance, { hash, sources: [source], children: new Map() });
        return;
      }
      node = child;
    }
  }

  search(perceptualHash, threshold) {
    if (this.root === undefined) return [];
    const hash = BigInt(`0x${perceptualHash}`);
    const matches = [];
    const pending = [this.root];
    while (pending.length > 0) {
      const node = pending.pop();
      const distance = hammingDistance(hash, node.hash);
      if (distance <= threshold) {
        for (const source of node.sources) matches.push({ source, distance });
      }
      const minimum = distance - threshold;
      const maximum = distance + threshold;
      for (const [edgeDistance, child] of node.children) {
        if (edgeDistance >= minimum && edgeDistance <= maximum) pending.push(child);
      }
    }
    return matches;
  }
}

export function createPartitionLeakageGuard(perceptualDistance = 4) {
  requireNonnegativeInteger(
    perceptualDistance,
    "perceptualDistance",
    "Track 5 partition leakage guard options",
  );
  if (perceptualDistance > 64) {
    throw new ContractError(
      "Track 5 partition leakage guard perceptualDistance cannot exceed 64.",
    );
  }
  const exactByHash = new Map();
  const perceptualIndex = new PerceptualHashIndex();

  return Object.freeze({
    add(source) {
      const exactBucket = exactByHash.get(source.exact_sha256) ?? [];
      exactBucket.push(source);
      exactByHash.set(source.exact_sha256, exactBucket);
      perceptualIndex.add(source);
    },
    conflicts(source) {
      if (
        (exactByHash.get(source.exact_sha256) ?? []).some(
          (previous) => previous.split !== source.split,
        )
      ) {
        return true;
      }
      return perceptualIndex
        .search(source.perceptual_hash, perceptualDistance)
        .some(({ source: previous }) => previous.split !== source.split);
    },
  });
}

function normalizeSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) {
    throw new ContractError("Track 5 leakage audit requires a non-empty source array.");
  }
  return sources.map((source, index) => {
    const contractName = `Track 5 leakage source ${index}`;
    requireObject(source, contractName);
    requireFields(
      source,
      ["source_id", "split", "exact_sha256", "perceptual_hash"],
      contractName,
    );
    requireNonemptyString(source.source_id, "source_id", contractName);
    requireNonemptyString(source.split, "split", contractName);
    requireLowercaseHex(source.exact_sha256, "exact_sha256", 64, contractName);
    requireLowercaseHex(source.perceptual_hash, "perceptual_hash", 16, contractName);
    return source;
  });
}

function collisionKey(collision) {
  return [
    collision.left_source_id ?? collision.source_id,
    collision.right_source_id ?? collision.organizer_image_id,
    collision.distance ?? -1,
    collision.match_type ?? "",
  ].join("\0");
}

function sortCollisions(collisions) {
  return collisions.toSorted((left, right) => compareText(collisionKey(left), collisionKey(right)));
}

function crossPartitionAudits(sources, perceptualDistance) {
  const exactByHash = new Map();
  const perceptualIndex = new PerceptualHashIndex();
  const exact = [];
  const perceptual = [];

  for (const source of sources.toSorted((left, right) => compareText(left.source_id, right.source_id))) {
    for (const previous of exactByHash.get(source.exact_sha256) ?? []) {
      if (previous.split !== source.split) {
        exact.push({
          left_source_id: previous.source_id,
          left_split: previous.split,
          right_source_id: source.source_id,
          right_split: source.split,
          exact_sha256: source.exact_sha256,
        });
      }
    }
    const exactBucket = exactByHash.get(source.exact_sha256) ?? [];
    exactBucket.push(source);
    exactByHash.set(source.exact_sha256, exactBucket);

    for (const { source: previous, distance } of perceptualIndex.search(
      source.perceptual_hash,
      perceptualDistance,
    )) {
      if (previous.split !== source.split) {
        perceptual.push({
          left_source_id: previous.source_id,
          left_split: previous.split,
          right_source_id: source.source_id,
          right_split: source.split,
          distance,
          threshold: perceptualDistance,
        });
      }
    }
    perceptualIndex.add(source);
  }
  return {
    exactByHash,
    perceptualIndex,
    exact: sortCollisions(exact),
    perceptual: sortCollisions(perceptual),
  };
}

function organizerAudit(organizerHashes, exactByHash, perceptualIndex, perceptualDistance) {
  if (organizerHashes === undefined) {
    return {
      status: "not-available",
      usage: ORGANIZER_DEMONSTRATION_POLICY.usage,
      prohibited_uses: ORGANIZER_DEMONSTRATION_POLICY.prohibited_uses,
      overlaps: [],
    };
  }
  if (!Array.isArray(organizerHashes)) {
    throw new ContractError("Organizer demonstration hashes must be an array when provided.");
  }
  if (organizerHashes.length === 0) {
    throw new ContractError(
      "Organizer demonstration hashes must be a non-empty array when provided.",
    );
  }

  const overlapKeys = new Set();
  const overlaps = [];
  for (const [index, organizerImage] of organizerHashes.entries()) {
    const contractName = `Organizer demonstration hash ${index}`;
    requireObject(organizerImage, contractName);
    requireFields(
      organizerImage,
      ["image_id", "collection", "exact_sha256", "perceptual_hash"],
      contractName,
    );
    requireNonemptyString(organizerImage.image_id, "image_id", contractName);
    requireNonemptyString(organizerImage.collection, "collection", contractName);
    requireLowercaseHex(organizerImage.exact_sha256, "exact_sha256", 64, contractName);
    requireLowercaseHex(organizerImage.perceptual_hash, "perceptual_hash", 16, contractName);

    for (const source of exactByHash.get(organizerImage.exact_sha256) ?? []) {
      const overlap = {
        source_id: source.source_id,
        split: source.split,
        organizer_image_id: organizerImage.image_id,
        organizer_collection: organizerImage.collection,
        match_type: "exact",
        distance: 0,
      };
      overlapKeys.add(collisionKey(overlap));
      overlaps.push(overlap);
    }
    for (const { source, distance } of perceptualIndex.search(
      organizerImage.perceptual_hash,
      perceptualDistance,
    )) {
      const overlap = {
        source_id: source.source_id,
        split: source.split,
        organizer_image_id: organizerImage.image_id,
        organizer_collection: organizerImage.collection,
        match_type: distance === 0 ? "perceptual-exact" : "perceptual-near",
        distance,
      };
      const key = collisionKey(overlap);
      if (!overlapKeys.has(key)) {
        overlapKeys.add(key);
        overlaps.push(overlap);
      }
    }
  }

  return {
    status: overlaps.length === 0 ? "passed" : "failed",
    usage: ORGANIZER_DEMONSTRATION_POLICY.usage,
    prohibited_uses: ORGANIZER_DEMONSTRATION_POLICY.prohibited_uses,
    overlaps: sortCollisions(overlaps),
  };
}

export function auditTrack5Sources(
  sources,
  { organizerHashes, perceptualDistance = 4 } = {},
) {
  requireNonnegativeInteger(
    perceptualDistance,
    "perceptualDistance",
    "Track 5 leakage audit options",
  );
  if (perceptualDistance > 64) {
    throw new ContractError("Track 5 leakage audit perceptualDistance cannot exceed 64.");
  }
  const normalizedSources = normalizeSources(sources);
  const crossPartition = crossPartitionAudits(normalizedSources, perceptualDistance);
  const organizer = organizerAudit(
    organizerHashes,
    crossPartition.exactByHash,
    crossPartition.perceptualIndex,
    perceptualDistance,
  );
  const passed =
    crossPartition.exact.length === 0 &&
    crossPartition.perceptual.length === 0 &&
    organizer.status !== "failed";

  return Object.freeze({
    audit_schema_version: "track5-leakage-audit-v1",
    status: passed ? "passed" : "failed",
    perceptual_distance_threshold: perceptualDistance,
    source_count: normalizedSources.length,
    cross_partition_exact: Object.freeze(crossPartition.exact),
    cross_partition_perceptual: Object.freeze(crossPartition.perceptual),
    organizer_demonstration: Object.freeze(organizer),
  });
}

export function assertLeakageAuditPassed(audit) {
  requireObject(audit, "Track 5 leakage audit");
  if (audit.status !== "passed") {
    const exactCount = audit.cross_partition_exact?.length ?? 0;
    const perceptualCount = audit.cross_partition_perceptual?.length ?? 0;
    const organizerCount = audit.organizer_demonstration?.overlaps?.length ?? 0;
    throw new ContractError(
      `Track 5 leakage audit failed: ${exactCount} exact cross-partition, ${perceptualCount} perceptual cross-partition, and ${organizerCount} organizer overlap(s).`,
    );
  }
  return audit;
}
