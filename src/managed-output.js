import { lstat, mkdir, realpath } from "node:fs/promises";
import { basename, dirname, isAbsolute, relative, resolve, sep } from "node:path";

import { ContractError } from "./contract-validation.js";

function escapesRoot(root, path) {
  const relation = relative(root, path);
  return relation === ".." || relation.startsWith(`..${sep}`) || isAbsolute(relation);
}

async function existingMetadata(path, description) {
  try {
    return await lstat(path);
  } catch (error) {
    if (error.code === "ENOENT") return undefined;
    throw new ContractError(`Managed output ${description} metadata could not be inspected.`);
  }
}

export async function resolveManagedOutputRoot(path, description) {
  const requested = resolve(path);
  const existing = await existingMetadata(requested, description);
  if (existing?.isSymbolicLink()) {
    throw new ContractError(
      `Managed output ${description} must not be a symlink or junction.`,
    );
  }
  if (existing === undefined) await mkdir(requested, { recursive: true });
  const created = await existingMetadata(requested, description);
  if (created === undefined || !created.isDirectory() || created.isSymbolicLink()) {
    throw new ContractError(`Managed output ${description} must be an ordinary directory.`);
  }
  try {
    return await realpath(requested);
  } catch {
    throw new ContractError(`Managed output ${description} could not be resolved.`);
  }
}

export async function managedOutputPath(root, relativePath, description) {
  const requestedRoot = resolve(root);
  const declaredRootMetadata = await existingMetadata(requestedRoot, "root");
  if (
    declaredRootMetadata === undefined ||
    !declaredRootMetadata.isDirectory() ||
    declaredRootMetadata.isSymbolicLink()
  ) {
    throw new ContractError("Managed output root changed or became redirected.");
  }
  const resolvedRoot = await realpath(requestedRoot).catch(() => {
    throw new ContractError("Managed output root could not be resolved.");
  });
  if (relative(requestedRoot, resolvedRoot) !== "") {
    throw new ContractError("Managed output root changed or became redirected.");
  }
  const rootMetadata = await existingMetadata(resolvedRoot, "root");
  if (rootMetadata === undefined || !rootMetadata.isDirectory() || rootMetadata.isSymbolicLink()) {
    throw new ContractError("Managed output root changed or became redirected.");
  }
  if (isAbsolute(relativePath)) {
    throw new ContractError(`Managed output ${description} must be relative.`);
  }
  const candidate = resolve(resolvedRoot, relativePath);
  if (candidate === resolvedRoot || escapesRoot(resolvedRoot, candidate)) {
    throw new ContractError(`Managed output ${description} escapes the output root.`);
  }
  const relation = relative(resolvedRoot, candidate);
  let current = resolvedRoot;
  for (const part of relation.split(sep)) {
    if (!part || part === "." || part === "..") {
      throw new ContractError(`Managed output ${description} has an invalid relative path.`);
    }
    current = resolve(current, part);
    const metadata = await existingMetadata(current, description);
    if (metadata?.isSymbolicLink()) {
      throw new ContractError(
        `Managed output ${description} is redirected by a symlink or junction.`,
      );
    }
  }
  const parent = dirname(candidate);
  const resolvedParent = await realpath(parent).catch(() => {
    throw new ContractError(`Managed output ${description} parent does not exist.`);
  });
  const resolvedCandidate = resolve(resolvedParent, basename(candidate));
  if (escapesRoot(resolvedRoot, resolvedCandidate)) {
    throw new ContractError(`Managed output ${description} resolves outside the output root.`);
  }
  return resolvedCandidate;
}

export async function ensureManagedDirectory(root, relativePath, description) {
  const candidate = await managedOutputPath(root, relativePath, description);
  try {
    await mkdir(candidate);
  } catch (error) {
    if (error.code !== "EEXIST") throw error;
  }
  const verified = await managedOutputPath(root, relativePath, description);
  const metadata = await existingMetadata(verified, description);
  if (metadata === undefined || !metadata.isDirectory() || metadata.isSymbolicLink()) {
    throw new ContractError(`Managed output ${description} must be an ordinary directory.`);
  }
  return verified;
}
