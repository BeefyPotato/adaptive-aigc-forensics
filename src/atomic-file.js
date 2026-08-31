import { randomBytes } from "node:crypto";
import { open, rename, unlink } from "node:fs/promises";
import { basename, dirname, resolve } from "node:path";

export async function writeFileAtomically(path, value) {
  const temporaryPath = resolve(
    dirname(path),
    `.${basename(path)}.${randomBytes(16).toString("hex")}.tmp`,
  );
  let handle;
  try {
    handle = await open(temporaryPath, "wx", 0o600);
    await handle.writeFile(value);
    await handle.sync();
    await handle.close();
    handle = undefined;
    await rename(temporaryPath, path);
  } catch (error) {
    if (handle !== undefined) await handle.close().catch(() => {});
    await unlink(temporaryPath).catch((cleanupError) => {
      if (cleanupError.code !== "ENOENT") throw cleanupError;
    });
    throw error;
  }
}
