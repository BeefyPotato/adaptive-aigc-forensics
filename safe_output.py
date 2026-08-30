"""Fail-closed managed-output paths and exclusive atomic file publication."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def _is_redirect(path: Path) -> bool:
    """Return whether an existing path is a symlink, junction, or other reparse point."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as error:
        raise ValueError("Signal managed output path metadata could not be inspected.") from error
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return (
        stat.S_ISLNK(metadata.st_mode)
        or bool(getattr(metadata, "st_file_attributes", 0) & reparse_flag)
        or bool(getattr(metadata, "st_reparse_tag", 0))
    )


def resolve_output_directory(path: Path | str, description: str) -> Path:
    """Create and canonicalize one explicit output root without accepting redirection."""
    requested = Path(path).absolute()
    if _is_redirect(requested):
        raise ValueError(f"{description} must not be a symlink, junction, or reparse point.")
    try:
        requested.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f"{description} could not be created.") from error
    if _is_redirect(requested):
        raise ValueError(f"{description} must not be a symlink, junction, or reparse point.")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{description} could not be resolved.") from error
    if not resolved.is_dir():
        raise ValueError(f"{description} must be a directory.")
    return resolved


def managed_output_path(
    output_directory: Path | str,
    relative_path: Path | str,
    description: str,
) -> Path:
    """Resolve a strictly contained path after rejecting redirected components."""
    root = Path(output_directory)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as error:
        raise ValueError("Signal managed output root could not be resolved.") from error
    if resolved_root != root.absolute() or _is_redirect(root) or not resolved_root.is_dir():
        raise ValueError("Signal managed output root changed or became redirected.")

    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"Signal managed output path {description} must be strictly relative.")
    candidate = resolved_root.joinpath(relative)
    current = resolved_root
    for part in relative.parts:
        current = current / part
        if _is_redirect(current):
            raise ValueError(
                f"Signal managed output path {description} is redirected by a symlink, "
                "junction, or reparse point."
            )
    try:
        resolved_candidate = candidate.resolve(strict=False)
        contained = resolved_candidate.relative_to(resolved_root)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"Signal managed output path {description} resolves outside the output directory."
        ) from error
    if not contained.parts:
        raise ValueError(
            f"Signal managed output path {description} must be strictly inside the output directory."
        )
    return resolved_candidate


def atomic_write_bytes(path: Path | str, value: bytes) -> None:
    """Publish bytes via an unpredictable, exclusively created sibling temporary file."""
    destination = Path(path)
    if _is_redirect(destination):
        raise ValueError("Signal output artifact must not be a redirected path.")
    if not destination.parent.is_dir() or _is_redirect(destination.parent):
        raise ValueError("Signal output artifact parent must be an ordinary directory.")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            descriptor = -1
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
