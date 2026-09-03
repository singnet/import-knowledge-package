"""Safe packing and extraction of the fixed memory archive layout."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

from .errors import ArchiveValidationError

FORMAT_VERSION = 1
MANIFEST_NAME = "manifest.json"
COMPONENT_FILES = {
    "history": {"history/history.metta"},
    "ltm": {"vector/collections.json", "vector/records.jsonl"},
}
ALLOWLIST = frozenset({MANIFEST_NAME, *set().union(*COMPONENT_FILES.values())})
MAX_COMPRESSED_BYTES = 500 * 1024 * 1024
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024


def expected_members(components: list[str]) -> set[str]:
    members = {MANIFEST_NAME}
    for component in components:
        try:
            members.update(COMPONENT_FILES[component])
        except KeyError as exc:
            raise ArchiveValidationError(f"Unknown memory component: {component!r}") from exc
    return members


def pack(staging: Path, destination: Path, members: set[str]) -> None:
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    if not members or not members <= ALLOWLIST:
        raise ArchiveValidationError("Archive member set is empty or unsupported")
    missing = [name for name in members if not (staging / name).is_file()]
    if missing:
        raise ArchiveValidationError(f"Cannot package missing files: {sorted(missing)}")

    with tarfile.open(destination, "w:gz") as output:
        for name in sorted(members):
            output.add(staging / name, arcname=name, recursive=False)


def unpack(archive_path: Path, destination: Path) -> set[str]:
    try:
        compressed_size = archive_path.stat().st_size
    except OSError as exc:
        raise ArchiveValidationError(f"Cannot read archive: {exc}") from exc
    if compressed_size > MAX_COMPRESSED_BYTES:
        raise ArchiveValidationError(
            f"Archive is {compressed_size} bytes; limit is {MAX_COMPRESSED_BYTES}"
        )

    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    seen: set[str] = set()
    extracted_size = 0

    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            for member in members:
                name = member.name
                path = Path(name)
                if name not in ALLOWLIST:
                    raise ArchiveValidationError(f"Unexpected archive member: {name!r}")
                if name in seen:
                    raise ArchiveValidationError(f"Duplicate archive member: {name!r}")
                if path.is_absolute() or ".." in path.parts:
                    raise ArchiveValidationError(f"Unsafe archive path: {name!r}")
                if not member.isfile():
                    raise ArchiveValidationError(f"Non-regular archive member: {name!r}")
                extracted_size += member.size
                if extracted_size > MAX_EXTRACTED_BYTES:
                    raise ArchiveValidationError(
                        f"Extracted archive exceeds {MAX_EXTRACTED_BYTES} bytes"
                    )
                seen.add(name)

            for member in members:
                target = (base / member.name).resolve()
                if base not in target.parents:
                    raise ArchiveValidationError(
                        f"Archive member escapes destination: {member.name!r}"
                    )
                source = archive.extractfile(member)
                if source is None:
                    raise ArchiveValidationError(
                        f"Cannot read archive member: {member.name!r}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    except (tarfile.TarError, OSError) as exc:
        if isinstance(exc, ArchiveValidationError):
            raise
        raise ArchiveValidationError(f"Unreadable archive: {exc}") from exc
    return seen
