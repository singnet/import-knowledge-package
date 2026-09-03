"""Schema and checksum validation for extracted memory archives."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .archive import FORMAT_VERSION, MANIFEST_NAME, expected_members
from .errors import ArchiveValidationError

FORMAT_NAME = "omega-user-memory"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_description(path: Path) -> dict[str, Any]:
    return {"size": path.stat().st_size, "sha256": sha256_file(path)}


def load_and_validate(staging: Path, actual_members: set[str]) -> tuple[dict, bool]:
    manifest_path = staging / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError(f"Invalid manifest.json: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ArchiveValidationError("manifest.json must contain an object")
    if manifest.get("format") != FORMAT_NAME:
        raise ArchiveValidationError("Unsupported memory archive format")
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ArchiveValidationError(
            f"Unsupported format version: {manifest.get('format_version')!r}"
        )
    _validate_manifest_metadata(manifest)

    components = manifest.get("components")
    if (
        not isinstance(components, list)
        or not components
        or len(components) != len(set(components))
        or any(component not in ("history", "ltm") for component in components)
    ):
        raise ArchiveValidationError("Manifest has invalid components")

    expected = expected_members(components)
    if actual_members != expected:
        raise ArchiveValidationError(
            f"Archive members do not match manifest: expected {sorted(expected)}, "
            f"found {sorted(actual_members)}"
        )

    files = manifest.get("files")
    expected_files = expected - {MANIFEST_NAME}
    if not isinstance(files, dict) or set(files) != expected_files:
        raise ArchiveValidationError("Manifest file list does not match its components")
    for name, description in files.items():
        if not isinstance(description, dict):
            raise ArchiveValidationError(f"Invalid file description for {name}")
        path = staging / name
        try:
            size = path.stat().st_size
            checksum = sha256_file(path)
        except OSError as exc:
            raise ArchiveValidationError(f"Cannot verify {name}: {exc}") from exc
        if description.get("size") != size or description.get("sha256") != checksum:
            raise ArchiveValidationError(f"Checksum or size mismatch for {name}")

    if "history" in components:
        _validate_history(staging / "history/history.metta")
    missing_embeddings = False
    if "ltm" in components:
        expected_dimension = _validate_collections(
            staging / "vector/collections.json"
        )
        missing_embeddings = _validate_records(
            staging / "vector/records.jsonl", manifest, expected_dimension
        )
    return manifest, missing_embeddings


def _validate_manifest_metadata(manifest: dict) -> None:
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise ArchiveValidationError("Manifest created_at is required")
    try:
        parsed_created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ArchiveValidationError("Manifest created_at must be ISO-8601") from exc
    if parsed_created_at.tzinfo is None:
        raise ArchiveValidationError("Manifest created_at must include a timezone")

    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ArchiveValidationError("Manifest source is required")
    for key in ("omega_version", "chromadb_version"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise ArchiveValidationError(f"Manifest source.{key} is required")

    if type(manifest.get("embeddings_included")) is not bool:
        raise ArchiveValidationError("Manifest embeddings_included must be a boolean")


def _validate_history(path: Path) -> None:
    try:
        path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ArchiveValidationError(f"Invalid history file: {exc}") from exc


def _validate_collections(path: Path) -> int | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveValidationError(f"Invalid collections.json: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("collections"), list):
        raise ArchiveValidationError("collections.json has invalid structure")
    if len(value["collections"]) != 1:
        raise ArchiveValidationError("Version 1 archives must describe one collection")
    collection = value["collections"][0]
    if not isinstance(collection, dict) or not isinstance(collection.get("name"), str):
        raise ArchiveValidationError("Invalid collection description")
    profile = collection.get("embedding_profile")
    if not isinstance(profile, dict):
        raise ArchiveValidationError("Collection embedding profile is missing")
    dimension = profile.get("vector_dimension")
    if dimension is not None and (type(dimension) is not int or dimension <= 0):
        raise ArchiveValidationError("Invalid embedding dimension")
    return dimension


def _validate_records(
    path: Path, manifest: dict, expected_dimension: int | None
) -> bool:
    expected_count = manifest.get("record_count")
    if type(expected_count) is not int or expected_count < 0:
        raise ArchiveValidationError("Manifest record_count is invalid")
    seen: set[str] = set()
    missing_embeddings = False
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ArchiveValidationError(
                        f"Invalid records.jsonl line {line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ArchiveValidationError(
                        f"Record on line {line_number} is not an object"
                    )
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id or record_id in seen:
                    raise ArchiveValidationError(
                        f"Missing or duplicate record ID on line {line_number}"
                    )
                if not isinstance(record.get("document"), str):
                    raise ArchiveValidationError(
                        f"Record document on line {line_number} is not text"
                    )
                if not isinstance(record.get("metadata"), dict):
                    raise ArchiveValidationError(
                        f"Record metadata on line {line_number} is not an object"
                    )
                embedding = record.get("embedding")
                if embedding is None:
                    missing_embeddings = True
                elif not isinstance(embedding, list) or not embedding or any(
                    not isinstance(value, (int, float)) or isinstance(value, bool)
                    for value in embedding
                ):
                    raise ArchiveValidationError(
                        f"Invalid embedding on line {line_number}"
                    )
                elif expected_dimension is not None and len(embedding) != expected_dimension:
                    raise ArchiveValidationError(
                        f"Embedding on line {line_number} has dimension {len(embedding)}; "
                        f"manifest profile declares {expected_dimension}"
                    )
                seen.add(record_id)
                count += 1
    except (OSError, UnicodeDecodeError) as exc:
        raise ArchiveValidationError(f"Cannot read records.jsonl: {exc}") from exc
    if count != expected_count:
        raise ArchiveValidationError(
            f"Manifest declares {expected_count} records but archive contains {count}"
        )
    return missing_embeddings
