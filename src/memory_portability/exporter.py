"""Logical export of Omega history and user LTM records."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .archive import expected_members, pack, unpack
from .storage import DEFAULT_BATCH_SIZE, MemoryStore
from .validator import FORMAT_NAME, file_description, load_and_validate, sha256_file


def export_memory(
    store: MemoryStore,
    transfer_dir: Path,
    component: str = "both",
    include_embeddings: bool = True,
    filename: str | None = None,
) -> dict:
    """Create and atomically publish a memory archive.

    The caller must hold Omega's memory-write lock for the duration of this
    synchronous call when a cross-component point-in-time snapshot is required.
    """
    if component not in {"history", "ltm", "both"}:
        raise ValueError("component must be 'history', 'ltm', or 'both'")
    components = ["history", "ltm"] if component == "both" else [component]
    transfer_dir = transfer_dir.resolve()
    transfer_dir.mkdir(parents=True, exist_ok=True)
    if not transfer_dir.is_dir():
        raise NotADirectoryError(transfer_dir)

    archive_name = filename or _default_filename()
    _validate_filename(archive_name)
    final_path = transfer_dir / archive_name
    if final_path.exists():
        raise FileExistsError(final_path)

    staging = transfer_dir / f".memory-export-{uuid.uuid4().hex}"
    temporary_archive = transfer_dir / f".{archive_name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir()
    record_count = 0
    try:
        if "history" in components:
            history_path = staging / "history/history.metta"
            history_path.parent.mkdir(parents=True)
            history_path.write_text(store.read_history() or "", encoding="utf-8")

        if "ltm" in components:
            records_path = staging / "vector/records.jsonl"
            records_path.parent.mkdir(parents=True)
            with records_path.open("w", encoding="utf-8") as output:
                for batch in store.iter_user_records(
                    batch_size=DEFAULT_BATCH_SIZE,
                    include_embeddings=include_embeddings,
                ):
                    for record in batch:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        record_count += 1
            collections = {
                "collections": [
                    {
                        "name": store.collection_name,
                        "embedding_profile": store.active_embedding_profile(),
                    }
                ]
            }
            (staging / "vector/collections.json").write_text(
                json.dumps(collections, indent=2, sort_keys=True), encoding="utf-8"
            )

        files = {}
        for name in expected_members(components) - {"manifest.json"}:
            files[name] = file_description(staging / name)
        manifest = {
            "format": FORMAT_NAME,
            "format_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "components": components,
            "record_count": record_count,
            "embeddings_included": include_embeddings if "ltm" in components else False,
            "source": store.archive_metadata(),
            "files": files,
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

        members = expected_members(components)
        pack(staging, temporary_archive, members)
        verification = staging / ".verification"
        actual_members = unpack(temporary_archive, verification)
        load_and_validate(verification, actual_members)
        os.replace(temporary_archive, final_path)
        return {
            "filename": archive_name,
            "components": components,
            "record_count": record_count,
            "size": final_path.stat().st_size,
            "sha256": sha256_file(final_path),
        }
    finally:
        temporary_archive.unlink(missing_ok=True)
        shutil.rmtree(staging, ignore_errors=True)


def _default_filename() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"omega-memory-{timestamp}.tar.gz"


def _validate_filename(filename: str) -> None:
    if (
        "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or not filename.endswith(".tar.gz")
    ):
        raise ValueError("filename must be a plain .tar.gz filename")
