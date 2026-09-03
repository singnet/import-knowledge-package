"""Validated, transactional overwrite and append restore operations."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .archive import unpack
from .errors import MemoryImportError, RecoveryError
from .records import (
    iter_records,
    profiles_compatible,
    reembed_records,
    source_embedding_profile,
)
from .storage import MemoryStore
from .validator import load_and_validate, sha256_file

MARKER_NAME = ".memory-import-in-progress.json"
ROLLBACK_NAME = ".memory-import-rollback"
RECEIPTS_NAME = ".memory-import-receipts"


def import_archive(
    store: MemoryStore,
    transfer_dir: Path,
    filename: str,
    mode: str = "overwrite",
    include_history: bool = True,
    include_vectors: bool = True,
) -> dict:
    if mode not in {"overwrite", "append"}:
        raise ValueError("mode must be 'overwrite' or 'append'")
    if not include_history and not include_vectors:
        raise ValueError("At least one memory component must be enabled")
    _validate_filename(filename)
    archive_path = transfer_dir.resolve() / filename
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    state_dir = store.state_dir
    marker = state_dir / MARKER_NAME
    if marker.exists():
        raise RecoveryError("An interrupted import must be recovered before importing")

    staging = state_dir / f".memory-import-staging-{uuid.uuid4().hex}"
    digest = sha256_file(archive_path)
    try:
        actual_members = unpack(archive_path, staging)
        manifest, missing_embeddings = load_and_validate(staging, actual_members)
        components = manifest["components"]
        do_history = include_history and "history" in components
        do_vectors = include_vectors and "ltm" in components
        if not do_history and not do_vectors:
            raise MemoryImportError("Selected components are not present in the archive")

        receipt = _receipt_path(state_dir, digest, mode, do_history, do_vectors)
        if receipt.exists():
            return {"status": "already-imported", "receipt": receipt.name}

        reembedded = False
        if do_vectors:
            source_profile = source_embedding_profile(staging)
            active_profile = store.active_embedding_profile()
            if not profiles_compatible(
                source_profile, active_profile, missing_embeddings
            ):
                reembed_records(staging, store)
                reembedded = True

        if mode == "overwrite":
            _import_overwrite(
                store, staging, do_history, do_vectors, marker, receipt, digest
            )
        else:
            _import_append(
                store, staging, do_history, do_vectors, marker, receipt, digest
            )
        return {
            "status": "imported",
            "mode": mode,
            "history": do_history,
            "ltm": do_vectors,
            "reembedded": reembedded,
            "receipt": receipt.name,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def recover(store: MemoryStore) -> dict:
    state_dir = store.state_dir
    marker_path = state_dir / MARKER_NAME
    if not marker_path.exists():
        return {"status": "clean"}
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"Import marker is unreadable: {exc}") from exc
    if not isinstance(marker, dict):
        raise RecoveryError("Import marker has invalid structure")

    receipt_name = marker.get("receipt")
    if (
        isinstance(receipt_name, str)
        and Path(receipt_name).name == receipt_name
        and (state_dir / RECEIPTS_NAME / receipt_name).is_file()
    ):
        _finish_transaction(state_dir)
        return {"status": "completed"}

    mode = marker.get("mode")
    try:
        if mode == "overwrite":
            _restore_rollback(store, state_dir / ROLLBACK_NAME)
        elif mode == "append":
            if marker.get("history"):
                _restore_history(store, state_dir / ROLLBACK_NAME)
            append_ids = marker.get("append_ids")
            if not isinstance(append_ids, list) or any(
                not isinstance(record_id, str) for record_id in append_ids
            ):
                raise ValueError("Append marker IDs are invalid")
            store.delete_records(append_ids)
        else:
            raise ValueError(f"Unknown transaction mode: {mode!r}")
    except Exception as exc:
        raise RecoveryError(f"Automatic import recovery failed: {exc}") from exc
    _finish_transaction(state_dir)
    return {"status": "rolled-back"}


def _import_overwrite(
    store: MemoryStore,
    staging: Path,
    history: bool,
    vectors: bool,
    marker: Path,
    receipt: Path,
    digest: str,
) -> None:
    rollback = store.state_dir / ROLLBACK_NAME
    _create_rollback(store, rollback, history, vectors)
    _write_json_atomic(
        marker,
        {
            "mode": "overwrite",
            "history": history,
            "vectors": vectors,
            "receipt": receipt.name,
        },
    )
    try:
        if history:
            store.write_history(
                (staging / "history/history.metta").read_text(encoding="utf-8")
            )
        if vectors:
            store.replace_user_records(iter_records(staging / "vector/records.jsonl"))
        store.smoke_test(history, vectors)
    except Exception as exc:
        try:
            _restore_rollback(store, rollback)
        except Exception as rollback_error:
            raise MemoryImportError(
                f"Import failed and rollback failed: {rollback_error}"
            ) from exc
        _finish_transaction(store.state_dir)
        raise MemoryImportError(f"Overwrite import failed and was rolled back: {exc}") from exc

    _write_receipt(receipt, digest, "overwrite", history, vectors)
    _finish_transaction(store.state_dir)


def _import_append(
    store: MemoryStore,
    staging: Path,
    history: bool,
    vectors: bool,
    marker: Path,
    receipt: Path,
    digest: str,
) -> None:
    rollback = store.state_dir / ROLLBACK_NAME
    _create_rollback(store, rollback, history, False)
    import_id = uuid.uuid4().hex
    marker_value = {
        "mode": "append",
        "history": history,
        "vectors": vectors,
        "append_ids": [],
        "receipt": receipt.name,
    }
    _write_json_atomic(marker, marker_value)
    try:
        if history:
            text = (staging / "history/history.metta").read_text(encoding="utf-8")
            if text:
                current_history = store.read_history() or ""
                separator = "" if not current_history or current_history.endswith("\n") else "\n"
                store.append_history(separator + text)

        if vectors:
            for batch in iter_records(staging / "vector/records.jsonl"):
                appended = []
                for record in batch:
                    new_id = f"import-{import_id}-{record['id']}"
                    appended.append(
                        {
                            **record,
                            "id": new_id,
                            "metadata": {
                                **record.get("metadata", {}),
                                "memory_import_id": import_id,
                            },
                        }
                    )
                intended_ids = marker_value["append_ids"] + [
                    record["id"] for record in appended
                ]
                marker_value["append_ids"] = intended_ids
                _write_json_atomic(marker, marker_value)
                store.upsert_records(appended)
        store.smoke_test(history, vectors)
    except Exception as exc:
        try:
            if history:
                _restore_history(store, rollback)
            store.delete_records(marker_value["append_ids"])
        except Exception as rollback_error:
            raise MemoryImportError(
                f"Append failed and cleanup failed: {rollback_error}"
            ) from exc
        _finish_transaction(store.state_dir)
        raise MemoryImportError(f"Append import failed and was rolled back: {exc}") from exc

    _write_receipt(receipt, digest, "append", history, vectors)
    _finish_transaction(store.state_dir)


def _create_rollback(
    store: MemoryStore, rollback: Path, history: bool, vectors: bool
) -> None:
    shutil.rmtree(rollback, ignore_errors=True)
    rollback.mkdir(parents=True)
    state = {"history": history, "vectors": vectors}
    if history:
        current = store.read_history()
        state["history_present"] = current is not None
        if current is not None:
            (rollback / "history.metta").write_text(current, encoding="utf-8")
    if vectors:
        state["chroma_present"] = store.vector_store_exists()
        with (rollback / "records.jsonl").open("w", encoding="utf-8") as output:
            if state["chroma_present"]:
                for batch in store.iter_user_records(include_embeddings=True):
                    for record in batch:
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
    _write_json_atomic(rollback / "state.json", state)


def _restore_rollback(store: MemoryStore, rollback: Path) -> None:
    state = _read_rollback_state(rollback)
    if state.get("history"):
        _restore_history(store, rollback, state)
    if state.get("vectors"):
        if state["chroma_present"]:
            store.replace_user_records(iter_records(rollback / "records.jsonl"))
        else:
            store.remove_vector_store()
    store.smoke_test(
        bool(state.get("history")),
        bool(state.get("vectors") and state.get("chroma_present")),
    )


def _restore_history(
    store: MemoryStore, rollback: Path, state: dict | None = None
) -> None:
    state = state or _read_rollback_state(rollback)
    if state.get("history") is not True or type(state.get("history_present")) is not bool:
        raise ValueError("Rollback does not contain a valid history state")
    if state["history_present"]:
        store.write_history((rollback / "history.metta").read_text(encoding="utf-8"))
    else:
        store.write_history(None)


def _read_rollback_state(rollback: Path) -> dict:
    try:
        state = json.loads((rollback / "state.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Rollback state is unreadable: {exc}") from exc
    if not isinstance(state, dict):
        raise ValueError("Rollback state is invalid")
    if type(state.get("history")) is not bool or type(state.get("vectors")) is not bool:
        raise ValueError("Rollback component flags are invalid")
    if state["vectors"] and type(state.get("chroma_present")) is not bool:
        raise ValueError("Rollback Chroma presence flag is invalid")
    return state


def _receipt_path(
    state_dir: Path,
    digest: str,
    mode: str,
    history: bool,
    vectors: bool,
) -> Path:
    components = "-".join(
        name for name, enabled in (("history", history), ("ltm", vectors)) if enabled
    )
    return state_dir / RECEIPTS_NAME / f"{digest}-{mode}-{components}.json"


def _write_receipt(
    path: Path, digest: str, mode: str, history: bool, vectors: bool
) -> None:
    _write_json_atomic(
        path,
        {
            "archive_sha256": digest,
            "mode": mode,
            "history": history,
            "ltm": vectors,
            "imported_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _finish_transaction(state_dir: Path) -> None:
    (state_dir / MARKER_NAME).unlink(missing_ok=True)
    shutil.rmtree(state_dir / ROLLBACK_NAME, ignore_errors=True)


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def _validate_filename(filename: str) -> None:
    if (
        "/" in filename
        or "\\" in filename
        or Path(filename).name != filename
        or not filename.endswith(".tar.gz")
    ):
        raise ValueError("filename must be a plain .tar.gz filename")
