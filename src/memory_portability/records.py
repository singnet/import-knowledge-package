"""Streaming JSONL record helpers and embedding compatibility handling."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from .errors import ArchiveValidationError
from .storage import DEFAULT_BATCH_SIZE, MemoryStore


def iter_records(path: Path, batch_size: int = DEFAULT_BATCH_SIZE) -> Iterator[list[dict]]:
    batch: list[dict] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ArchiveValidationError(
                    f"Invalid JSONL record on line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ArchiveValidationError(
                    f"JSONL record on line {line_number} is not an object"
                )
            batch.append(record)
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def source_embedding_profile(staging: Path) -> dict:
    value = json.loads((staging / "vector/collections.json").read_text(encoding="utf-8"))
    return value["collections"][0]["embedding_profile"]


def profiles_compatible(source: dict, active: dict, missing_embeddings: bool) -> bool:
    if missing_embeddings:
        return False
    for key in ("provider", "model"):
        source_value = source.get(key)
        active_value = active.get(key)
        if not isinstance(source_value, str) or not isinstance(active_value, str):
            return False
        if source_value.casefold() != active_value.casefold():
            return False
    source_dimension = source.get("vector_dimension")
    active_dimension = active.get("vector_dimension")
    if (
        type(source_dimension) is not int
        or source_dimension <= 0
        or type(active_dimension) is not int
        or active_dimension <= 0
    ):
        return False
    return source_dimension == active_dimension


def reembed_records(staging: Path, store: MemoryStore) -> None:
    path = staging / "vector/records.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    active = store.active_embedding_profile()
    expected_dimension = active.get("vector_dimension")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            for batch in iter_records(path):
                documents = [record["document"] for record in batch]
                embeddings = store.embed(documents)
                if len(embeddings) != len(batch):
                    raise ValueError(
                        "Embedding provider returned a different number of vectors"
                    )
                for record, embedding in zip(batch, embeddings):
                    if not isinstance(embedding, list) or not embedding:
                        raise ValueError("Embedding provider returned an invalid vector")
                    if expected_dimension is not None and len(embedding) != expected_dimension:
                        raise ValueError(
                            f"Embedding dimension {len(embedding)} does not match "
                            f"active profile dimension {expected_dimension}"
                        )
                    record["embedding"] = [float(value) for value in embedding]
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
