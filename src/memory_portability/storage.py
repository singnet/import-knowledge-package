"""Omega memory storage primitives reused by export and import.

This module intentionally follows the same direct, environment-driven approach
as ``import_knowledge.py`` and the existing memory extraction script.  It does not copy raw
ChromaDB files; records are read and written through Chroma's public API.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterable, Iterator
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import chromadb

DEFAULT_COLLECTION = "memories"
DEFAULT_BATCH_SIZE = 500
DEFAULT_OMEGA_ROOT = Path("/PeTTa/repos/Omega")
_KNOWN_DIMENSIONS = {
    "intfloat/e5-large-v2": 1024,
    "text-embedding-3-large": 3072,
}


def is_user_record(metadata: object) -> bool:
    """Exclude knowledge priors, hash sentinels, and other operational rows."""
    return isinstance(metadata, dict) and (
        metadata.get("record_kind") == "user_memory"
        or ("record_kind" not in metadata and "type" not in metadata)
    )


class MemoryStore:
    """Read and mutate Omega's persistent user-memory components."""

    def __init__(
        self,
        memory_dir: Path,
        chroma_path: Path,
        collection_name: str = DEFAULT_COLLECTION,
        embed_batch: Callable[[list[str]], list[list[float]]] | None = None,
        embedding_profile: dict[str, Any] | None = None,
    ) -> None:
        self.memory_dir = Path(memory_dir).resolve()
        self.chroma_path = Path(chroma_path).resolve()
        self.collection_name = collection_name
        self._embed_batch = embed_batch
        self._embedding_profile = embedding_profile
        self._embedding_initialized = False
        self._client = None
        self._collection = None

    @property
    def state_dir(self) -> Path:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        return self.memory_dir

    @property
    def history_path(self) -> Path:
        return self.memory_dir / "history.metta"

    def collection(self):
        if self._collection is None:
            self.chroma_path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=str(self.chroma_path))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
            )
        return self._collection

    def read_history(self) -> str | None:
        if not self.history_path.exists():
            return None
        return self.history_path.read_text(encoding="utf-8")

    def write_history(self, text: str | None) -> None:
        if text is None:
            self.history_path.unlink(missing_ok=True)
            return
        self.state_dir
        temporary = self.history_path.with_suffix(".metta.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, self.history_path)

    def append_history(self, text: str) -> None:
        self.state_dir
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(text)

    def iter_user_records(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        include_embeddings: bool = True,
    ) -> Iterator[list[dict[str, Any]]]:
        collection = self.collection()
        includes = ["documents", "metadatas"]
        if include_embeddings:
            includes.append("embeddings")

        total = collection.count()
        for offset in range(0, total, batch_size):
            result = collection.get(limit=batch_size, offset=offset, include=includes)
            ids = result.get("ids") or []
            documents = result.get("documents") or [None] * len(ids)
            metadatas = result.get("metadatas") or [None] * len(ids)
            embeddings = result.get("embeddings") if include_embeddings else None
            records: list[dict[str, Any]] = []
            for index, record_id in enumerate(ids):
                metadata = metadatas[index] or {}
                if not is_user_record(metadata):
                    continue
                embedding = None
                if embeddings is not None:
                    embedding = [float(value) for value in embeddings[index]]
                records.append(
                    {
                        "id": str(record_id),
                        "document": documents[index],
                        "metadata": dict(metadata),
                        "embedding": embedding,
                    }
                )
            if records:
                yield records

    def user_record_ids(self) -> list[str]:
        result = self.collection().get(include=["metadatas"])
        return [
            str(record_id)
            for record_id, metadata in zip(
                result.get("ids") or [], result.get("metadatas") or []
            )
            if is_user_record(metadata or {})
        ]

    def upsert_records(self, records: Iterable[dict[str, Any]]) -> None:
        batch = list(records)
        if not batch:
            return
        if any(record.get("embedding") is None for record in batch):
            raise ValueError("All imported vector records must contain an embedding")
        self.collection().upsert(
            ids=[record["id"] for record in batch],
            documents=[record["document"] for record in batch],
            metadatas=[
                record.get("metadata") or {"record_kind": "user_memory"}
                for record in batch
            ],
            embeddings=[record["embedding"] for record in batch],
        )

    def replace_user_records(self, batches: Iterable[list[dict[str, Any]]]) -> None:
        existing = self.user_record_ids()
        if existing:
            self.collection().delete(ids=existing)
        for batch in batches:
            self.upsert_records(batch)

    def delete_records(self, record_ids: list[str]) -> None:
        if record_ids:
            self.collection().delete(ids=record_ids)

    def vector_store_exists(self) -> bool:
        return self.chroma_path.exists()

    def remove_vector_store(self) -> None:
        """Remove a vector store created by a failed import into fresh memory."""
        self._collection = None
        self._client = None
        shutil.rmtree(self.chroma_path, ignore_errors=True)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._embed_batch is not None:
            return self._embed_batch(texts)

        # Reuse the package's existing OpenAI/local embedding implementation.
        from import_knowledge.import_knowledge import embed_batch, init_embeddings

        provider = os.environ.get("EMBEDDING_PROVIDER", "Local").casefold()
        if not self._embedding_initialized:
            if provider == "openai":
                model = os.environ.get("OPENAI_EMBEDDING_MODEL")
                init_embeddings(mode="openai", model_name=model)
            else:
                model = os.environ.get("SENTENCE_TRANSFORMERS_MODEL")
                init_embeddings(mode="local", model_name=model)
            self._embedding_initialized = True
        return embed_batch(texts)

    def active_embedding_profile(self) -> dict[str, Any]:
        if self._embedding_profile is not None:
            return dict(self._embedding_profile)
        provider = os.environ.get("EMBEDDING_PROVIDER", "Local")
        if provider.casefold() == "openai":
            model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
        else:
            model = os.environ.get(
                "SENTENCE_TRANSFORMERS_MODEL", "intfloat/e5-large-v2"
            )
        dimension_text = os.environ.get("EMBEDDING_DIMENSION")
        dimension = int(dimension_text) if dimension_text else _KNOWN_DIMENSIONS.get(model)
        return {"provider": provider, "model": model, "vector_dimension": dimension}

    def archive_metadata(self) -> dict[str, str]:
        try:
            package_version = version("import-kb")
        except PackageNotFoundError:
            package_version = "development"
        return {
            "omega_version": os.environ.get("OMEGA_VERSION", "unknown"),
            "import_kb_version": package_version,
            "chromadb_version": chromadb.__version__,
        }

    def smoke_test(self, history: bool, vectors: bool) -> None:
        if history:
            self.read_history()
        if vectors:
            self.collection().get(limit=1, include=["documents"])
