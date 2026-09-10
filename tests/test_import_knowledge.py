import importlib
import json
import sys
import types

import chromadb
import numpy
import pytest

from import_knowledge import import_knowledge as ik


ASI_BASE_URL = "https://inference.asicloud.cudos.org/v1"
OPENAI_DEFAULT_REQUEST = {"base_url": None, "api_key": None, "model": "text-embedding-3-large"}


class FakeOpenAI:
    """Replaces the OpenAI SDK client and records which endpoint served each embedding request."""

    requests = []

    def __init__(self, api_key=None, base_url=None):
        self.api_key = api_key
        self.base_url = base_url
        self.embeddings = types.SimpleNamespace(create=self._create)

    def _create(self, *, model, input):
        FakeOpenAI.requests.append({"base_url": self.base_url, "api_key": self.api_key, "model": model})
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2]) for _ in input])


class FailingOpenAI(FakeOpenAI):
    def _create(self, *, model, input):
        raise RuntimeError("Error code: 429 - insufficient_balance")


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts):
        return numpy.full((len(texts), 2), 0.5)


@pytest.fixture(autouse=True)
def fresh_module(monkeypatch):
    importlib.reload(ik)
    monkeypatch.setattr(ik.openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-test-key")
    monkeypatch.setenv("ASI_API_KEY", "asi-test-key")
    FakeOpenAI.requests = []
    yield
    importlib.reload(ik)


def write_knowledge(tmp_path, count):
    path = tmp_path / "knowledge.jsonl"
    lines = [json.dumps({"id": f"k{i}", "document": f"fact {i}", "metadata": {"domain": "test"}}) for i in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_import(monkeypatch, tmp_path, knowledge_files, *args):
    monkeypatch.setattr(ik, "KNOWLEDGE_FILES", [str(path) for path in knowledge_files])
    monkeypatch.setattr(ik, "CURRICULUM_FILE", str(tmp_path / "missing-curriculum.metta"))
    monkeypatch.setattr(ik, "DB_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(sys, "argv", ["import-knowledge", *args])
    ik.main()


def test_switching_to_asicloud_routes_requests_to_asicloud():
    ik.init_embeddings("openai")
    ik.init_embeddings("asicloud")

    ik.embed_batch(["probe"])

    assert FakeOpenAI.requests == [
        {"base_url": ASI_BASE_URL, "api_key": "asi-test-key", "model": "WhereIsAI/UAE-Large-V1"}
    ]


def test_switching_back_to_openai_routes_requests_to_openai():
    ik.init_embeddings("asicloud")
    ik.init_embeddings("openai")

    ik.embed_batch(["probe"])

    assert FakeOpenAI.requests == [OPENAI_DEFAULT_REQUEST]


def test_missing_asi_key_raises_and_keeps_previous_provider(monkeypatch):
    ik.init_embeddings("openai")
    monkeypatch.delenv("ASI_API_KEY")

    with pytest.raises(RuntimeError, match="ASI_API_KEY"):
        ik.init_embeddings("asicloud")

    ik.embed_batch(["probe"])
    assert FakeOpenAI.requests == [OPENAI_DEFAULT_REQUEST]


def test_unknown_provider_raises_and_keeps_local_embeddings(monkeypatch):
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeSentenceTransformer)
    ik.init_embeddings("local")

    with pytest.raises(ValueError):
        ik.init_embeddings("bogus")

    assert ik.embed_batch(["probe"]) == [[0.5, 0.5]]
    assert FakeOpenAI.requests == []


def test_failed_embedding_request_exits_non_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(ik.openai, "OpenAI", FailingOpenAI)

    with pytest.raises(SystemExit) as exit_info:
        run_import(monkeypatch, tmp_path, [write_knowledge(tmp_path, 2)], "--provider", "asicloud")

    assert exit_info.value.code not in (0, None)


def test_missing_knowledge_files_exit_non_zero(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as exit_info:
        run_import(monkeypatch, tmp_path, [tmp_path / "missing.jsonl"])

    assert exit_info.value.code not in (0, None)


def test_successful_import_stores_every_record(monkeypatch, tmp_path):
    run_import(monkeypatch, tmp_path, [write_knowledge(tmp_path, 3)], "--provider", "asicloud")

    collection = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection("memories")
    assert collection.count() == 3
    assert {request["base_url"] for request in FakeOpenAI.requests} == {ASI_BASE_URL}
