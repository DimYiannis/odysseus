"""FastEmbed clients must share one ONNX model per process.

RAG, semantic memory, the tool index and the HTTP-lane probe fallback each build
a FastEmbedClient. Each used to load its own TextEmbedding with ONNX Runtime's
CPU arena on, and the main process reached ~1 GB at idle. These tests pin one
load per (model, cache dir), with the arena disabled.
"""

from __future__ import annotations

import sys
import types

import pytest

import src.embeddings as embeddings


@pytest.fixture
def fake_fastembed(monkeypatch, tmp_path):
    loads = []

    class TextEmbedding:
        def __init__(self, **kwargs):
            loads.append(kwargs)

        def embed(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

    module = types.ModuleType("fastembed")
    module.TextEmbedding = TextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)
    monkeypatch.setattr(embeddings, "_shared_models", {})
    monkeypatch.setattr(embeddings, "FASTEMBED_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FASTEMBED_MODEL", raising=False)
    return loads


def test_clients_for_same_model_load_it_once_without_cpu_arena(fake_fastembed):
    first = embeddings.FastEmbedClient()
    second = embeddings.FastEmbedClient()

    assert len(fake_fastembed) == 1
    assert fake_fastembed[0]["enable_cpu_mem_arena"] is False
    assert first.encode(["a"]).shape == (1, 3)
    assert second.encode(["b", "c"]).shape == (2, 3)


def test_distinct_models_get_distinct_loads(fake_fastembed):
    embeddings.FastEmbedClient("model-a")
    embeddings.FastEmbedClient("model-b")
    embeddings.FastEmbedClient("model-a")

    assert [load["model_name"] for load in fake_fastembed] == ["model-a", "model-b"]
