"""Tests for the pluggable EmbeddingFunction protocol and custom embedders."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx, OpenAIEmbeddingProvider
from contextidx.core.embedding import EmbeddingFunction
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class FakeEmbedder:
    """Custom embedding function for testing."""

    def __init__(self, dimension: int = _DIM):
        self._dim = dimension
        self.embed_calls: list[str] = []
        self.batch_calls: list[list[str]] = []

    async def embed(self, text: str) -> list[float]:
        self.embed_calls.append(text)
        seed = float(hash(text) % 1000) / 1000
        return [math.sin(seed * (i + 1)) for i in range(self._dim)]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(texts)
        return [await self.embed(t) for t in texts]


class TestEmbeddingProtocol:
    def test_fake_embedder_satisfies_protocol(self):
        embedder = FakeEmbedder()
        assert isinstance(embedder, EmbeddingFunction)

    def test_openai_provider_satisfies_protocol(self):
        provider = OpenAIEmbeddingProvider(api_key="fake")
        assert isinstance(provider, EmbeddingFunction)


class TestCustomEmbeddingInContextIdx:
    async def test_store_uses_custom_embedder(self, tmp_path):
        embedder = FakeEmbedder()
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "custom_emb.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            uid = await idx.astore(
                content="test with custom embedder",
                scope={"user_id": "u1"},
            )
            assert uid.startswith("ctx_")
            assert len(embedder.embed_calls) == 1
            assert embedder.embed_calls[0] == "test with custom embedder"

    async def test_retrieve_uses_custom_embedder(self, tmp_path):
        embedder = FakeEmbedder()
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "custom_ret.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            await idx.astore(
                content="important fact",
                scope={"user_id": "u1"},
            )
            results = await idx.aretrieve(
                "what is important?",
                scope={"user_id": "u1"},
            )
            assert len(embedder.embed_calls) == 2
            assert embedder.embed_calls[1] == "what is important?"
            assert len(results) >= 1

    async def test_batch_store_uses_custom_embedder(self, tmp_path):
        embedder = FakeEmbedder()
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "custom_batch.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            ids = await idx.astore_batch([
                {"content": "item 1", "scope": {"user_id": "u1"}},
                {"content": "item 2", "scope": {"user_id": "u1"}},
            ])
            assert len(ids) == 2
            assert len(embedder.batch_calls) == 1
            assert embedder.batch_calls[0] == ["item 1", "item 2"]

    async def test_precomputed_embedding_skips_embedder(self, tmp_path):
        embedder = FakeEmbedder()
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "skip_emb.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            await idx.astore(
                content="with precomputed",
                scope={"user_id": "u1"},
                embedding=_emb(1.0),
            )
            assert len(embedder.embed_calls) == 0

            await idx.aretrieve(
                "query",
                scope={"user_id": "u1"},
                query_embedding=_emb(1.0),
            )
            assert len(embedder.embed_calls) == 0
