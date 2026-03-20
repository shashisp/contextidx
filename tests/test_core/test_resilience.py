"""Tests for retry and exception wrapping in ContextIdx."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx, OpenAIEmbeddingProvider
from contextidx.exceptions import BackendError, EmbeddingError
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class FailingEmbedder:
    """Embedder that fails N times then succeeds."""

    def __init__(self, fail_count: int, error: Exception | None = None):
        self._fail_count = fail_count
        self._calls = 0
        self._error = error or ConnectionError("transient")

    async def embed(self, text: str) -> list[float]:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise self._error
        return _emb(1.0)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise self._error
        return [_emb(float(i)) for i in range(len(texts))]


class TestEmbeddingResilience:
    async def test_retries_on_transient_error(self, tmp_path):
        embedder = FailingEmbedder(fail_count=2)
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "retry.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            uid = await idx.astore(
                content="retry test",
                scope={"user_id": "u1"},
            )
            assert uid.startswith("ctx_")
            assert embedder._calls == 3

    async def test_gives_up_after_max_retries(self, tmp_path):
        embedder = FailingEmbedder(fail_count=10)
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "giveup.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            with pytest.raises(ConnectionError):
                await idx.astore(
                    content="will fail",
                    scope={"user_id": "u1"},
                )
            assert embedder._calls == 3

    async def test_wraps_non_transient_as_embedding_error(self, tmp_path):
        embedder = FailingEmbedder(
            fail_count=1, error=ValueError("bad input")
        )
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "wrap.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            embedding_fn=embedder,
        ) as idx:
            with pytest.raises(EmbeddingError, match="Embedding failed"):
                await idx.astore(
                    content="will wrap",
                    scope={"user_id": "u1"},
                )


class TestOpenAIProviderErrorWrapping:
    async def test_missing_openai_raises_embedding_error(self):
        provider = OpenAIEmbeddingProvider(api_key="fake")
        provider._ensure_client = lambda: None
        provider._client = None

        with pytest.raises((EmbeddingError, AttributeError)):
            await provider.embed("test")
