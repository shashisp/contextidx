"""Tests for ContextIdx async context manager (__aenter__/__aexit__)."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class TestAsyncContextManager:
    async def test_async_with_initializes_and_closes(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "ctx_mgr.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            openai_api_key="test",
        ) as idx:
            assert idx._initialized is True
            uid = await idx.astore(
                content="context manager test",
                scope={"user_id": "u1"},
                embedding=_emb(1.0),
            )
            assert uid.startswith("ctx_")

            results = await idx.aretrieve(
                "test", scope={"user_id": "u1"}, query_embedding=_emb(1.0)
            )
            assert len(results) >= 1

        assert idx._initialized is False

    async def test_async_with_closes_on_exception(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "ctx_err.db")

        with pytest.raises(RuntimeError, match="deliberate"):
            async with ContextIdx(
                backend=backend,
                internal_store=store,
                openai_api_key="test",
            ) as idx:
                await idx.astore(
                    content="before error",
                    scope={"user_id": "u1"},
                    embedding=_emb(2.0),
                )
                raise RuntimeError("deliberate")

        assert idx._initialized is False

    async def test_explicit_lifecycle_still_works(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "explicit.db")
        idx = ContextIdx(
            backend=backend,
            internal_store=store,
            openai_api_key="test",
        )
        await idx.ainitialize()
        assert idx._initialized is True
        await idx.aclose()
        assert idx._initialized is False
