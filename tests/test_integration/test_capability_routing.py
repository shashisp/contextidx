"""Integration tests: capability flag auto-routing of internal store."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.backend_metadata_store import BackendMetadataStore
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryHybridBackend, InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class TestAutoRouting:
    async def test_vector_only_backend_uses_sqlite(self, tmp_path):
        """When backend.supports_metadata_store is False, SQLiteStore is used."""
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "auto.db"),
            openai_api_key="test",
        )
        assert isinstance(ctx._store, SQLiteStore)

    async def test_metadata_backend_uses_backend_store(self, tmp_path):
        """When backend.supports_metadata_store is True, BackendMetadataStore is used."""
        backend = InMemoryHybridBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "auto_graph.db"),
            openai_api_key="test",
        )
        assert isinstance(ctx._store, BackendMetadataStore)

    async def test_explicit_store_overrides_auto_routing(self, tmp_path):
        """An explicitly provided internal_store takes precedence."""
        backend = InMemoryHybridBackend()
        explicit_store = SQLiteStore(path=tmp_path / "explicit.db")
        ctx = ContextIdx(
            backend=backend,
            internal_store=explicit_store,
            openai_api_key="test",
        )
        assert ctx._store is explicit_store

    async def test_backend_metadata_store_write_read(self, tmp_path):
        """Full write/read cycle works through BackendMetadataStore."""
        backend = InMemoryHybridBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "routing.db"),
            openai_api_key="test",
        )
        await ctx.ainitialize()
        try:
            emb = _emb(1.0)
            uid = await ctx.astore(
                content="auto-routed context",
                scope={"user_id": "u1"},
                embedding=emb,
            )
            assert uid.startswith("ctx_")

            results = await ctx.aretrieve(
                query="auto routed",
                scope={"user_id": "u1"},
                query_embedding=emb,
            )
            assert len(results) >= 1
            assert results[0].content == "auto-routed context"
        finally:
            await ctx.aclose()

    async def test_backend_metadata_store_graph_works(self, tmp_path):
        """Graph operations still work with BackendMetadataStore."""
        backend = InMemoryHybridBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "graph.db"),
            openai_api_key="test",
        )
        await ctx.ainitialize()
        try:
            emb = _emb(2.0)
            id1 = await ctx.astore(
                content="version one",
                scope={"user_id": "u1"},
                embedding=emb,
            )
            id2 = await ctx.astore(
                content="version one updated",
                scope={"user_id": "u1"},
                embedding=emb,
            )

            lineage = await ctx.alineage(id2)
            assert len(lineage) >= 1
        finally:
            await ctx.aclose()
