"""Integration tests: auto-routing with PostgresStore and Redis pending buffer."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend


def _asyncpg_available() -> bool:
    try:
        import asyncpg
        return True
    except ImportError:
        return False


def _redis_available() -> bool:
    try:
        import redis.asyncio
        return True
    except ImportError:
        return False


_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class TestPostgresStoreRouting:
    """Test the store type auto-routing logic."""

    async def test_default_uses_sqlite(self, tmp_path):
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "test.db"),
            openai_api_key="test",
        )
        assert isinstance(ctx._store, SQLiteStore)

    @pytest.mark.skipif(not _asyncpg_available(), reason="asyncpg not installed")
    async def test_dsn_routes_to_postgres(self, tmp_path):
        from contextidx.store.postgres_store import PostgresStore
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_dsn="postgresql://localhost/test",
            openai_api_key="test",
        )
        assert isinstance(ctx._store, PostgresStore)

    @pytest.mark.skipif(not _asyncpg_available(), reason="asyncpg not installed")
    async def test_explicit_postgres_type(self, tmp_path):
        from contextidx.store.postgres_store import PostgresStore
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_type="postgres",
            internal_store_dsn="postgresql://localhost/test",
            openai_api_key="test",
        )
        assert isinstance(ctx._store, PostgresStore)

    async def test_explicit_store_overrides_dsn(self, tmp_path):
        backend = InMemoryVectorBackend()
        explicit_store = SQLiteStore(path=tmp_path / "explicit.db")
        ctx = ContextIdx(
            backend=backend,
            internal_store=explicit_store,
            internal_store_dsn="postgresql://localhost/test",
            openai_api_key="test",
        )
        assert ctx._store is explicit_store

    async def test_sqlite_type_explicit(self, tmp_path):
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_type="sqlite",
            internal_store_path=str(tmp_path / "sqlite.db"),
            openai_api_key="test",
        )
        assert isinstance(ctx._store, SQLiteStore)


class TestRedisBufferRouting:
    """Test the pending buffer type routing logic."""

    async def test_default_uses_memory(self):
        from contextidx.utils.pending_buffer import PendingBuffer
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(backend=backend, openai_api_key="test")
        assert isinstance(ctx._pending, PendingBuffer)

    @pytest.mark.skipif(not _redis_available(), reason="redis not installed")
    async def test_redis_type_creates_redis_buffer(self):
        from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            pending_buffer_type="redis",
            redis_url="redis://localhost:6379/0",
            openai_api_key="test",
        )
        assert isinstance(ctx._pending, RedisPendingBuffer)


class TestNewInitParams:
    """Test that the new init parameters are accepted without errors."""

    async def test_all_new_params(self, tmp_path):
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_type="sqlite",
            internal_store_path=str(tmp_path / "test.db"),
            pending_buffer_type="memory",
            openai_api_key="test",
        )
        assert isinstance(ctx._store, SQLiteStore)

    async def test_full_lifecycle_with_sqlite(self, tmp_path):
        backend = InMemoryVectorBackend()
        ctx = ContextIdx(
            backend=backend,
            internal_store_path=str(tmp_path / "lifecycle.db"),
            openai_api_key="test",
        )
        await ctx.ainitialize()
        try:
            uid = await ctx.astore(
                content="routing test",
                scope={"user_id": "u1"},
                embedding=_emb(1.0),
            )
            assert uid.startswith("ctx_")

            results = await ctx.aretrieve(
                query="routing",
                scope={"user_id": "u1"},
                query_embedding=_emb(1.0),
            )
            assert len(results) >= 1
        finally:
            await ctx.aclose()
