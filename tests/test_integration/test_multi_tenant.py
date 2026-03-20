"""Integration tests: multi-tenant scope isolation."""

from __future__ import annotations

import math

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryHybridBackend, InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


@pytest.fixture
async def multi_tenant_idx(tmp_path):
    backend = InMemoryHybridBackend()
    store = SQLiteStore(path=tmp_path / "tenant.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx
    await ctx.aclose()


@pytest.fixture
async def vector_multi_tenant_idx(tmp_path):
    backend = InMemoryVectorBackend()
    store = SQLiteStore(path=tmp_path / "vtenant.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx
    await ctx.aclose()


class TestMultiTenantIsolation:
    async def test_different_users_isolated(self, multi_tenant_idx):
        """Context stored for user A is not returned for user B."""
        emb = _emb(1.0)
        await multi_tenant_idx.astore(
            content="Alice secret preference",
            scope={"user_id": "alice"},
            embedding=emb,
        )
        await multi_tenant_idx.astore(
            content="Bob secret preference",
            scope={"user_id": "bob"},
            embedding=emb,
        )

        alice_results = await multi_tenant_idx.aretrieve(
            query="preference",
            scope={"user_id": "alice"},
            query_embedding=emb,
        )
        bob_results = await multi_tenant_idx.aretrieve(
            query="preference",
            scope={"user_id": "bob"},
            query_embedding=emb,
        )

        assert all(r.scope["user_id"] == "alice" for r in alice_results)
        assert all(r.scope["user_id"] == "bob" for r in bob_results)

    async def test_scope_isolation_with_sessions(self, multi_tenant_idx):
        """Different sessions within the same user are isolated."""
        emb = _emb(2.0)
        await multi_tenant_idx.astore(
            content="session 1 data",
            scope={"user_id": "u1", "session_id": "s1"},
            embedding=emb,
        )
        await multi_tenant_idx.astore(
            content="session 2 data",
            scope={"user_id": "u1", "session_id": "s2"},
            embedding=emb,
        )

        s1_results = await multi_tenant_idx.aretrieve(
            query="data",
            scope={"user_id": "u1", "session_id": "s1"},
            query_embedding=emb,
        )
        s2_results = await multi_tenant_idx.aretrieve(
            query="data",
            scope={"user_id": "u1", "session_id": "s2"},
            query_embedding=emb,
        )

        s1_contents = {r.content for r in s1_results}
        s2_contents = {r.content for r in s2_results}

        assert "session 1 data" in s1_contents
        assert "session 2 data" in s2_contents

    async def test_empty_scope_query_returns_nothing_for_scoped_data(
        self, multi_tenant_idx
    ):
        """Querying with a different scope returns no results from another scope."""
        emb = _emb(3.0)
        await multi_tenant_idx.astore(
            content="scoped fact",
            scope={"user_id": "u99"},
            embedding=emb,
        )
        results = await multi_tenant_idx.aretrieve(
            query="scoped",
            scope={"user_id": "u_other"},
            query_embedding=emb,
        )
        assert len(results) == 0

    async def test_vector_only_scope_isolation(self, vector_multi_tenant_idx):
        """Scope isolation also works with vector-only backends."""
        emb = _emb(4.0)
        await vector_multi_tenant_idx.astore(
            content="user1 context",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        await vector_multi_tenant_idx.astore(
            content="user2 context",
            scope={"user_id": "u2"},
            embedding=emb,
        )

        r1 = await vector_multi_tenant_idx.aretrieve(
            query="context",
            scope={"user_id": "u1"},
            query_embedding=emb,
        )
        assert all(r.scope["user_id"] == "u1" for r in r1)

    async def test_multiple_users_store_and_retrieve(self, multi_tenant_idx):
        """Multiple users can store and retrieve independently."""
        for i in range(5):
            await multi_tenant_idx.astore(
                content=f"fact for user {i}",
                scope={"user_id": f"user_{i}"},
                embedding=_emb(float(i)),
            )

        for i in range(5):
            results = await multi_tenant_idx.aretrieve(
                query=f"fact for user {i}",
                scope={"user_id": f"user_{i}"},
                query_embedding=_emb(float(i)),
            )
            assert len(results) >= 1
            assert results[0].scope["user_id"] == f"user_{i}"
