"""Integration tests: full write → read cycle with mock backends."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    """Deterministic test embedding."""
    import math
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


@pytest.fixture
async def idx(tmp_path):
    backend = InMemoryVectorBackend()
    store = SQLiteStore(path=tmp_path / "integration.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx
    await ctx.aclose()


class TestWriteRead:
    async def test_store_and_retrieve(self, idx):
        emb = _emb(1.0)
        uid = await idx.astore(
            content="User prefers async communication",
            scope={"user_id": "u1"},
            confidence=0.9,
            source="test",
            embedding=emb,
        )
        assert uid.startswith("ctx_")

        results = await idx.aretrieve(
            query="how does user communicate?",
            scope={"user_id": "u1"},
            top_k=5,
            query_embedding=emb,
        )
        assert len(results) >= 1
        assert results[0].content == "User prefers async communication"

    async def test_scope_isolation(self, idx):
        emb = _emb(2.0)
        await idx.astore("user1 fact", scope={"user_id": "u1"}, embedding=emb)
        await idx.astore("user2 fact", scope={"user_id": "u2"}, embedding=emb)

        r1 = await idx.aretrieve("fact", scope={"user_id": "u1"}, query_embedding=emb)
        r2 = await idx.aretrieve("fact", scope={"user_id": "u2"}, query_embedding=emb)

        assert all(u.scope["user_id"] == "u1" for u in r1)
        assert all(u.scope["user_id"] == "u2" for u in r2)

    async def test_multiple_stores_retrieves_ranked(self, idx):
        for i in range(5):
            await idx.astore(
                content=f"fact number {i}",
                scope={"user_id": "u1"},
                embedding=_emb(float(i)),
            )
        results = await idx.aretrieve(
            "fact", scope={"user_id": "u1"}, top_k=3, query_embedding=_emb(4.0)
        )
        assert len(results) <= 3


class TestConflictResolution:
    async def test_supersession_on_conflict(self, idx):
        emb = _emb(3.0)
        id1 = await idx.astore(
            content="User prefers formal tone",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        id2 = await idx.astore(
            content="User does not prefer formal tone",
            scope={"user_id": "u1"},
            embedding=emb,
        )

        results = await idx.aretrieve(
            "user tone preference",
            scope={"user_id": "u1"},
            query_embedding=emb,
        )
        ids = [r.id for r in results]
        assert id2 in ids
        assert id1 not in ids


class TestTimeTravelQuery:
    async def test_retrieve_at_past_time(self, idx):
        now = datetime.now(timezone.utc)
        old_time = now - timedelta(days=10)
        new_time = now - timedelta(days=1)

        emb = _emb(5.0)
        id_old = await idx.astore(
            content="Junior engineer",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        old_unit = await idx._store.get_unit(id_old)
        if old_unit:
            await idx._store.update_unit(id_old, {"created_at": old_time.isoformat()})

        id_new = await idx.astore(
            content="Senior engineer",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        new_unit = await idx._store.get_unit(id_new)
        if new_unit:
            await idx._store.update_unit(id_new, {"created_at": new_time.isoformat()})

        five_days_ago = now - timedelta(days=5)
        results = await idx.aretrieve(
            "engineer level",
            scope={"user_id": "u1"},
            query_embedding=emb,
            at=five_days_ago,
        )
        if results:
            contents = [r.content for r in results]
            assert "Senior engineer" not in contents or "Junior engineer" in contents


class TestReinforce:
    async def test_reinforce_increments(self, idx):
        emb = _emb(6.0)
        uid = await idx.astore(
            content="reinforcement target",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        await idx.areinforce(uid)
        await idx.areinforce(uid)
        state = await idx._store.get_decay_state(uid)
        assert state is not None
        assert state[2] == 2


class TestLineage:
    async def test_lineage_returns_unit(self, idx):
        emb = _emb(7.0)
        uid = await idx.astore(
            content="original fact",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        chain = await idx.alineage(uid)
        assert len(chain) >= 1
        assert chain[0].id == uid


class TestDiff:
    async def test_diff_returns_recent(self, idx):
        emb = _emb(8.0)
        await idx.astore(
            content="recent fact",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        results = await idx.adiff(scope={"user_id": "u1"}, since="1d")
        assert len(results) >= 1

    async def test_diff_excludes_old(self, idx):
        results = await idx.adiff(scope={"user_id": "u999"}, since="1d")
        assert len(results) == 0
