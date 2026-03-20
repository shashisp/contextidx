import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.core.context_unit import ContextUnit
from contextidx.store.sqlite_store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(path=tmp_path / "test.db")
    await s.initialize()
    yield s
    await s.close()


def _unit(
    uid: str = "ctx_test1",
    content: str = "test content",
    scope: dict | None = None,
) -> ContextUnit:
    return ContextUnit(
        id=uid,
        content=content,
        scope=scope or {"user_id": "u1"},
        confidence=0.9,
        source="test",
    )


class TestCRUD:
    async def test_create_and_get(self, store):
        unit = _unit()
        await store.create_unit(unit)
        retrieved = await store.get_unit("ctx_test1")
        assert retrieved is not None
        assert retrieved.content == "test content"
        assert retrieved.scope == {"user_id": "u1"}

    async def test_get_missing(self, store):
        result = await store.get_unit("nonexistent")
        assert result is None

    async def test_update(self, store):
        unit = _unit()
        await store.create_unit(unit)
        await store.update_unit("ctx_test1", {"confidence": 0.5})
        updated = await store.get_unit("ctx_test1")
        assert updated is not None
        assert updated.confidence == 0.5

    async def test_delete(self, store):
        unit = _unit()
        await store.create_unit(unit)
        await store.delete_unit("ctx_test1")
        result = await store.get_unit("ctx_test1")
        assert result is None


class TestScopeQuery:
    async def test_find_in_scope(self, store):
        await store.create_unit(_unit("a", "one", {"user_id": "u1"}))
        await store.create_unit(_unit("b", "two", {"user_id": "u1"}))
        await store.create_unit(_unit("c", "three", {"user_id": "u2"}))

        results = await store.find_units_in_scope({"user_id": "u1"})
        assert len(results) == 2

    async def test_excludes_superseded_by_default(self, store):
        unit = _unit("s1")
        await store.create_unit(unit)
        await store.update_unit("s1", {"superseded_by": "other"})

        results = await store.find_units_in_scope({"user_id": "u1"})
        assert len(results) == 0

    async def test_includes_superseded_when_asked(self, store):
        await store.create_unit(_unit("s1"))
        await store.update_unit("s1", {"superseded_by": "other"})

        results = await store.find_units_in_scope(
            {"user_id": "u1"}, include_superseded=True
        )
        assert len(results) == 1


class TestGraph:
    async def test_add_and_get_edges(self, store):
        await store.create_unit(_unit("a"))
        await store.create_unit(_unit("b"))
        now = datetime.now(timezone.utc)
        await store.add_graph_edge("a", "b", "supersedes", now)
        edges = await store.get_graph_edges("a")
        assert len(edges) == 1
        assert edges[0][0] == "a"
        assert edges[0][1] == "b"
        assert edges[0][2] == "supersedes"


class TestDecayState:
    async def test_upsert_and_get(self, store):
        await store.create_unit(_unit("d1"))
        now = datetime.now(timezone.utc)
        await store.upsert_decay_state("d1", 0.85, now, 0)
        state = await store.get_decay_state("d1")
        assert state is not None
        assert abs(state[0] - 0.85) < 1e-6
        assert state[2] == 0

    async def test_increment_reinforcement(self, store):
        await store.create_unit(_unit("r1"))
        now = datetime.now(timezone.utc)
        await store.upsert_decay_state("r1", 0.5, now, 0)
        count = await store.increment_reinforcement("r1")
        assert count == 1
        count = await store.increment_reinforcement("r1")
        assert count == 2


class TestWAL:
    async def test_append_and_replay(self, store):
        now = datetime.now(timezone.utc)
        seq = await store.append_wal("u1", "store", "both", {"data": 1}, now)
        assert seq is not None

        pending = await store.get_pending_wal()
        assert len(pending) == 1
        assert pending[0]["unit_id"] == "u1"

    async def test_mark_applied(self, store):
        now = datetime.now(timezone.utc)
        seq = await store.append_wal("u1", "store", "both", {}, now)
        await store.mark_wal_applied(seq, now)

        pending = await store.get_pending_wal()
        assert len(pending) == 0

    async def test_mark_failed(self, store):
        now = datetime.now(timezone.utc)
        seq = await store.append_wal("u1", "store", "both", {}, now)
        await store.mark_wal_failed(seq)

        pending = await store.get_pending_wal()
        assert len(pending) == 0


class TestCheckpoints:
    async def test_update_and_get(self, store):
        now = datetime.now(timezone.utc)
        await store.update_checkpoint("vector_backend", now, 42)
        cp = await store.get_checkpoint("vector_backend")
        assert cp is not None
        assert cp[1] == 42

    async def test_get_missing(self, store):
        cp = await store.get_checkpoint("nonexistent")
        assert cp is None
