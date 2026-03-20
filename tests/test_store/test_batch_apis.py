"""Tests for batch store APIs (get_units_batch, get_decay_states_batch)."""

from datetime import datetime, timezone

import pytest

from contextidx.core.context_unit import ContextUnit
from contextidx.store.sqlite_store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(path=tmp_path / "batch_test.db")
    await s.initialize()
    yield s
    await s.close()


def _unit(uid: str, content: str = "test") -> ContextUnit:
    return ContextUnit(
        id=uid,
        content=content,
        scope={"user_id": "u1"},
        confidence=0.9,
        source="test",
    )


class TestGetUnitsBatch:
    async def test_returns_all_existing(self, store):
        for i in range(5):
            await store.create_unit(_unit(f"u{i}", f"content {i}"))

        result = await store.get_units_batch(["u0", "u1", "u2", "u3", "u4"])
        assert len(result) == 5
        assert result["u2"].content == "content 2"

    async def test_skips_missing_ids(self, store):
        await store.create_unit(_unit("u0"))
        await store.create_unit(_unit("u1"))

        result = await store.get_units_batch(["u0", "u1", "missing"])
        assert len(result) == 2
        assert "missing" not in result

    async def test_empty_ids(self, store):
        result = await store.get_units_batch([])
        assert result == {}

    async def test_all_missing(self, store):
        result = await store.get_units_batch(["x", "y", "z"])
        assert result == {}

    async def test_single_id(self, store):
        await store.create_unit(_unit("solo"))
        result = await store.get_units_batch(["solo"])
        assert len(result) == 1
        assert result["solo"].id == "solo"


class TestGetDecayStatesBatch:
    async def test_returns_all_existing(self, store):
        now = datetime.now(timezone.utc)
        for i in range(3):
            await store.create_unit(_unit(f"d{i}"))
            await store.upsert_decay_state(f"d{i}", 0.9 - i * 0.1, now, i)

        result = await store.get_decay_states_batch(["d0", "d1", "d2"])
        assert len(result) == 3
        assert abs(result["d0"][0] - 0.9) < 1e-6
        assert result["d2"][2] == 2

    async def test_skips_missing(self, store):
        now = datetime.now(timezone.utc)
        await store.create_unit(_unit("d0"))
        await store.upsert_decay_state("d0", 0.5, now, 0)

        result = await store.get_decay_states_batch(["d0", "no_state"])
        assert len(result) == 1
        assert "no_state" not in result

    async def test_empty_ids(self, store):
        result = await store.get_decay_states_batch([])
        assert result == {}
