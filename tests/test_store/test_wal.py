from datetime import datetime, timezone

import pytest

from contextidx.store.sqlite_store import SQLiteStore
from contextidx.utils.wal import WAL


@pytest.fixture
async def wal_store(tmp_path):
    s = SQLiteStore(path=tmp_path / "wal_test.db")
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def wal(wal_store):
    return WAL(wal_store)


class TestWAL:
    async def test_append_returns_seq(self, wal):
        seq = await wal.append("u1", "store", "both", {"k": "v"})
        assert isinstance(seq, int)

    async def test_replay_pending(self, wal):
        await wal.append("u1", "store", "both", {"a": 1})
        await wal.append("u2", "store", "both", {"b": 2})
        entries = await wal.replay_pending()
        assert len(entries) == 2
        assert entries[0].unit_id == "u1"
        assert entries[1].unit_id == "u2"

    async def test_mark_applied_removes_from_pending(self, wal):
        seq = await wal.append("u1", "store", "both", {})
        await wal.mark_applied(seq)
        entries = await wal.replay_pending()
        assert len(entries) == 0

    async def test_mark_failed_removes_from_pending(self, wal):
        seq = await wal.append("u1", "store", "both", {})
        await wal.mark_failed(seq)
        entries = await wal.replay_pending()
        assert len(entries) == 0

    async def test_replay_ordering(self, wal):
        s1 = await wal.append("first", "store", "both", {})
        s2 = await wal.append("second", "store", "both", {})
        entries = await wal.replay_pending()
        assert entries[0].seq < entries[1].seq
