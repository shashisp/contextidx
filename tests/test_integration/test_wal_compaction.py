"""Integration tests for WAL compaction."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from contextidx.store.sqlite_store import SQLiteStore
from contextidx.utils.wal import WAL


class TestWalCompaction:
    @pytest.fixture
    async def store(self, tmp_path):
        store = SQLiteStore(path=tmp_path / "wal_test.db")
        await store.initialize()
        yield store
        await store.close()

    @pytest.fixture
    def wal(self, store):
        return WAL(store)

    async def test_compact_removes_applied_entries(self, store, wal):
        """Applied entries older than retention should be removed."""
        seq1 = await wal.append("u1", "store", "both", {"id": "u1"})
        seq2 = await wal.append("u2", "store", "both", {"id": "u2"})

        old_time = datetime.now(timezone.utc) - timedelta(hours=48)
        await store.mark_wal_applied(seq1, old_time)
        await store.mark_wal_applied(seq2, old_time)

        removed = await wal.compact(retention_hours=24)
        assert removed == 2

        pending = await wal.replay_pending()
        assert len(pending) == 0

    async def test_compact_preserves_pending_entries(self, store, wal):
        """Pending entries should not be removed by compaction."""
        await wal.append("u1", "store", "both", {"id": "u1"})
        await wal.append("u2", "store", "both", {"id": "u2"})

        removed = await wal.compact(retention_hours=24)
        assert removed == 0

        pending = await wal.replay_pending()
        assert len(pending) == 2

    async def test_compact_preserves_recent_applied(self, store, wal):
        """Recently applied entries within retention should be preserved."""
        seq = await wal.append("u1", "store", "both", {"id": "u1"})
        now = datetime.now(timezone.utc)
        await store.mark_wal_applied(seq, now)

        removed = await wal.compact(retention_hours=24)
        assert removed == 0

    async def test_compact_mixed_entries(self, store, wal):
        """Only old applied entries are removed; pending and recent stay."""
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        recent = datetime.now(timezone.utc) - timedelta(hours=1)

        seq1 = await wal.append("u1", "store", "both", {"id": "u1"})
        seq2 = await wal.append("u2", "store", "both", {"id": "u2"})
        seq3 = await wal.append("u3", "store", "both", {"id": "u3"})

        await store.mark_wal_applied(seq1, old)
        await store.mark_wal_applied(seq2, recent)
        # seq3 stays pending

        removed = await wal.compact(retention_hours=24)
        assert removed == 1  # only seq1

    async def test_compact_wal_store_method(self, store):
        """Test compact_wal directly on the store."""
        now = datetime.now(timezone.utc)
        old = now - timedelta(hours=48)

        seq = await store.append_wal("u1", "store", "both", {"id": "u1"}, now)
        await store.mark_wal_applied(seq, old)

        removed = await store.compact_wal(before=now - timedelta(hours=24))
        assert removed == 1
