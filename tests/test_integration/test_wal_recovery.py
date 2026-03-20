"""Integration test: WAL crash recovery and pending buffer read-after-write."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    import math
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


class TestWALRecovery:
    async def test_replay_pending_wal_on_init(self, tmp_path):
        """Simulate a crash: create WAL entry but skip actual store write.
        On re-init, the WAL should be replayed."""
        db_path = tmp_path / "wal_recovery.db"
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=db_path)
        await store.initialize()

        unit = ContextUnit(
            id=generate_unit_id(),
            content="Should survive crash",
            scope={"user_id": "u1"},
            confidence=0.9,
            source="test",
            embedding=_emb(1.0),
        )
        now = datetime.now(timezone.utc)
        await store.append_wal(
            unit_id=unit.id,
            operation="store",
            store_target="both",
            payload=unit.model_dump(mode="json"),
            written_at=now,
        )
        pending = await store.get_pending_wal()
        assert len(pending) == 1

        await store.close()

        # Re-init: WAL should replay
        ctx = ContextIdx(
            backend=backend,
            internal_store=SQLiteStore(path=db_path),
        )
        await ctx.ainitialize()

        recovered = await ctx._store.get_unit(unit.id)
        assert recovered is not None
        assert recovered.content == "Should survive crash"

        pending = await ctx._store.get_pending_wal()
        assert len(pending) == 0

        await ctx.aclose()


class TestPendingBufferConsistency:
    async def test_read_after_write_immediate(self, tmp_path):
        """A write followed by an immediate read should find the unit."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "raw.db")
        ctx = ContextIdx(
            backend=backend,
            internal_store=store,
        )
        await ctx.ainitialize()

        emb = _emb(2.0)
        uid = await ctx.astore(
            content="immediately readable",
            scope={"user_id": "u1"},
            embedding=emb,
        )

        results = await ctx.aretrieve(
            query="immediately",
            scope={"user_id": "u1"},
            query_embedding=emb,
        )
        assert any(r.id == uid for r in results)

        await ctx.aclose()
