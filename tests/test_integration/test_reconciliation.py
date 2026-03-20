"""Integration tests: checkpoint reconciliation."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


@pytest.fixture
async def reconcile_setup(tmp_path):
    """Create a ContextIdx and store a few units, then yield components."""
    backend = InMemoryVectorBackend()
    store = SQLiteStore(path=tmp_path / "reconcile.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx, backend, store
    await ctx.aclose()


class TestReconciliation:
    async def test_reconcile_no_drift(self, reconcile_setup):
        """When stores are in sync, reconciliation re-inserts nothing."""
        ctx, backend, store = reconcile_setup
        emb = _emb(1.0)
        await ctx.astore(
            content="synced fact",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        stats = await ctx.areconcile()
        assert stats["checked"] >= 1
        assert stats["reinserted"] == 0
        assert stats["errors"] == 0

    async def test_reconcile_detects_missing_vector(self, reconcile_setup):
        """If a vector is missing from the backend, reconciliation re-inserts it."""
        ctx, backend, store = reconcile_setup
        emb = _emb(2.0)
        uid = await ctx.astore(
            content="will be orphaned",
            scope={"user_id": "u1"},
            embedding=emb,
        )

        # Simulate drift: remove from backend only
        await backend.delete(uid)
        assert uid not in backend._vectors

        stats = await ctx.areconcile()
        assert stats["reinserted"] >= 1
        assert uid in backend._vectors

    async def test_reconcile_updates_checkpoint(self, reconcile_setup):
        """After reconciliation, a checkpoint is recorded."""
        ctx, backend, store = reconcile_setup
        emb = _emb(3.0)
        await ctx.astore(
            content="checkpoint test",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        await ctx.areconcile()

        checkpoint = await store.get_checkpoint("vector_backend")
        assert checkpoint is not None
        last_synced, units_synced = checkpoint
        assert units_synced >= 1

    async def test_reconcile_idempotent(self, reconcile_setup):
        """Running reconciliation twice should not duplicate inserts."""
        ctx, backend, store = reconcile_setup
        emb = _emb(4.0)
        uid = await ctx.astore(
            content="idempotent check",
            scope={"user_id": "u1"},
            embedding=emb,
        )

        await backend.delete(uid)
        stats1 = await ctx.areconcile()
        assert stats1["reinserted"] >= 1

        # Second run should find everything in sync (since checkpoint advanced)
        stats2 = await ctx.areconcile()
        assert stats2["reinserted"] == 0


class TestFindActiveWithoutVector:
    async def test_find_active_without_vector(self, tmp_path):
        store = SQLiteStore(path=tmp_path / "fawv.db")
        await store.initialize()
        try:
            from contextidx.core.context_unit import ContextUnit

            unit = ContextUnit(
                id="test_unit_1",
                content="test",
                scope={"user_id": "u1"},
            )
            await store.create_unit(unit)
            now = datetime.now(timezone.utc)

            since = datetime.min.replace(tzinfo=timezone.utc)
            ids = await store.find_active_without_vector(since)
            assert "test_unit_1" in ids
        finally:
            await store.close()
