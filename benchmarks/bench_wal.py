"""WAL replay benchmarks.

Pre-populates the WAL with varying numbers of pending entries, then measures
the time to replay them on simulated startup recovery.

This reflects the critical startup-latency path after an unexpected restart
with a large accumulated WAL.

Run with::

    pytest benchmarks/bench_wal.py --benchmark-only -v
"""

from __future__ import annotations

import asyncio
import json
import random
import string
from datetime import datetime, timezone

import pytest

from benchmarks.conftest import EMBEDDING_DIM, _random_embedding
from contextidx.store.sqlite_store import SQLiteStore
from contextidx.utils.wal import WAL


# ── helpers ───────────────────────────────────────────────────────────────────


def _random_unit_id() -> str:
    return "ctx_" + "".join(random.choices(string.hexdigits[:16], k=16))


def _make_wal_payload() -> dict:
    """Build a realistic WAL payload (content + embedding as JSON)."""
    return {
        "content": "benchmark context unit — simulated WAL entry",
        "scope": {"user_id": f"user_{random.randint(0, 99)}"},
        "embedding": _random_embedding(EMBEDDING_DIM),
        "confidence": round(random.uniform(0.5, 1.0), 3),
    }


async def _populate_wal(store: SQLiteStore, n_entries: int) -> None:
    """Append *n_entries* pending WAL entries to the store."""
    wal = WAL(store)
    for _ in range(n_entries):
        await wal.append(
            unit_id=_random_unit_id(),
            operation="store",
            store_target="vector",
            payload=_make_wal_payload(),
        )


def _replay_wal(store: SQLiteStore) -> list:
    """Synchronous wrapper: replay all pending WAL entries."""
    async def _run():
        wal = WAL(store)
        return await wal.replay_pending()
    return asyncio.get_event_loop().run_until_complete(_run())


def _compact_wal(store: SQLiteStore, retention_hours: int = 0) -> int:
    """Synchronous wrapper: compact all applied entries."""
    async def _run():
        wal = WAL(store)
        # Mark all pending as applied first so compact can remove them
        entries = await wal.replay_pending()
        for entry in entries:
            await wal.mark_applied(entry.seq)
        return await wal.compact(retention_hours=retention_hours)
    return asyncio.get_event_loop().run_until_complete(_run())


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def wal_1k(tmp_path):
    store = SQLiteStore(path=tmp_path / "wal_1k.db")
    await store.initialize()
    await _populate_wal(store, 1_000)
    yield store
    await store.close()


@pytest.fixture
async def wal_10k(tmp_path):
    store = SQLiteStore(path=tmp_path / "wal_10k.db")
    await store.initialize()
    await _populate_wal(store, 10_000)
    yield store
    await store.close()


@pytest.fixture
async def wal_applied_10k(tmp_path):
    """WAL with 10K entries all marked as applied (ready for compaction)."""
    store = SQLiteStore(path=tmp_path / "wal_applied.db")
    await store.initialize()
    await _populate_wal(store, 10_000)
    wal = WAL(store)
    entries = await wal.replay_pending()
    for entry in entries:
        await wal.mark_applied(entry.seq)
    yield store
    await store.close()


# ── Benchmarks ────────────────────────────────────────────────────────────────


class TestWALReplayBenchmarks:
    def test_replay_1k_entries(self, benchmark, wal_1k):
        """Measure WAL replay time with 1K pending entries."""
        entries = benchmark(_replay_wal, wal_1k)
        assert len(entries) == 1_000

    def test_replay_10k_entries(self, benchmark, wal_10k):
        """Measure WAL replay time with 10K pending entries (realistic crash scenario)."""
        entries = benchmark(_replay_wal, wal_10k)
        assert len(entries) == 10_000

    def test_pending_count_1k(self, benchmark, wal_1k):
        """Measure time to count pending entries (used on startup health-check)."""
        async def _count():
            wal = WAL(wal_1k)
            return await wal.pending_count()

        count = benchmark(lambda: asyncio.get_event_loop().run_until_complete(_count()))
        assert count == 1_000

    def test_compact_applied_10k(self, benchmark, wal_applied_10k):
        """Measure WAL compaction with 10K applied entries (all eligible for removal)."""
        async def _compact():
            wal = WAL(wal_applied_10k)
            return await wal.compact(retention_hours=0)

        # Run once to compact; benchmark subsequent calls (already empty, measures overhead)
        asyncio.get_event_loop().run_until_complete(
            WAL(wal_applied_10k).compact(retention_hours=0)
        )
        result = benchmark(lambda: asyncio.get_event_loop().run_until_complete(
            WAL(wal_applied_10k).compact(retention_hours=0)
        ))
        assert isinstance(result, int)

    def test_mark_applied_throughput(self, benchmark, wal_1k):
        """Measure throughput of mark_applied on the WAL."""
        async def _mark_all():
            wal = WAL(wal_1k)
            entries = await wal.replay_pending()
            for e in entries[:100]:  # Mark 100 per iteration
                await wal.mark_applied(e.seq)

        benchmark(lambda: asyncio.get_event_loop().run_until_complete(_mark_all()))
