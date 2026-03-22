"""Concurrent write benchmarks.

Measures throughput and correctness under concurrent ``astore_batch`` calls
using ``asyncio.gather()``.  Tests the ``PendingBuffer``, WAL, and SQLiteStore
under concurrent load.

Run with::

    pytest benchmarks/bench_concurrent.py --benchmark-only -v
"""

from __future__ import annotations

import asyncio
import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from benchmarks.conftest import EMBEDDING_DIM, BenchmarkBackend, _random_embedding
from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore


# ── helpers ───────────────────────────────────────────────────────────────────


class _FakeEmbedder:
    """Deterministic embedder that never calls a real API."""

    async def embed(self, text: str) -> list[float]:
        return _random_embedding(EMBEDDING_DIM)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [_random_embedding(EMBEDDING_DIM) for _ in texts]


def _chunk_texts(n_writers: int, n_per_writer: int) -> list[list[dict]]:
    """Return *n_writers* batches, each with *n_per_writer* items."""
    base = datetime.now(timezone.utc) - timedelta(days=30)
    batches: list[list[dict]] = []
    for w in range(n_writers):
        items = [
            {
                "content": f"writer {w} unit {i}: context data about user activity",
                "scope": {"user_id": f"user_{w % 5}"},
                "source": f"writer_{w}",
                "timestamp": base + timedelta(seconds=i * 60),
            }
            for i in range(n_per_writer)
        ]
        batches.append(items)
    return batches


def _run_concurrent(idx: ContextIdx, batches: list[list[dict]]) -> list[str]:
    """Run all batches concurrently and return all stored IDs."""
    async def _go():
        tasks = [idx.astore_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        return [uid for ids in results for uid in ids]

    return asyncio.get_event_loop().run_until_complete(_go())


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def ctx_instance(tmp_path):
    store = SQLiteStore(path=tmp_path / "concurrent.db")
    backend = BenchmarkBackend()
    idx = ContextIdx(
        backend=backend,
        internal_store=store,
        conflict_detection="none",
        embedding_fn=_FakeEmbedder(),
        decay_threshold=0.0,
    )
    await idx.ainitialize()
    yield idx
    await idx.aclose()


# ── benchmarks ────────────────────────────────────────────────────────────────


class TestConcurrentWriteBenchmarks:
    def test_concurrent_5_writers_20_each(self, benchmark, ctx_instance):
        """5 concurrent writers × 20 units each = 100 total writes."""
        batches = _chunk_texts(n_writers=5, n_per_writer=20)
        ids = benchmark(_run_concurrent, ctx_instance, batches)
        # All 100 units should be stored (no conflicts because conflict_detection=none)
        assert len(ids) == 100

    def test_concurrent_10_writers_10_each(self, benchmark, ctx_instance):
        """10 concurrent writers × 10 units each = 100 total writes."""
        batches = _chunk_texts(n_writers=10, n_per_writer=10)
        ids = benchmark(_run_concurrent, ctx_instance, batches)
        assert len(ids) == 100

    def test_concurrent_20_writers_5_each(self, benchmark, ctx_instance):
        """20 concurrent writers × 5 units each = 100 total writes."""
        batches = _chunk_texts(n_writers=20, n_per_writer=5)
        ids = benchmark(_run_concurrent, ctx_instance, batches)
        assert len(ids) == 100

    def test_concurrent_writes_no_duplicates(self, ctx_instance):
        """Concurrent writes must produce unique IDs (no races on ID generation)."""
        batches = _chunk_texts(n_writers=10, n_per_writer=50)
        ids = _run_concurrent(ctx_instance, batches)
        assert len(ids) == len(set(ids)), "Duplicate IDs found under concurrent load"
