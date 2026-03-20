"""Benchmark single vs batched write paths.

These benchmarks measure the store-layer cost (SQLite + in-memory backend)
without hitting a real embedding API.
"""

from __future__ import annotations

import asyncio
import math
import random

import pytest

from benchmarks.conftest import EMBEDDING_DIM, BenchmarkBackend, _random_embedding
from contextidx.store.sqlite_store import SQLiteStore


@pytest.fixture
async def store_and_backend(tmp_path):
    backend = BenchmarkBackend()
    store = SQLiteStore(path=tmp_path / "bench.db")
    await store.initialize()
    await backend.initialize()
    yield store, backend
    await store.close()
    await backend.close()


def test_single_write_100(benchmark, store_and_backend):
    """100 sequential stores through the backend + SQLite."""
    store, backend = store_and_backend

    async def run():
        from contextidx.core.context_unit import ContextUnit, generate_unit_id

        for i in range(100):
            unit = ContextUnit(
                id=generate_unit_id(),
                content=f"bench write {i}",
                scope={"user_id": "u1"},
                confidence=0.8,
                source="bench",
                decay_model="exponential",
                decay_rate=0.02,
            )
            unit.embedding = _random_embedding()
            await backend.store(
                id=unit.id,
                embedding=unit.embedding,
                metadata={"scope": unit.scope, "source": unit.source},
            )
            await store.create_unit(unit)

    benchmark(lambda: asyncio.get_event_loop().run_until_complete(run()))
