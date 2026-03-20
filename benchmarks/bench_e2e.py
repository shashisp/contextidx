"""End-to-end benchmarks: populate stores and measure retrieve + store throughput.

Runs against InMemoryBackend + SQLiteStore to isolate contextidx overhead
from network / external service latency.
"""

from __future__ import annotations

import asyncio
import random
from datetime import datetime, timezone

import pytest

from benchmarks.conftest import EMBEDDING_DIM, BenchmarkBackend, _make_units, _random_embedding
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine
from contextidx.store.sqlite_store import SQLiteStore


@pytest.fixture
async def populated_store_10k(tmp_path, units_10k):
    store = SQLiteStore(path=tmp_path / "e2e_10k.db")
    await store.initialize()
    backend = BenchmarkBackend()
    await backend.initialize()
    for u in units_10k:
        await store.create_unit(u)
        await backend.store(id=u.id, embedding=u.embedding, metadata={"scope": u.scope})
    yield store, backend, units_10k
    await store.close()
    await backend.close()


@pytest.fixture
async def populated_store_100k(tmp_path, units_100k):
    store = SQLiteStore(path=tmp_path / "e2e_100k.db")
    await store.initialize()
    backend = BenchmarkBackend()
    await backend.initialize()
    for u in units_100k:
        await store.create_unit(u)
        await backend.store(id=u.id, embedding=u.embedding, metadata={"scope": u.scope})
    yield store, backend, units_100k
    await store.close()
    await backend.close()


def _simulate_retrieve(store, backend, units, query_emb, decay_engine, scoring_engine, top_k=10):
    """Simulate the aretrieve hot path: search -> load -> score -> rank."""

    async def _run():
        results = await backend.search(query_emb, top_k=top_k * 3, filters=None)
        candidates = []
        now = datetime.now(timezone.utc)
        for sr in results:
            unit = await store.get_unit(sr.id)
            if unit is None:
                continue
            decay = decay_engine.compute_decay(unit, now)
            score = scoring_engine.compute_score(
                unit=unit,
                semantic_score=sr.score,
                query_time=now,
                decay_score=decay,
                reinforcement_count=0,
            )
            candidates.append((unit, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]

    return asyncio.get_event_loop().run_until_complete(_run())


def test_e2e_retrieve_10k(benchmark, populated_store_10k, query_embedding):
    store, backend, units = populated_store_10k
    de = DecayEngine()
    se = ScoringEngine()
    benchmark(_simulate_retrieve, store, backend, units, query_embedding, de, se)


def test_e2e_retrieve_100k(benchmark, populated_store_100k, query_embedding):
    store, backend, units = populated_store_100k
    de = DecayEngine()
    se = ScoringEngine()
    benchmark(_simulate_retrieve, store, backend, units, query_embedding, de, se)


def test_e2e_batch_store_100(benchmark, tmp_path):
    """Store 100 units sequentially (backend + SQLite) to measure throughput."""

    async def _setup():
        store = SQLiteStore(path=tmp_path / "batch_w.db")
        await store.initialize()
        backend = BenchmarkBackend()
        await backend.initialize()
        return store, backend

    store, backend = asyncio.get_event_loop().run_until_complete(_setup())

    async def run():
        for i in range(100):
            unit = ContextUnit(
                id=generate_unit_id(),
                content=f"e2e write {i}",
                scope={"user_id": "u1"},
                confidence=0.8,
                source="bench",
                decay_model="exponential",
                decay_rate=0.02,
            )
            unit.embedding = _random_embedding()
            await backend.store(id=unit.id, embedding=unit.embedding, metadata={"scope": unit.scope})
            await store.create_unit(unit)

    benchmark(lambda: asyncio.get_event_loop().run_until_complete(run()))


def test_consolidation_tick_10k(benchmark, units_10k):
    """Measure find_redundant_pairs at 10k scale."""
    from contextidx.core.consolidation import find_redundant_pairs

    benchmark(find_redundant_pairs, units_10k, 0.92)


def test_batch_decay_100k(benchmark, units_100k):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)
    rcs = [0] * len(units_100k)
    benchmark(engine.batch_compute_decay, units_100k, now, rcs)


def test_batch_score_100k(benchmark, units_100k):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)
    n = len(units_100k)
    sem = [random.random() for _ in range(n)]
    decay = [random.random() for _ in range(n)]
    rcs = [random.randint(0, 5) for _ in range(n)]
    benchmark(engine.batch_compute_score, units_100k, sem, now, decay, rcs)


# ── 1M-scale benchmarks ──


def test_batch_decay_1m(benchmark, units_1m):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)
    rcs = [0] * len(units_1m)
    benchmark(engine.batch_compute_decay, units_1m, now, rcs)


def test_batch_score_1m(benchmark, units_1m):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)
    n = len(units_1m)
    sem = [random.random() for _ in range(n)]
    decay = [random.random() for _ in range(n)]
    rcs = [random.randint(0, 5) for _ in range(n)]
    benchmark(engine.batch_compute_score, units_1m, sem, now, decay, rcs)


@pytest.fixture
async def populated_store_1m(tmp_path, units_1m):
    store = SQLiteStore(path=tmp_path / "e2e_1m.db")
    await store.initialize()
    backend = BenchmarkBackend()
    await backend.initialize()
    for u in units_1m:
        await store.create_unit(u)
        await backend.store(id=u.id, embedding=u.embedding, metadata={"scope": u.scope})
    yield store, backend, units_1m
    await store.close()
    await backend.close()


def test_e2e_retrieve_1m(benchmark, populated_store_1m, query_embedding):
    store, backend, units = populated_store_1m
    de = DecayEngine()
    se = ScoringEngine()
    benchmark(_simulate_retrieve, store, backend, units, query_embedding, de, se)
