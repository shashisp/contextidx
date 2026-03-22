"""Hybrid search benchmarks (BM25 + vector RRF).

Uses a mock backend that exposes ``supports_hybrid_search = True`` so the
retrieval path exercises the hybrid code branch in ``aretrieve``.

Run with::

    pytest benchmarks/bench_hybrid.py --benchmark-only -v
"""

from __future__ import annotations

import asyncio
import math
import random

import pytest

from benchmarks.conftest import EMBEDDING_DIM, _make_units, _random_embedding
from contextidx.backends.base import SearchResult, VectorBackend
from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine
from contextidx.store.sqlite_store import SQLiteStore


# ── Hybrid-capable backend ────────────────────────────────────────────────────


class HybridBenchmarkBackend(VectorBackend):
    """In-memory backend that reports hybrid search support.

    BM25 scores are simulated via random values so we measure the overhead
    of the hybrid code path, not actual BM25 computation.
    """

    supports_hybrid_search = True
    supports_metadata_store = False

    def __init__(self) -> None:
        self._vectors: dict[str, tuple[list[float], dict]] = {}

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    async def store(self, id: str, embedding: list[float], metadata: dict | None = None) -> str:
        self._vectors[id] = (embedding, metadata or {})
        return id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        results = []
        for vid, (emb, meta) in self._vectors.items():
            if filters:
                scope = meta.get("scope", {})
                if not all(scope.get(k) == v for k, v in filters.items()):
                    continue
            dot = sum(a * b for a, b in zip(query_embedding, emb))
            results.append(SearchResult(id=vid, score=dot, metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Simulate RRF fusion: vector score + random BM25 score."""
        vec_results = await self.search(query_embedding, top_k * 2, filters)
        results = []
        for sr in vec_results[:top_k]:
            bm25 = random.uniform(0.0, 1.0)
            meta = dict(sr.metadata)
            meta["bm25_score"] = bm25
            # Reciprocal Rank Fusion: rrf = 1/(k+rank)
            rrf = 1.0 / (60 + 1) + bm25 * 0.5
            results.append(SearchResult(id=sr.id, score=rrf, metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._vectors.pop(id, None)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            emb, existing = self._vectors[id]
            existing.update(metadata)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
async def hybrid_backend_1k(tmp_path):
    units = _make_units(1_000)
    backend = HybridBenchmarkBackend()
    store = SQLiteStore(path=tmp_path / "hybrid_1k.db")
    await backend.initialize()
    await store.initialize()
    for u in units:
        await backend.store(u.id, u.embedding, {"scope": u.scope})
        await store.create_unit(u)
    yield backend, store, units
    await store.close()
    await backend.close()


@pytest.fixture
async def hybrid_backend_10k(tmp_path):
    units = _make_units(10_000)
    backend = HybridBenchmarkBackend()
    store = SQLiteStore(path=tmp_path / "hybrid_10k.db")
    await backend.initialize()
    await store.initialize()
    for u in units:
        await backend.store(u.id, u.embedding, {"scope": u.scope})
        await store.create_unit(u)
    yield backend, store, units
    await store.close()
    await backend.close()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _hybrid_retrieve(backend, store, query_emb, scope, top_k=10):
    """Run one hybrid search + scoring cycle (sync wrapper for benchmark)."""

    async def _run():
        results = await backend.hybrid_search(
            query="benchmark query text",
            query_embedding=query_emb,
            top_k=top_k * 3,
            filters=scope,
        )
        ids = [sr.id for sr in results]
        units_map = await store.get_units_batch(ids)
        engine = ScoringEngine()
        decay_engine = DecayEngine()
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        scored = []
        for sr in results:
            unit = units_map.get(sr.id)
            if unit is None:
                continue
            bm25 = sr.metadata.get("bm25_score")
            decay = decay_engine.compute_decay(unit, now)
            score = engine.compute_score(
                unit=unit,
                semantic_score=sr.score,
                query_time=now,
                decay_score=decay,
                reinforcement_count=0,
                bm25_score=bm25,
            )
            scored.append((unit, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    return asyncio.get_event_loop().run_until_complete(_run())


# ── Benchmarks ────────────────────────────────────────────────────────────────


class TestHybridSearchBenchmarks:
    def test_hybrid_retrieve_1k(self, benchmark, hybrid_backend_1k):
        backend, store, units = hybrid_backend_1k
        q_emb = _random_embedding()
        scope = {"user_id": "user_0"}
        result = benchmark(_hybrid_retrieve, backend, store, q_emb, scope)
        assert len(result) <= 10

    def test_hybrid_retrieve_10k(self, benchmark, hybrid_backend_10k):
        backend, store, units = hybrid_backend_10k
        q_emb = _random_embedding()
        scope = {"user_id": "user_0"}
        result = benchmark(_hybrid_retrieve, backend, store, q_emb, scope)
        assert len(result) <= 10

    def test_hybrid_vs_vector_only_overhead(self, benchmark, hybrid_backend_1k):
        """Measure additional latency introduced by BM25 fusion vs pure vector."""
        backend, store, units = hybrid_backend_1k
        q_emb = _random_embedding()
        scope = {"user_id": "user_0"}

        async def _vector_only():
            return await backend.search(q_emb, top_k=30, filters=scope)

        def run_vector():
            return asyncio.get_event_loop().run_until_complete(_vector_only())

        result = benchmark(run_vector)
        assert len(result) <= 30
