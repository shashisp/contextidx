"""Benchmark fixtures that pre-populate InMemoryVectorBackend at various scales."""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.backends.base import SearchResult, VectorBackend
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.store.sqlite_store import SQLiteStore

EMBEDDING_DIM = 64


def _random_embedding(dim: int = EMBEDDING_DIM) -> list[float]:
    vec = [random.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


def _make_units(n: int) -> list[ContextUnit]:
    base = datetime.now(timezone.utc) - timedelta(days=90)
    units = []
    for i in range(n):
        unit = ContextUnit(
            id=generate_unit_id(),
            content=f"benchmark context unit number {i}",
            scope={"user_id": f"user_{i % 10}"},
            confidence=0.5 + random.random() * 0.5,
            source="benchmark",
            decay_model="exponential",
            decay_rate=0.02,
        )
        unit.embedding = _random_embedding()
        unit.timestamp = base + timedelta(seconds=i * 60)
        units.append(unit)
    return units


class BenchmarkBackend(VectorBackend):
    """Simple in-memory backend for benchmarking (no filtering overhead)."""

    def __init__(self):
        self._vectors: dict[str, tuple[list[float], dict]] = {}

    @property
    def supports_metadata_store(self) -> bool:
        return False

    async def store(self, id: str, embedding: list[float], metadata: dict | None = None) -> str:
        self._vectors[id] = (embedding, metadata or {})
        return id

    async def search(
        self, query_embedding: list[float], top_k: int, filters: dict | None = None
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

    async def delete(self, id: str) -> None:
        self._vectors.pop(id, None)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            emb, existing = self._vectors[id]
            existing.update(metadata)

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.fixture(params=[1_000, 10_000])
def unit_population(request):
    """Parametrized fixture returning (units, backend, store_path) at scale."""
    return _make_units(request.param)


@pytest.fixture
def units_1k():
    return _make_units(1_000)


@pytest.fixture
def units_10k():
    return _make_units(10_000)


@pytest.fixture
def units_100k():
    return _make_units(100_000)


_CACHED_1M = None


@pytest.fixture(scope="session")
def units_1m():
    """Session-scoped 1M fixture to avoid re-creation across test functions."""
    global _CACHED_1M  # noqa: PLW0603
    if _CACHED_1M is None:
        _CACHED_1M = _make_units(1_000_000)
    return _CACHED_1M


@pytest.fixture
def bench_backend():
    return BenchmarkBackend()


@pytest.fixture
def query_embedding():
    return _random_embedding()
