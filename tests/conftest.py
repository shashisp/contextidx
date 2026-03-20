"""Shared test fixtures."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict

import pytest

from contextidx.backends.base import SearchResult, VectorBackend
from contextidx.store.sqlite_store import SQLiteStore


class InMemoryVectorBackend(VectorBackend):
    """In-memory vector backend for testing without real DB."""

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
            score = _cosine_similarity(query_embedding, emb)
            results.append(SearchResult(id=vid, score=score, metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._vectors.pop(id, None)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            emb, existing = self._vectors[id]
            existing.update(metadata)
            self._vectors[id] = (emb, existing)

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass


class InMemoryHybridBackend(VectorBackend):
    """In-memory backend with hybrid search and metadata store support.

    Simulates a backend like Weaviate that supports both hybrid (BM25 + vector)
    search and native metadata storage.
    """

    def __init__(self):
        self._vectors: dict[str, tuple[list[float], dict]] = {}

    @property
    def supports_metadata_store(self) -> bool:
        return True

    @property
    def supports_hybrid_search(self) -> bool:
        return True

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
            score = _cosine_similarity(query_embedding, emb)
            results.append(SearchResult(id=vid, score=score, metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Simulated hybrid search that combines vector similarity with keyword matching."""
        results = []
        query_terms = set(query.lower().split())

        for vid, (emb, meta) in self._vectors.items():
            if filters:
                scope = meta.get("scope", {})
                if not all(scope.get(k) == v for k, v in filters.items()):
                    continue

            vec_score = _cosine_similarity(query_embedding, emb)
            bm25_score = _simple_bm25(query_terms, meta)
            combined = alpha * vec_score + (1.0 - alpha) * bm25_score

            result_meta = dict(meta)
            result_meta["bm25_score"] = bm25_score
            results.append(
                SearchResult(id=vid, score=combined, metadata=result_meta)
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._vectors.pop(id, None)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._vectors:
            emb, existing = self._vectors[id]
            existing.update(metadata)
            self._vectors[id] = (emb, existing)

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _simple_bm25(query_terms: set[str], meta: dict) -> float:
    """Naive BM25-like scoring for test purposes.

    Counts how many query terms appear in the metadata content/source fields.
    """
    text_fields = " ".join(
        str(v) for k, v in meta.items() if k in {"content", "source"}
    ).lower()
    if not text_fields or not query_terms:
        return 0.0
    matches = sum(1 for t in query_terms if t in text_fields)
    return matches / len(query_terms)


@pytest.fixture
def mock_backend():
    return InMemoryVectorBackend()


@pytest.fixture
def mock_hybrid_backend():
    return InMemoryHybridBackend()


@pytest.fixture
async def sqlite_store(tmp_path):
    store = SQLiteStore(path=tmp_path / "test_meta.db")
    await store.initialize()
    yield store
    await store.close()
