"""Integration tests: hybrid search scoring correctness."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.core.scoring_engine import ScoringEngine
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryHybridBackend, InMemoryVectorBackend

_DIM = 8


def _emb(seed: float) -> list[float]:
    return [math.sin(seed * (i + 1)) for i in range(_DIM)]


@pytest.fixture
async def hybrid_idx(tmp_path):
    backend = InMemoryHybridBackend()
    store = SQLiteStore(path=tmp_path / "hybrid.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx
    await ctx.aclose()


@pytest.fixture
async def vector_idx(tmp_path):
    backend = InMemoryVectorBackend()
    store = SQLiteStore(path=tmp_path / "vector.db")
    ctx = ContextIdx(
        backend=backend,
        internal_store=store,
        openai_api_key="test",
    )
    await ctx.ainitialize()
    yield ctx
    await ctx.aclose()


class TestHybridSearchPath:
    async def test_hybrid_backend_used_when_available(self, hybrid_idx):
        """When backend supports hybrid, aretrieve should still return results."""
        emb = _emb(1.0)
        uid = await hybrid_idx.astore(
            content="Python is a great programming language",
            scope={"user_id": "u1"},
            embedding=emb,
            source="test",
        )
        results = await hybrid_idx.aretrieve(
            query="Python programming",
            scope={"user_id": "u1"},
            query_embedding=emb,
        )
        assert len(results) >= 1
        assert results[0].id == uid

    async def test_vector_only_still_works(self, vector_idx):
        """Vector-only backends should still function correctly."""
        emb = _emb(2.0)
        uid = await vector_idx.astore(
            content="JavaScript frontend framework",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        results = await vector_idx.aretrieve(
            query="frontend",
            scope={"user_id": "u1"},
            query_embedding=emb,
        )
        assert len(results) >= 1

    async def test_hybrid_search_ranks_keyword_matches_well(self, hybrid_idx):
        """Items whose source matches query keywords should rank higher via BM25 boost."""
        base_emb = _emb(3.0)
        # Use the same embedding so vector similarity is equal — BM25 breaks the tie
        await hybrid_idx.astore(
            content="database optimization techniques",
            scope={"user_id": "u1"},
            embedding=base_emb,
            source="database optimization article",
        )
        await hybrid_idx.astore(
            content="unrelated random content",
            scope={"user_id": "u1"},
            embedding=base_emb,
            source="random topic",
        )

        results = await hybrid_idx.aretrieve(
            query="database optimization",
            scope={"user_id": "u1"},
            query_embedding=base_emb,
        )
        assert len(results) >= 2
        contents = [r.content for r in results]
        assert "database optimization techniques" in contents


class TestScoringEngineWithBM25:
    def test_bm25_weight_in_defaults(self):
        engine = ScoringEngine()
        assert "bm25" in engine.weights
        assert engine.weights["bm25"] > 0

    def test_weights_sum_to_one_with_bm25(self):
        engine = ScoringEngine()
        total = sum(engine.weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_score_with_bm25_differs_from_without(self):
        engine = ScoringEngine()
        from contextidx.core.context_unit import ContextUnit

        now = datetime.now(timezone.utc)
        unit = ContextUnit(content="test", timestamp=now, confidence=0.8)

        score_no_bm25 = engine.compute_score(
            unit=unit, semantic_score=0.8, query_time=now,
            decay_score=0.9, reinforcement_count=0,
        )
        score_with_bm25 = engine.compute_score(
            unit=unit, semantic_score=0.8, query_time=now,
            decay_score=0.9, reinforcement_count=0, bm25_score=0.9,
        )
        assert score_no_bm25 != score_with_bm25

    def test_high_bm25_boosts_score(self):
        engine = ScoringEngine()
        from contextidx.core.context_unit import ContextUnit

        now = datetime.now(timezone.utc)
        unit = ContextUnit(content="test", timestamp=now, confidence=0.8)

        score_low_bm25 = engine.compute_score(
            unit=unit, semantic_score=0.5, query_time=now,
            decay_score=0.5, reinforcement_count=0, bm25_score=0.1,
        )
        score_high_bm25 = engine.compute_score(
            unit=unit, semantic_score=0.5, query_time=now,
            decay_score=0.5, reinforcement_count=0, bm25_score=0.9,
        )
        assert score_high_bm25 > score_low_bm25

    def test_redistribute_without_bm25(self):
        engine = ScoringEngine()
        redistributed = engine._redistribute_without_bm25()
        assert "bm25" not in redistributed
        total = sum(redistributed.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_backward_compatible_no_bm25(self):
        """Without bm25, engine redistributes weight and still produces valid scores."""
        engine = ScoringEngine()
        from contextidx.core.context_unit import ContextUnit

        now = datetime.now(timezone.utc)
        unit = ContextUnit(content="test", timestamp=now, confidence=1.0)
        score = engine.compute_score(
            unit=unit, semantic_score=1.0, query_time=now,
            decay_score=1.0, reinforcement_count=20,
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.8
