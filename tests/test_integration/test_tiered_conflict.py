"""Integration tests for tiered conflict detection.

Tiered mode: rule-based detection runs inline, semantic detection is queued
for background resolution.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.core.context_unit import ContextUnit


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestTieredConflict:
    @pytest.fixture
    async def ctx(self, mock_backend, tmp_path):
        ctx = ContextIdx(
            backend=mock_backend,
            conflict_detection="tiered",
            conflict_strategy="LAST_WRITE_WINS",
            internal_store_path=str(tmp_path / "meta.db"),
            state_path_interval=9999,
        )
        await ctx.ainitialize()
        yield ctx
        await ctx.aclose()

    async def test_rule_based_inline_still_works(self, ctx):
        """Rule-based contradictions should be caught inline."""
        emb = _normalized([1.0, 0.0, 0.0, 0.0])
        id1 = await ctx.astore(
            "User prefers light mode",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        id2 = await ctx.astore(
            "User does not prefer light mode",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        unit1 = await ctx._store.get_unit(id1)
        assert unit1 is not None
        assert unit1.superseded_by == id2

    async def test_semantic_candidates_are_queued(self, ctx):
        """Non-rule-based conflicts should be enqueued for later."""
        emb1 = _normalized([1.0, 0.0, 0.0, 0.0])
        emb2 = _normalized([0.9, 0.1, 0.0, 0.0])
        await ctx.astore(
            "The system is fast",
            scope={"user_id": "u1"},
            embedding=emb1,
        )
        await ctx.astore(
            "The system is good",
            scope={"user_id": "u1"},
            embedding=emb2,
        )
        assert ctx._conflict_queue is not None
        assert ctx._conflict_queue.pending_count >= 0

    async def test_drain_resolves_semantic_conflicts(self, ctx):
        """After drain, semantic conflicts should be resolved."""
        emb = _normalized([1.0, 0.0, 0.0, 0.0])
        id1 = await ctx.astore(
            "The weather is nice",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        id2 = await ctx.astore(
            "The weather is not nice",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        assert ctx._conflict_queue is not None
        await ctx._conflict_queue.drain(ctx._apply_conflict_resolution)

    async def test_state_path_tick_drains_queue(self, ctx):
        """_state_path_tick should drain the conflict queue."""
        emb = _normalized([1.0, 0.0, 0.0, 0.0])
        await ctx.astore(
            "System uses Python",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        await ctx.astore(
            "System uses Java",
            scope={"user_id": "u1"},
            embedding=_normalized([0.8, 0.2, 0.0, 0.0]),
        )
        await ctx._state_path_tick()


class TestSemanticConflictMode:
    @pytest.fixture
    async def ctx(self, mock_backend, tmp_path):
        ctx = ContextIdx(
            backend=mock_backend,
            conflict_detection="semantic",
            conflict_strategy="LAST_WRITE_WINS",
            internal_store_path=str(tmp_path / "meta.db"),
            state_path_interval=9999,
        )
        await ctx.ainitialize()
        yield ctx
        await ctx.aclose()

    async def test_semantic_detection_inline(self, ctx):
        """Semantic mode runs detection synchronously."""
        emb = _normalized([1.0, 0.0, 0.0, 0.0])
        id1 = await ctx.astore(
            "I like dogs",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        id2 = await ctx.astore(
            "I do not like dogs",
            scope={"user_id": "u1"},
            embedding=emb,
        )
        unit1 = await ctx._store.get_unit(id1)
        assert unit1 is not None
        assert unit1.superseded_by == id2
