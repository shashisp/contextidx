"""Tests for LLM re-ranking (OpenAIReranker) with mocked OpenAI responses.

Covers:
- Score blending at the 0.4/0.6 (composite/LLM) ratio
- Correct re-sorting of results after blending
- Graceful fallback to original ordering on any API error
- Short-circuit when rerank_pool <= 1 candidate
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.reranker import OpenAIReranker


# ── helpers ──────────────────────────────────────────────────────────────────


def _unit(content: str, confidence: float = 0.5) -> ContextUnit:
    u = ContextUnit(
        id=generate_unit_id(),
        content=content,
        scope={"user_id": "u1"},
        confidence=confidence,
        source="test",
    )
    return u


def _candidates(
    units: list[ContextUnit],
    composites: list[float],
) -> list[tuple[ContextUnit, float, float]]:
    return [(u, c, 0.0) for u, c in zip(units, composites)]


def _mock_openai_response(rankings: list[dict[str, Any]]) -> Any:
    """Build a mock OpenAI chat completion response with the given JSON rankings."""
    content = json.dumps(rankings)
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


# ── tests ─────────────────────────────────────────────────────────────────────


class TestOpenAIRerankerScoreBlending:
    """Verify the 0.4/0.6 composite/LLM blend formula."""

    @pytest.mark.asyncio
    async def test_blend_formula(self):
        """blended = 0.4 * composite + 0.6 * (llm_score/10)."""
        unit_a = _unit("Very relevant fact", confidence=0.9)
        unit_b = _unit("Less relevant fact", confidence=0.5)
        candidates = _candidates([unit_a, unit_b], [0.9, 0.5])

        # LLM rates a=9, b=2 on 0-10 scale
        mock_resp = _mock_openai_response([{"index": 0, "score": 9}, {"index": 1, "score": 2}])

        reranker = OpenAIReranker(blend=0.6)
        reranker._client = MagicMock()
        reranker._client.chat = MagicMock()
        reranker._client.chat.completions = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="test", candidates=candidates, top_k=2)

        # unit_a: 0.4 * 0.9 + 0.6 * 0.9 = 0.36 + 0.54 = 0.90
        # unit_b: 0.4 * 0.5 + 0.6 * 0.2 = 0.20 + 0.12 = 0.32
        blended_a = 0.4 * 0.9 + 0.6 * (9 / 10)
        blended_b = 0.4 * 0.5 + 0.6 * (2 / 10)

        assert len(result) == 2
        assert result[0][0].id == unit_a.id
        assert abs(result[0][1] - blended_a) < 1e-9
        assert abs(result[1][1] - blended_b) < 1e-9

    @pytest.mark.asyncio
    async def test_blend_zero_is_pure_composite(self):
        """blend=0 → blended score equals composite score."""
        unit_a = _unit("A", confidence=0.8)
        unit_b = _unit("B", confidence=0.6)
        candidates = _candidates([unit_a, unit_b], [0.8, 0.6])

        mock_resp = _mock_openai_response([{"index": 0, "score": 10}, {"index": 1, "score": 0}])

        reranker = OpenAIReranker(blend=0.0)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="test", candidates=candidates, top_k=2)

        # With blend=0: blended = 1.0 * composite + 0.0 * llm
        assert abs(result[0][1] - 0.8) < 1e-9
        assert abs(result[1][1] - 0.6) < 1e-9

    @pytest.mark.asyncio
    async def test_blend_one_is_pure_llm(self):
        """blend=1 → blended score is purely LLM-derived."""
        unit_a = _unit("A", confidence=0.1)
        unit_b = _unit("B", confidence=0.9)
        candidates = _candidates([unit_a, unit_b], [0.1, 0.9])

        # LLM ranks A much higher than B (reversing composite order)
        mock_resp = _mock_openai_response([{"index": 0, "score": 10}, {"index": 1, "score": 1}])

        reranker = OpenAIReranker(blend=1.0)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="test", candidates=candidates, top_k=2)

        # blend=1: blended = llm_score/10 only
        # A: 1.0, B: 0.1 → A is first despite low composite
        assert result[0][0].id == unit_a.id
        assert abs(result[0][1] - 1.0) < 1e-9


class TestOpenAIRerankerReordering:
    """Verify results are re-sorted by blended score."""

    @pytest.mark.asyncio
    async def test_reorder_reverses_original_ranking(self):
        """LLM can completely reverse the composite order."""
        unit_low = _unit("Low composite but highly relevant", confidence=0.2)
        unit_high = _unit("High composite but less relevant", confidence=0.9)
        # composite order: high first, low second
        candidates = _candidates([unit_high, unit_low], [0.9, 0.2])

        # LLM rates low=10, high=2 → reversal
        mock_resp = _mock_openai_response([{"index": 0, "score": 2}, {"index": 1, "score": 10}])

        reranker = OpenAIReranker(blend=0.8)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="find something specific", candidates=candidates, top_k=2)

        assert result[0][0].id == unit_low.id, "LLM should override composite ordering"

    @pytest.mark.asyncio
    async def test_candidates_beyond_rerank_pool_appended_unchanged(self):
        """Candidates outside the rerank pool (top_k * 2) pass through unchanged."""
        units = [_unit(f"unit {i}") for i in range(6)]
        composites = [0.9 - i * 0.1 for i in range(6)]
        candidates = _candidates(units, composites)

        # top_k=2 → rerank_pool = candidates[:4], tail = candidates[4:]
        mock_resp = _mock_openai_response(
            [{"index": 0, "score": 5}, {"index": 1, "score": 5},
             {"index": 2, "score": 5}, {"index": 3, "score": 5}]
        )

        reranker = OpenAIReranker(blend=0.5)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="q", candidates=candidates, top_k=2)

        # All 6 units should be present — the tail passes through
        assert len(result) == 6
        tail_ids = {r[0].id for r in result[4:]}
        assert tail_ids == {units[4].id, units[5].id}


class TestRerankerFallbackViaContextIdx:
    """Verify ContextIdx._rerank_with_llm falls back on errors.

    The fallback logic lives in ``ContextIdx._rerank_with_llm``, not in
    ``OpenAIReranker`` itself.  We test it by calling the method on a
    minimal duck-typed mock that satisfies its attribute requirements.
    """

    def _make_host(self, reranker):
        """Build a minimal host object that exposes _rerank_with_llm."""
        from contextidx.contextidx import ContextIdx

        host = object.__new__(ContextIdx)
        host._reranker = reranker
        host._rerank_client = None
        return host

    @pytest.mark.asyncio
    async def test_fallback_on_api_error(self):
        """API exception → _rerank_with_llm returns original ordering."""
        unit_a = _unit("A")
        unit_b = _unit("B")
        scored = _candidates([unit_a, unit_b], [0.9, 0.5])

        failing_reranker = AsyncMock(side_effect=Exception("API failure"))
        host = self._make_host(failing_reranker)

        from contextidx.contextidx import ContextIdx
        result = await ContextIdx._rerank_with_llm(host, "test", scored, 2)

        assert result[0][0].id == unit_a.id
        assert result[1][0].id == unit_b.id

    @pytest.mark.asyncio
    async def test_fallback_on_malformed_json(self):
        """Malformed JSON from reranker → fallback to original ordering."""
        unit_a = _unit("A")
        unit_b = _unit("B")
        scored = _candidates([unit_a, unit_b], [0.9, 0.5])

        bad_reranker = AsyncMock(side_effect=ValueError("invalid json"))
        host = self._make_host(bad_reranker)

        from contextidx.contextidx import ContextIdx
        result = await ContextIdx._rerank_with_llm(host, "test", scored, 2)

        assert result[0][0].id == unit_a.id

    @pytest.mark.asyncio
    async def test_no_api_call_when_single_candidate(self):
        """Single candidate → _rerank_with_llm short-circuits before calling reranker."""
        unit_a = _unit("only candidate")
        scored = _candidates([unit_a], [0.8])

        never_called = AsyncMock(side_effect=Exception("should not be called"))
        host = self._make_host(never_called)

        from contextidx.contextidx import ContextIdx
        result = await ContextIdx._rerank_with_llm(host, "test", scored, 5)

        never_called.assert_not_called()
        assert len(result) == 1
        assert result[0][0].id == unit_a.id

    @pytest.mark.asyncio
    async def test_missing_index_in_llm_response_uses_default_score(self):
        """If LLM omits an index, that candidate gets the default score 5.0/10."""
        unit_a = _unit("A")
        unit_b = _unit("B")
        candidates = _candidates([unit_a, unit_b], [0.8, 0.6])

        # LLM only returns score for index 0, omits index 1
        mock_resp = _mock_openai_response([{"index": 0, "score": 9}])

        reranker = OpenAIReranker(blend=0.6)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="test", candidates=candidates, top_k=2)

        # unit_b missing from LLM response → default score 5.0/10 = 0.5
        # unit_a blended: 0.4*0.8 + 0.6*0.9 = 0.32 + 0.54 = 0.86
        # unit_b blended: 0.4*0.6 + 0.6*0.5 = 0.24 + 0.30 = 0.54
        assert result[0][0].id == unit_a.id

    @pytest.mark.asyncio
    async def test_markdown_fenced_json_parsed_correctly(self):
        """LLM response wrapped in ```json fences should be parsed correctly."""
        unit_a = _unit("A")
        unit_b = _unit("B")
        candidates = _candidates([unit_a, unit_b], [0.5, 0.5])

        fenced_content = '```json\n[{"index":0,"score":8},{"index":1,"score":3}]\n```'
        mock_resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=fenced_content))]
        )

        reranker = OpenAIReranker(blend=0.6)
        reranker._client = MagicMock()
        reranker._client.chat.completions.create = AsyncMock(return_value=mock_resp)

        result = await reranker(query="test", candidates=candidates, top_k=2)

        assert result[0][0].id == unit_a.id
        assert result[0][1] > result[1][1]
