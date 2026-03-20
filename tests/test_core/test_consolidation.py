"""Tests for the consolidation module: redundancy detection, merge, lineage checks."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.core.consolidation import (
    find_redundant_pairs,
    merge_units,
    should_summarize_lineage,
)
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.temporal_graph import Relationship, TemporalGraph


def _unit(content: str, embedding: list[float] | None = None, **kw) -> ContextUnit:
    defaults = {
        "id": generate_unit_id(),
        "content": content,
        "scope": {"user_id": "u1"},
        "confidence": 0.8,
        "source": "test",
        "decay_model": "exponential",
        "decay_rate": 0.02,
    }
    defaults.update(kw)
    u = ContextUnit(**defaults)
    if embedding is not None:
        u.embedding = embedding
    return u


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestFindRedundantPairs:
    def test_identical_embeddings_are_redundant(self):
        emb = _normalized([1.0, 0.0, 0.0])
        u1 = _unit("alpha", embedding=emb, confidence=0.9)
        u2 = _unit("beta", embedding=emb, confidence=0.7)
        pairs = find_redundant_pairs([u1, u2], threshold=0.90)
        assert len(pairs) == 1
        assert pairs[0] == (u1.id, u2.id)

    def test_orthogonal_embeddings_not_redundant(self):
        u1 = _unit("alpha", embedding=_normalized([1.0, 0.0, 0.0]))
        u2 = _unit("beta", embedding=_normalized([0.0, 1.0, 0.0]))
        pairs = find_redundant_pairs([u1, u2], threshold=0.90)
        assert len(pairs) == 0

    def test_different_scopes_not_redundant(self):
        emb = _normalized([1.0, 0.0, 0.0])
        u1 = _unit("alpha", embedding=emb, scope={"user_id": "u1"})
        u2 = _unit("beta", embedding=emb, scope={"user_id": "u2"})
        pairs = find_redundant_pairs([u1, u2], threshold=0.90)
        assert len(pairs) == 0

    def test_no_embedding_skipped(self):
        u1 = _unit("alpha", embedding=_normalized([1.0, 0.0, 0.0]))
        u2 = _unit("beta")  # no embedding
        pairs = find_redundant_pairs([u1, u2], threshold=0.90)
        assert len(pairs) == 0

    def test_keeper_is_higher_confidence(self):
        emb = _normalized([1.0, 0.0, 0.0])
        u_low = _unit("low", embedding=emb, confidence=0.5)
        u_high = _unit("high", embedding=emb, confidence=0.9)
        pairs = find_redundant_pairs([u_low, u_high], threshold=0.90)
        assert pairs[0][0] == u_high.id
        assert pairs[0][1] == u_low.id


class TestMergeUnits:
    def test_merged_confidence_is_average(self):
        emb = _normalized([1.0, 0.0, 0.0])
        keeper = _unit("keeper", embedding=emb, confidence=0.8)
        absorbed = _unit("absorbed", embedding=emb, confidence=0.6)
        result = merge_units(keeper, absorbed)
        assert result.confidence == pytest.approx(0.7)
        assert result.version == keeper.version + 1

    def test_merged_confidence_capped_at_one(self):
        emb = _normalized([1.0, 0.0, 0.0])
        keeper = _unit("keeper", embedding=emb, confidence=1.0)
        absorbed = _unit("absorbed", embedding=emb, confidence=1.0)
        result = merge_units(keeper, absorbed)
        assert result.confidence <= 1.0


class TestShouldSummarizeLineage:
    def test_short_chain_below_threshold(self):
        graph = TemporalGraph()
        now = datetime.now(timezone.utc)
        ids = [generate_unit_id() for _ in range(5)]
        for i in range(len(ids) - 1):
            graph.add_edge(ids[i], ids[i + 1], Relationship.VERSION_OF, now)
        assert not should_summarize_lineage(ids[0], graph, max_chain=10)

    def test_long_chain_above_threshold(self):
        graph = TemporalGraph()
        now = datetime.now(timezone.utc)
        ids = [generate_unit_id() for _ in range(15)]
        for i in range(len(ids) - 1):
            graph.add_edge(ids[i], ids[i + 1], Relationship.VERSION_OF, now)
        assert should_summarize_lineage(ids[0], graph, max_chain=10)
