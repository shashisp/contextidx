"""Conflict detection benchmarks.

Measures ``detect_conflicts`` (rule-based) and ``detect_semantic_conflicts``
at different scope sizes (1K, 10K existing units).

Run with::

    pytest benchmarks/bench_conflict.py --benchmark-only -v
"""

from __future__ import annotations

import math
import random

import pytest

from benchmarks.conftest import EMBEDDING_DIM, _make_units, _random_embedding
from contextidx.core.conflict_resolver import ConflictResolver
from contextidx.core.context_unit import ContextUnit, generate_unit_id


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_incoming_unit(
    scope: dict[str, str],
    negation_prefix: bool = False,
) -> ContextUnit:
    """Create a new unit to be tested for conflicts."""
    content = (
        "not I am not in Paris anymore"
        if negation_prefix
        else "I am now in Berlin for work"
    )
    unit = ContextUnit(
        id=generate_unit_id(),
        content=content,
        scope=scope,
        confidence=0.8,
        source="test",
    )
    unit.embedding = _random_embedding()
    return unit


def _make_similar_unit(base_emb: list[float], scope: dict) -> ContextUnit:
    """Create a unit whose embedding is highly similar to *base_emb*."""
    perturbed = [x + random.gauss(0, 0.01) for x in base_emb]
    norm = math.sqrt(sum(x * x for x in perturbed))
    unit = ContextUnit(
        id=generate_unit_id(),
        content="I live in Paris",
        scope=scope,
        confidence=0.7,
        source="test",
    )
    unit.embedding = [x / norm for x in perturbed]
    return unit


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def existing_1k():
    return _make_units(1_000)


@pytest.fixture
def existing_10k():
    return _make_units(10_000)


@pytest.fixture
def resolver():
    return ConflictResolver(strategy="LAST_WRITE_WINS", semantic_threshold=0.80)


# ── Rule-based benchmarks ─────────────────────────────────────────────────────


class TestRuleBasedConflictBenchmarks:
    def test_detect_conflicts_1k_no_match(self, benchmark, resolver, existing_1k):
        """Rule-based detection with no actual conflicts (best case)."""
        incoming = _make_incoming_unit(scope={"user_id": "user_0"})
        units_in_scope = [u for u in existing_1k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver.detect_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)

    def test_detect_conflicts_10k_no_match(self, benchmark, resolver, existing_10k):
        """Rule-based detection at 10K scope size with no conflicts."""
        incoming = _make_incoming_unit(scope={"user_id": "user_0"})
        units_in_scope = [u for u in existing_10k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver.detect_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)

    def test_detect_conflicts_1k_with_negation(self, benchmark, resolver, existing_1k):
        """Rule-based detection when incoming unit contains negation patterns."""
        incoming = _make_incoming_unit(
            scope={"user_id": "user_0"}, negation_prefix=True
        )
        units_in_scope = [u for u in existing_1k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver.detect_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)


# ── Semantic conflict benchmarks ──────────────────────────────────────────────


class TestSemanticConflictBenchmarks:
    def test_detect_semantic_1k_no_match(self, benchmark, resolver, existing_1k):
        """Semantic detection at 1K with no high-similarity pairs."""
        incoming = _make_incoming_unit(scope={"user_id": "user_0"})
        units_in_scope = [u for u in existing_1k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver.detect_semantic_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)

    def test_detect_semantic_10k_no_match(self, benchmark, resolver, existing_10k):
        """Semantic detection at 10K with no high-similarity pairs."""
        incoming = _make_incoming_unit(scope={"user_id": "user_0"})
        units_in_scope = [u for u in existing_10k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver.detect_semantic_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)

    def test_detect_semantic_1k_with_similar(self, benchmark, resolver, existing_1k):
        """Semantic detection when a match exists — exercises conflict resolution."""
        scope = {"user_id": "user_0"}
        incoming = _make_incoming_unit(scope=scope)
        # Inject one unit that is highly similar to the incoming unit
        similar = _make_similar_unit(incoming.embedding, scope=scope)
        units_in_scope = [u for u in existing_1k if u.scope.get("user_id") == "user_0"]
        units_in_scope.append(similar)
        result = benchmark(resolver.detect_semantic_conflicts, incoming, units_in_scope)
        # The similar unit should be found
        assert any(u.id == similar.id for u in result)

    def test_detect_semantic_threshold_0_5_wider_net(self, benchmark, existing_1k):
        """Lower threshold = more candidates found, measures scaling of that path."""
        resolver_wide = ConflictResolver(strategy="LAST_WRITE_WINS", semantic_threshold=0.50)
        scope = {"user_id": "user_0"}
        incoming = _make_incoming_unit(scope=scope)
        units_in_scope = [u for u in existing_1k if u.scope.get("user_id") == "user_0"]
        result = benchmark(resolver_wide.detect_semantic_conflicts, incoming, units_in_scope)
        assert isinstance(result, list)
