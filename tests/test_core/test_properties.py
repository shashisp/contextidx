"""Property-based tests for decay and scoring (hypothesis).

Verifies invariants that must hold for all valid inputs:
- Composite score is always in [0, 1].
- Decay score is always in [0, confidence].
- Recency signal is monotonically decreasing with age.
- Score is monotonically increasing with semantic similarity (other signals equal).
- Score is monotonically increasing with confidence (other signals equal).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_unit(
    confidence: float,
    decay_rate: float,
    age_days: float,
    decay_model: str = "exponential",
) -> ContextUnit:
    now = datetime.now(timezone.utc)
    u = ContextUnit(
        id=generate_unit_id(),
        content="property test unit",
        scope={"user_id": "u1"},
        confidence=confidence,
        source="test",
        decay_model=decay_model,
        decay_rate=decay_rate,
    )
    u.timestamp = now - timedelta(days=age_days)
    return u


def _score(
    unit: ContextUnit,
    semantic: float,
    bm25: float | None = None,
    half_life_days: float = 30.0,
    reinforcement_count: int = 0,
) -> float:
    engine = ScoringEngine(recency_half_life_days=half_life_days)
    decay_engine = DecayEngine()
    now = datetime.now(timezone.utc)
    decay = decay_engine.compute_decay(unit, now)
    return engine.compute_score(
        unit=unit,
        semantic_score=semantic,
        query_time=now,
        decay_score=decay,
        reinforcement_count=reinforcement_count,
        bm25_score=bm25,
    )


# ── scoring invariants ────────────────────────────────────────────────────────


class TestScoringInvariants:
    """Composite score must always be in [0, 1]."""

    @given(
        confidence=st.floats(0.0, 1.0),
        semantic=st.floats(0.0, 1.0),
        age_days=st.floats(0.0, 3650.0),
        decay_rate=st.floats(0.001, 1.0),
        reinforcement=st.integers(0, 100),
    )
    @settings(max_examples=300)
    def test_composite_score_in_unit_interval(
        self, confidence, semantic, age_days, decay_rate, reinforcement
    ):
        assume(math.isfinite(confidence))
        assume(math.isfinite(semantic))
        assume(math.isfinite(age_days))
        assume(math.isfinite(decay_rate))
        unit = _make_unit(confidence, decay_rate, age_days)
        score = _score(unit, semantic, reinforcement_count=reinforcement)
        assert 0.0 <= score <= 1.0, (
            f"Score {score} out of [0,1] for conf={confidence}, sem={semantic}, "
            f"age={age_days}, rate={decay_rate}, rein={reinforcement}"
        )

    @given(
        confidence=st.floats(0.0, 1.0),
        semantic=st.floats(0.0, 1.0),
        bm25=st.floats(0.0, 1.0),
        age_days=st.floats(0.0, 365.0),
        decay_rate=st.floats(0.001, 1.0),
    )
    @settings(max_examples=200)
    def test_composite_score_with_bm25_in_unit_interval(
        self, confidence, semantic, bm25, age_days, decay_rate
    ):
        assume(all(math.isfinite(x) for x in [confidence, semantic, bm25, age_days, decay_rate]))
        unit = _make_unit(confidence, decay_rate, age_days)
        score = _score(unit, semantic, bm25=bm25)
        assert 0.0 <= score <= 1.0

    @given(
        half_life=st.floats(1.0, 365.0),
        age_days=st.floats(0.0, 1000.0),
    )
    @settings(max_examples=100)
    def test_composite_score_invariant_across_half_life_values(self, half_life, age_days):
        assume(math.isfinite(half_life) and math.isfinite(age_days))
        unit = _make_unit(0.8, 0.02, age_days)
        score = _score(unit, 0.7, half_life_days=half_life)
        assert 0.0 <= score <= 1.0


class TestScoringMonotonicity:
    """Directional invariants: more signal → higher score."""

    @given(
        sem_low=st.floats(0.0, 0.49),
        sem_high=st.floats(0.51, 1.0),
        confidence=st.floats(0.4, 0.9),
        age_days=st.floats(0.0, 30.0),
    )
    @settings(max_examples=150)
    def test_higher_semantic_score_increases_composite(
        self, sem_low, sem_high, confidence, age_days
    ):
        assume(sem_low < sem_high)
        assume(all(math.isfinite(x) for x in [sem_low, sem_high, confidence, age_days]))
        unit = _make_unit(confidence, 0.02, age_days)
        low = _score(unit, sem_low)
        high = _score(unit, sem_high)
        assert low <= high, (
            f"Expected score({sem_low})={low} <= score({sem_high})={high}"
        )

    @given(
        conf_low=st.floats(0.1, 0.49),
        conf_high=st.floats(0.51, 0.99),
        semantic=st.floats(0.3, 0.9),
        age_days=st.floats(0.0, 30.0),
    )
    @settings(max_examples=150)
    def test_higher_confidence_increases_composite(
        self, conf_low, conf_high, semantic, age_days
    ):
        assume(conf_low < conf_high)
        assume(all(math.isfinite(x) for x in [conf_low, conf_high, semantic, age_days]))
        unit_low = _make_unit(conf_low, 0.02, age_days)
        unit_high = _make_unit(conf_high, 0.02, age_days)
        low = _score(unit_low, semantic)
        high = _score(unit_high, semantic)
        assert low <= high, (
            f"Expected score(conf={conf_low})={low} <= score(conf={conf_high})={high}"
        )


# ── decay invariants ──────────────────────────────────────────────────────────


class TestDecayInvariants:
    """Decay score must be non-negative and at most the initial confidence."""

    @given(
        confidence=st.floats(0.0, 1.0),
        decay_rate=st.floats(0.001, 2.0),
        age_days=st.floats(0.0, 3650.0),
    )
    @settings(max_examples=300)
    def test_exponential_decay_non_negative(self, confidence, decay_rate, age_days):
        assume(all(math.isfinite(x) for x in [confidence, decay_rate, age_days]))
        unit = _make_unit(confidence, decay_rate, age_days, decay_model="exponential")
        decay_engine = DecayEngine()
        score = decay_engine.compute_decay(unit, datetime.now(timezone.utc))
        assert score >= 0.0, f"Decay score {score} is negative"

    @given(
        confidence=st.floats(0.0, 1.0),
        decay_rate=st.floats(0.001, 2.0),
        age_days=st.floats(0.0, 3650.0),
    )
    @settings(max_examples=300)
    def test_exponential_decay_at_most_confidence(self, confidence, decay_rate, age_days):
        assume(all(math.isfinite(x) for x in [confidence, decay_rate, age_days]))
        unit = _make_unit(confidence, decay_rate, age_days, decay_model="exponential")
        decay_engine = DecayEngine()
        score = decay_engine.compute_decay(unit, datetime.now(timezone.utc))
        # Decay can only reduce the confidence, never increase it
        assert score <= confidence + 1e-9, (
            f"Decay score {score} exceeds confidence {confidence}"
        )

    @given(
        confidence=st.floats(0.01, 1.0),
        decay_rate=st.floats(0.001, 0.5),
        age_young=st.floats(0.0, 30.0),
        age_old=st.floats(60.0, 365.0),
    )
    @settings(max_examples=200)
    def test_older_units_decay_more(self, confidence, decay_rate, age_young, age_old):
        """An older unit must have a decay score ≤ a younger unit."""
        assume(all(math.isfinite(x) for x in [confidence, decay_rate, age_young, age_old]))
        assume(age_young < age_old)
        unit_young = _make_unit(confidence, decay_rate, age_young, decay_model="exponential")
        unit_old = _make_unit(confidence, decay_rate, age_old, decay_model="exponential")
        decay_engine = DecayEngine()
        now = datetime.now(timezone.utc)
        score_young = decay_engine.compute_decay(unit_young, now)
        score_old = decay_engine.compute_decay(unit_old, now)
        assert score_old <= score_young + 1e-9, (
            f"Expected older decay ({score_old}) <= younger decay ({score_young})"
        )

    @given(
        confidence=st.floats(0.0, 1.0),
        decay_rate=st.floats(0.001, 2.0),
        age_days=st.floats(0.0, 3650.0),
    )
    @settings(max_examples=200)
    def test_linear_decay_non_negative(self, confidence, decay_rate, age_days):
        assume(all(math.isfinite(x) for x in [confidence, decay_rate, age_days]))
        unit = _make_unit(confidence, decay_rate, age_days, decay_model="linear")
        decay_engine = DecayEngine()
        score = decay_engine.compute_decay(unit, datetime.now(timezone.utc))
        assert score >= 0.0, f"Linear decay score {score} is negative"


# ── recency invariants ────────────────────────────────────────────────────────


class TestRecencyInvariants:
    """Recency signal is monotonically non-increasing with age."""

    @given(
        age_young=st.floats(0.0, 14.0),
        age_old=st.floats(30.0, 365.0),
        half_life=st.floats(7.0, 90.0),
    )
    @settings(max_examples=200)
    def test_recency_decreases_with_age(self, age_young, age_old, half_life):
        assume(age_young < age_old)
        assume(all(math.isfinite(x) for x in [age_young, age_old, half_life]))
        now = datetime.now(timezone.utc)
        recency_young = ScoringEngine._recency_score(
            now - timedelta(days=age_young), now, half_life
        )
        recency_old = ScoringEngine._recency_score(
            now - timedelta(days=age_old), now, half_life
        )
        assert recency_old <= recency_young + 1e-9, (
            f"recency(age={age_old})={recency_old} > recency(age={age_young})={recency_young}"
        )

    @given(age_days=st.floats(0.0, 3650.0))
    @settings(max_examples=100)
    def test_recency_score_in_unit_interval(self, age_days):
        assume(math.isfinite(age_days))
        now = datetime.now(timezone.utc)
        recency = ScoringEngine._recency_score(now - timedelta(days=age_days), now, 30.0)
        assert 0.0 <= recency <= 1.0 + 1e-9
