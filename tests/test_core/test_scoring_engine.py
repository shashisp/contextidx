import math
from datetime import datetime, timedelta, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.core.scoring_engine import ScoringEngine


def _make_unit(age_days: float = 0, confidence: float = 0.8) -> ContextUnit:
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    return ContextUnit(content="test", timestamp=ts, confidence=confidence)


class TestScoringEngine:
    def test_default_weights_sum_to_one(self):
        engine = ScoringEngine()
        total = sum(engine.weights.values())
        assert math.isclose(total, 1.0)

    def test_custom_weights_normalized(self):
        engine = ScoringEngine(weights={"semantic": 1.0, "recency": 1.0})
        total = sum(engine.weights.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)

    def test_perfect_signals_high_score(self):
        engine = ScoringEngine()
        unit = _make_unit(age_days=0, confidence=1.0)
        now = datetime.now(timezone.utc)
        score = engine.compute_score(
            unit=unit,
            semantic_score=1.0,
            query_time=now,
            decay_score=1.0,
            reinforcement_count=20,
        )
        assert score > 0.9

    def test_all_zero_signals(self):
        engine = ScoringEngine()
        unit = _make_unit(age_days=365, confidence=0.0)
        now = datetime.now(timezone.utc)
        score = engine.compute_score(
            unit=unit,
            semantic_score=0.0,
            query_time=now,
            decay_score=0.0,
            reinforcement_count=0,
        )
        assert score < 0.05

    def test_recent_beats_old(self):
        engine = ScoringEngine()
        now = datetime.now(timezone.utc)

        recent = _make_unit(age_days=1, confidence=0.8)
        old = _make_unit(age_days=100, confidence=0.8)

        score_recent = engine.compute_score(recent, 0.8, now, 0.8, 0)
        score_old = engine.compute_score(old, 0.8, now, 0.3, 0)
        assert score_recent > score_old

    def test_score_clamped_to_0_1(self):
        engine = ScoringEngine()
        unit = _make_unit(confidence=1.0)
        now = datetime.now(timezone.utc)
        score = engine.compute_score(unit, 1.5, now, 1.5, 100)
        assert 0.0 <= score <= 1.0
