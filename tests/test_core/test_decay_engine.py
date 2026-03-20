import math
from datetime import datetime, timedelta, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.core.decay_engine import DecayEngine


def _make_unit(
    decay_model: str = "exponential",
    confidence: float = 1.0,
    decay_rate: float = 0.023,
    age_days: float = 0,
    expires_at=None,
) -> ContextUnit:
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    return ContextUnit(
        content="test",
        decay_model=decay_model,
        confidence=confidence,
        decay_rate=decay_rate,
        timestamp=ts,
        expires_at=expires_at,
    )


class TestExponentialDecay:
    def test_no_decay_at_creation(self):
        engine = DecayEngine()
        unit = _make_unit(age_days=0)
        score = engine.compute_decay(unit, unit.timestamp)
        assert abs(score - 1.0) < 1e-6

    def test_half_life(self):
        engine = DecayEngine()
        rate = math.log(2) / 30.0
        unit = _make_unit(decay_rate=rate, age_days=30)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert abs(score - 0.5) < 0.05

    def test_approaches_zero(self):
        engine = DecayEngine()
        unit = _make_unit(age_days=365)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score < 0.01

    def test_future_time_returns_confidence(self):
        engine = DecayEngine()
        unit = _make_unit(age_days=10)
        past = unit.timestamp - timedelta(days=1)
        score = engine.compute_decay(unit, past)
        assert score == unit.confidence


class TestLinearDecay:
    def test_no_decay_at_creation(self):
        engine = DecayEngine()
        unit = _make_unit(decay_model="linear", age_days=0)
        score = engine.compute_decay(unit, unit.timestamp)
        assert abs(score - 1.0) < 1e-6

    def test_half_life_is_half(self):
        engine = DecayEngine()
        rate = math.log(2) / 30.0
        unit = _make_unit(decay_model="linear", decay_rate=rate, age_days=15)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert abs(score - 0.5) < 0.05

    def test_zero_at_full_life(self):
        engine = DecayEngine()
        rate = math.log(2) / 30.0
        unit = _make_unit(decay_model="linear", decay_rate=rate, age_days=30)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score < 0.05

    def test_clamps_to_zero(self):
        engine = DecayEngine()
        unit = _make_unit(decay_model="linear", age_days=365)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score == 0.0


class TestStepDecay:
    def test_before_expiry(self):
        engine = DecayEngine()
        expires = datetime.now(timezone.utc) + timedelta(days=1)
        unit = _make_unit(decay_model="step", expires_at=expires)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score == unit.confidence

    def test_after_expiry(self):
        engine = DecayEngine()
        expires = datetime.now(timezone.utc) - timedelta(days=1)
        unit = _make_unit(decay_model="step", expires_at=expires)
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score == 0.0

    def test_no_expiry_returns_confidence(self):
        engine = DecayEngine()
        unit = _make_unit(decay_model="step")
        now = datetime.now(timezone.utc)
        score = engine.compute_decay(unit, now)
        assert score == unit.confidence


class TestReinforcement:
    def test_reinforcement_reduces_decay(self):
        engine = DecayEngine()
        unit = _make_unit(age_days=60)
        now = datetime.now(timezone.utc)
        base = engine.compute_decay(unit, now, reinforcement_count=0)
        reinforced = engine.compute_decay(unit, now, reinforcement_count=3)
        assert reinforced > base

    def test_zero_reinforcement_no_effect(self):
        engine = DecayEngine()
        unit = _make_unit(age_days=30)
        now = datetime.now(timezone.utc)
        s1 = engine.compute_decay(unit, now, reinforcement_count=0)
        s2 = engine.compute_decay(unit, now, reinforcement_count=0)
        assert s1 == s2
