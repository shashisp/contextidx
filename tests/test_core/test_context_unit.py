from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from contextidx.core.context_unit import ContextUnit, generate_unit_id


class TestGenerateUnitId:
    def test_format(self):
        uid = generate_unit_id()
        assert uid.startswith("ctx_")
        assert len(uid) == 20  # "ctx_" + 16 hex chars

    def test_unique(self):
        ids = {generate_unit_id() for _ in range(100)}
        assert len(ids) == 100


class TestContextUnit:
    def test_defaults(self):
        unit = ContextUnit(content="hello")
        assert unit.content == "hello"
        assert unit.confidence == 0.8
        assert unit.decay_model == "exponential"
        assert unit.version == 1
        assert unit.superseded_by is None
        assert unit.embedding is None
        assert unit.id.startswith("ctx_")

    def test_validation_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ContextUnit(content="x", confidence=-0.1)
        with pytest.raises(ValidationError):
            ContextUnit(content="x", confidence=1.1)

    def test_validation_decay_rate_positive(self):
        with pytest.raises(ValidationError):
            ContextUnit(content="x", decay_rate=0.0)
        with pytest.raises(ValidationError):
            ContextUnit(content="x", decay_rate=-1.0)

    def test_validation_decay_model_enum(self):
        for model in ("exponential", "linear", "step"):
            unit = ContextUnit(content="x", decay_model=model)
            assert unit.decay_model == model
        with pytest.raises(ValidationError):
            ContextUnit(content="x", decay_model="unknown")

    def test_serialization_roundtrip(self):
        unit = ContextUnit(
            content="test content",
            scope={"user_id": "u1"},
            confidence=0.9,
        )
        json_str = unit.model_dump_json()
        restored = ContextUnit.model_validate_json(json_str)
        assert restored.content == unit.content
        assert restored.scope == unit.scope
        assert restored.confidence == unit.confidence
        assert restored.id == unit.id

    def test_is_superseded(self):
        unit = ContextUnit(content="x")
        assert not unit.is_superseded
        unit.superseded_by = "ctx_other"
        assert unit.is_superseded

    def test_is_expired(self):
        past = datetime.now(timezone.utc) - timedelta(hours=1)
        unit = ContextUnit(content="x", expires_at=past)
        assert unit.is_expired

        future = datetime.now(timezone.utc) + timedelta(hours=1)
        unit2 = ContextUnit(content="x", expires_at=future)
        assert not unit2.is_expired

        unit3 = ContextUnit(content="x")
        assert not unit3.is_expired

    def test_matches_scope(self):
        unit = ContextUnit(content="x", scope={"user_id": "u1", "session_id": "s1"})
        assert unit.matches_scope({"user_id": "u1"})
        assert unit.matches_scope({"user_id": "u1", "session_id": "s1"})
        assert not unit.matches_scope({"user_id": "u2"})
        assert unit.matches_scope({})

    def test_decay_rate_from_half_life(self):
        rate = ContextUnit.decay_rate_from_half_life(30.0)
        assert abs(rate - 0.023104906) < 0.001

    def test_age_days_freshly_created(self):
        unit = ContextUnit(content="x")
        assert unit.age_days >= 0.0
        assert unit.age_days < 0.01  # well under 1 second old

    def test_age_days_old_unit(self):
        past = datetime.now(timezone.utc) - timedelta(days=5)
        unit = ContextUnit(content="x", timestamp=past)
        assert 4.9 < unit.age_days < 5.1

    def test_age_days_is_float(self):
        unit = ContextUnit(content="x")
        assert isinstance(unit.age_days, float)
