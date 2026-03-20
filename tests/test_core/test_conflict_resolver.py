from datetime import datetime, timezone

from contextidx.core.conflict_resolver import ConflictResolver
from contextidx.core.context_unit import ContextUnit


def _unit(content: str, scope: dict | None = None, confidence: float = 0.8) -> ContextUnit:
    return ContextUnit(
        content=content,
        scope=scope or {"user_id": "u1"},
        confidence=confidence,
    )


class TestRuleBasedDetection:
    def test_detects_negation_conflict(self):
        resolver = ConflictResolver()
        new = _unit("User prefers async communication")
        existing = [_unit("User does not prefer async communication")]
        conflicts = resolver.detect_conflicts(new, existing)
        assert len(conflicts) == 1

    def test_no_conflict_different_topics(self):
        resolver = ConflictResolver()
        new = _unit("User likes Python")
        existing = [_unit("User enjoys hiking on weekends")]
        conflicts = resolver.detect_conflicts(new, existing)
        assert len(conflicts) == 0

    def test_skips_superseded_units(self):
        resolver = ConflictResolver()
        new = _unit("User likes cats")
        old = _unit("User does not like cats")
        old.superseded_by = "some_id"
        conflicts = resolver.detect_conflicts(new, [old])
        assert len(conflicts) == 0

    def test_scope_isolation(self):
        resolver = ConflictResolver()
        new = _unit("User likes dogs", scope={"user_id": "u1"})
        other_scope = _unit("User does not like dogs", scope={"user_id": "u2"})
        conflicts = resolver.detect_conflicts(new, [other_scope])
        assert len(conflicts) == 0


class TestLastWriteWins:
    def test_new_unit_wins(self):
        resolver = ConflictResolver(strategy="LAST_WRITE_WINS")
        new = _unit("User prefers X")
        old = _unit("User does not prefer X")
        result = resolver.resolve(new, [old])
        assert result.winner.id == new.id
        assert len(result.superseded) == 1
        assert not result.needs_review


class TestHighestConfidence:
    def test_high_confidence_wins(self):
        resolver = ConflictResolver(strategy="HIGHEST_CONFIDENCE")
        new = _unit("A", confidence=0.6)
        old = _unit("B", confidence=0.95)
        result = resolver.resolve(new, [old])
        assert result.winner.id == old.id
        assert new in result.superseded


class TestMerge:
    def test_creates_merged_unit(self):
        resolver = ConflictResolver(strategy="MERGE")
        new = _unit("A", confidence=0.8)
        old = _unit("B", confidence=0.8)
        result = resolver.resolve(new, [old])
        assert result.winner.id != new.id
        assert result.winner.id != old.id
        assert "[MERGED:" in result.winner.content
        assert result.winner.confidence < 0.8
        assert result.needs_review


class TestManual:
    def test_flags_for_review(self):
        resolver = ConflictResolver(strategy="MANUAL")
        new = _unit("A")
        old = _unit("B")
        result = resolver.resolve(new, [old])
        assert result.needs_review
        assert len(result.superseded) == 0

    def test_no_conflicts_no_review(self):
        resolver = ConflictResolver(strategy="MANUAL")
        new = _unit("A")
        result = resolver.resolve(new, [])
        assert not result.needs_review


class TestSemanticStub:
    def test_returns_empty(self):
        resolver = ConflictResolver()
        new = _unit("A")
        result = resolver.detect_semantic_conflicts(new, [_unit("B")])
        assert result == []
