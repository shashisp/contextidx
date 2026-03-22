from datetime import datetime, timedelta, timezone

import pytest

from contextidx.core.conflict_resolver import ConflictResolver, MergeStrategy
from contextidx.core.context_unit import ContextUnit


def _unit(
    content: str,
    scope: dict | None = None,
    confidence: float = 0.8,
    days_ago: int = 0,
) -> ContextUnit:
    return ContextUnit(
        content=content,
        scope=scope or {"user_id": "u1"},
        confidence=confidence,
        timestamp=datetime.now(timezone.utc) - timedelta(days=days_ago),
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


class TestMergeConcat:
    def test_concat_is_default(self):
        resolver = ConflictResolver(strategy="MERGE")
        new = _unit("A", confidence=0.8)
        old = _unit("B", confidence=0.8)
        result = resolver.resolve(new, [old])
        assert "[MERGED:" in result.winner.content
        assert result.needs_review

    def test_concat_explicit(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="CONCAT")
        new = _unit("New fact", confidence=0.8)
        old = _unit("Old fact", confidence=0.8)
        result = resolver.resolve(new, [old])
        assert "[MERGED:" in result.winner.content
        assert result.winner.confidence < 0.8


class TestMergeRecencyWeighted:
    def test_new_content_wins(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED")
        new = _unit("Senior engineer at StartupXYZ", confidence=0.85, days_ago=2)
        old = _unit("Data analyst at TechCorp", confidence=0.9, days_ago=180)
        result = resolver.resolve(new, [old])
        assert result.winner.content == "Senior engineer at StartupXYZ"
        assert "[MERGED:" not in result.winner.content

    def test_confidence_boosted(self):
        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="RECENCY_WEIGHTED", confidence_boost=0.2
        )
        new = _unit("New diet", confidence=0.85)
        old = _unit("Old diet", confidence=0.9)
        result = resolver.resolve(new, [old])
        assert result.winner.confidence == 1.0

    def test_confidence_boost_from_multiple_old_units(self):
        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="RECENCY_WEIGHTED", confidence_boost=0.1
        )
        new = _unit("Mediterranean diet", confidence=0.8)
        old1 = _unit("Keto diet", confidence=0.7)
        old2 = _unit("Vegetarian", confidence=0.6)
        result = resolver.resolve(new, [old1, old2])
        assert abs(result.winner.confidence - 0.93) < 0.01

    def test_no_needs_review(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED")
        new = _unit("A")
        old = _unit("B")
        result = resolver.resolve(new, [old])
        assert not result.needs_review

    def test_supersedes_all(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED")
        new = _unit("A")
        old1 = _unit("B")
        old2 = _unit("C")
        result = resolver.resolve(new, [old1, old2])
        assert len(result.superseded) == 3

    def test_new_id_generated(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="RECENCY_WEIGHTED")
        new = _unit("A")
        old = _unit("B")
        result = resolver.resolve(new, [old])
        assert result.winner.id != new.id
        assert result.winner.id != old.id


class TestMergeLLMSummarized:
    async def test_llm_merge_calls_merge_fn(self):
        async def mock_merge_fn(new_content: str, old_contents: list[str]) -> str:
            return f"Updated: {new_content} (previously: {', '.join(old_contents)})"

        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=mock_merge_fn,
        )
        new = _unit("Senior engineer", confidence=0.85)
        old = _unit("Data analyst", confidence=0.9)
        result = await resolver.aresolve(new, [old])
        assert "Updated: Senior engineer" in result.winner.content
        assert "previously: Data analyst" in result.winner.content
        assert not result.needs_review

    async def test_llm_merge_preserves_confidence(self):
        async def mock_merge_fn(new_content: str, old_contents: list[str]) -> str:
            return "merged"

        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=mock_merge_fn,
        )
        new = _unit("A", confidence=0.85)
        old = _unit("B", confidence=0.9)
        result = await resolver.aresolve(new, [old])
        assert result.winner.confidence == 0.85

    async def test_falls_back_on_no_fn(self):
        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=None,
        )
        new = _unit("New fact", confidence=0.85)
        old = _unit("Old fact", confidence=0.9)
        result = await resolver.aresolve(new, [old])
        assert result.winner.content == "New fact"
        assert not result.needs_review

    async def test_falls_back_on_exception(self):
        async def failing_merge_fn(new_content: str, old_contents: list[str]) -> str:
            raise RuntimeError("LLM API error")

        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=failing_merge_fn,
        )
        new = _unit("New fact", confidence=0.85)
        old = _unit("Old fact", confidence=0.9)
        result = await resolver.aresolve(new, [old])
        assert result.winner.content == "New fact"
        assert not result.needs_review

    async def test_sync_resolve_falls_back(self):
        async def mock_merge_fn(new_content: str, old_contents: list[str]) -> str:
            return "should not be called"

        resolver = ConflictResolver(
            strategy="MERGE", merge_strategy="LLM_SUMMARIZED", merge_fn=mock_merge_fn,
        )
        new = _unit("New", confidence=0.8)
        old = _unit("Old", confidence=0.7)
        result = resolver.resolve(new, [old])
        assert result.winner.content == "New"

    async def test_aresolve_delegates_non_merge(self):
        resolver = ConflictResolver(strategy="LAST_WRITE_WINS")
        new = _unit("A")
        old = _unit("B")
        result = await resolver.aresolve(new, [old])
        assert result.winner.id == new.id

    async def test_aresolve_empty_conflicts(self):
        resolver = ConflictResolver(strategy="MERGE", merge_strategy="LLM_SUMMARIZED")
        new = _unit("A")
        result = await resolver.aresolve(new, [])
        assert result.winner.id == new.id
        assert not result.needs_review
