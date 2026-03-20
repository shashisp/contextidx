from __future__ import annotations

import math
import re
from enum import Enum
from typing import Literal

from contextidx.core.context_unit import ContextUnit, generate_unit_id

_VERB_PAIRS: list[tuple[str, str]] = [
    (r"\bprefers\s+", r"\bdoes\s+not\s+prefer\s+"),
    (r"\blikes\s+", r"\bdoes\s+not\s+like\s+"),
    (r"\bwants\s+", r"\bdoes\s+not\s+want\s+"),
    (r"\bis\s+a\s+", r"\bis\s+no\s+longer\s+a\s+"),
]


class ConflictStrategy(str, Enum):
    LAST_WRITE_WINS = "LAST_WRITE_WINS"
    HIGHEST_CONFIDENCE = "HIGHEST_CONFIDENCE"
    MERGE = "MERGE"
    MANUAL = "MANUAL"


class ConflictResult:
    """Outcome of conflict resolution."""

    def __init__(
        self,
        winner: ContextUnit,
        superseded: list[ContextUnit],
        needs_review: bool = False,
    ):
        self.winner = winner
        self.superseded = superseded
        self.needs_review = needs_review


class ConflictResolver:
    """Detects and resolves conflicting context units within the same scope."""

    def __init__(
        self,
        strategy: Literal[
            "LAST_WRITE_WINS", "HIGHEST_CONFIDENCE", "MERGE", "MANUAL"
        ] = "LAST_WRITE_WINS",
        semantic_threshold: float = 0.85,
    ):
        self._strategy = ConflictStrategy(strategy)
        self._semantic_threshold = semantic_threshold

    @property
    def strategy(self) -> ConflictStrategy:
        return self._strategy

    def detect_conflicts(
        self,
        new_unit: ContextUnit,
        existing_units: list[ContextUnit],
    ) -> list[ContextUnit]:
        """Rule-based O(1)-per-unit conflict detection.

        Checks for explicit negation patterns and high keyword overlap
        between the new unit and each existing non-superseded unit in scope.
        """
        conflicts: list[ContextUnit] = []
        for existing in existing_units:
            if existing.is_superseded:
                continue
            if not existing.matches_scope(new_unit.scope):
                continue
            if self._is_contradictory(new_unit.content, existing.content):
                conflicts.append(existing)
        return conflicts

    def resolve(
        self,
        new_unit: ContextUnit,
        conflicting: list[ContextUnit],
    ) -> ConflictResult:
        """Resolve conflicts according to the configured strategy."""
        if not conflicting:
            return ConflictResult(winner=new_unit, superseded=[])

        if self._strategy == ConflictStrategy.LAST_WRITE_WINS:
            return self._resolve_last_write_wins(new_unit, conflicting)
        elif self._strategy == ConflictStrategy.HIGHEST_CONFIDENCE:
            return self._resolve_highest_confidence(new_unit, conflicting)
        elif self._strategy == ConflictStrategy.MERGE:
            return self._resolve_merge(new_unit, conflicting)
        elif self._strategy == ConflictStrategy.MANUAL:
            return self._resolve_manual(new_unit, conflicting)
        else:
            raise ValueError(f"Unknown strategy: {self._strategy}")

    def detect_semantic_conflicts(
        self,
        new_unit: ContextUnit,
        existing_units: list[ContextUnit],
    ) -> list[ContextUnit]:
        """Semantic conflict detection via embedding cosine similarity.

        Units with similarity above ``_semantic_threshold`` that also show
        contradictory negation patterns are flagged as conflicts.
        Both the new unit and each existing unit must have embeddings.
        """
        if not new_unit.embedding:
            return []

        conflicts: list[ContextUnit] = []
        for existing in existing_units:
            if existing.is_superseded:
                continue
            if not existing.matches_scope(new_unit.scope):
                continue
            if not existing.embedding:
                continue

            sim = _cosine_similarity(new_unit.embedding, existing.embedding)
            if sim >= self._semantic_threshold:
                if self._is_contradictory(new_unit.content, existing.content):
                    conflicts.append(existing)
        return conflicts

    def detect_tiered(
        self,
        new_unit: ContextUnit,
        existing_units: list[ContextUnit],
    ) -> tuple[list[ContextUnit], list[ContextUnit]]:
        """Two-phase conflict detection: fast rule-based + deferred semantic.

        Returns:
            (inline_conflicts, candidates_for_semantic) — the first list
            should be resolved immediately; the second list should be
            queued for asynchronous semantic resolution.
        """
        inline = self.detect_conflicts(new_unit, existing_units)
        inline_ids = {u.id for u in inline}
        candidates = [
            u for u in existing_units
            if u.id not in inline_ids
            and not u.is_superseded
            and u.matches_scope(new_unit.scope)
        ]
        return inline, candidates

    # ── Resolution strategies ──

    @staticmethod
    def _resolve_last_write_wins(
        new_unit: ContextUnit, conflicting: list[ContextUnit]
    ) -> ConflictResult:
        return ConflictResult(winner=new_unit, superseded=list(conflicting))

    @staticmethod
    def _resolve_highest_confidence(
        new_unit: ContextUnit, conflicting: list[ContextUnit]
    ) -> ConflictResult:
        all_units = [new_unit, *conflicting]
        all_units.sort(key=lambda u: u.confidence, reverse=True)
        winner = all_units[0]
        superseded = [u for u in all_units if u.id != winner.id]
        return ConflictResult(winner=winner, superseded=superseded)

    @staticmethod
    def _resolve_merge(
        new_unit: ContextUnit, conflicting: list[ContextUnit]
    ) -> ConflictResult:
        merged_content = new_unit.content
        for u in conflicting:
            merged_content += f" [MERGED: {u.content}]"

        avg_confidence = (
            new_unit.confidence + sum(u.confidence for u in conflicting)
        ) / (1 + len(conflicting))

        merged = new_unit.model_copy(
            update={
                "id": generate_unit_id(),
                "content": merged_content,
                "confidence": max(0.0, avg_confidence * 0.8),
            }
        )
        return ConflictResult(
            winner=merged,
            superseded=[new_unit, *conflicting],
            needs_review=True,
        )

    @staticmethod
    def _resolve_manual(
        new_unit: ContextUnit, conflicting: list[ContextUnit]
    ) -> ConflictResult:
        return ConflictResult(
            winner=new_unit,
            superseded=[],
            needs_review=True,
        )

    # ── Detection helpers ──

    @staticmethod
    def _is_contradictory(content_a: str, content_b: str) -> bool:
        for pos_pat, neg_pat in _VERB_PAIRS:
            a_pos = re.search(pos_pat, content_a, re.IGNORECASE) is not None
            b_neg = re.search(neg_pat, content_b, re.IGNORECASE) is not None
            if a_pos and b_neg:
                return True
            a_neg = re.search(neg_pat, content_a, re.IGNORECASE) is not None
            b_pos = re.search(pos_pat, content_b, re.IGNORECASE) is not None
            if a_neg and b_pos:
                return True

        a_words = set(content_a.lower().split())
        b_words = set(content_b.lower().split())
        if len(a_words) == 0 or len(b_words) == 0:
            return False
        overlap = len(a_words & b_words)
        max_len = max(len(a_words), len(b_words))
        high_overlap = overlap / max_len > 0.5

        a_has_not = bool(re.search(r"\bnot\b|\bno\b|\bnever\b|\bdon'?t\b", content_a, re.IGNORECASE))
        b_has_not = bool(re.search(r"\bnot\b|\bno\b|\bnever\b|\bdon'?t\b", content_b, re.IGNORECASE))

        return high_overlap and (a_has_not != b_has_not)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
