from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Literal, Protocol, runtime_checkable

from contextidx.core.context_unit import ContextUnit, generate_unit_id

logger = logging.getLogger("contextidx.conflict_resolver")

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


@runtime_checkable
class ConflictJudgeFn(Protocol):
    """Protocol for LLM-based conflict judges.

    Given two statements from the same scope, return ``True`` if they
    provide *different* answers to the same question/attribute (i.e. one
    should supersede the other).
    """

    async def __call__(self, statement_a: str, statement_b: str) -> bool: ...


@runtime_checkable
class MergeFn(Protocol):
    """Protocol for custom merge functions (e.g. LLM-summarized merges).

    Given the new content and a list of old contents being superseded,
    return a single merged string.
    """

    async def __call__(self, new_content: str, old_contents: list[str]) -> str: ...


class ConflictResolver:
    """Detects and resolves conflicting context units within the same scope."""

    def __init__(
        self,
        strategy: Literal[
            "LAST_WRITE_WINS", "HIGHEST_CONFIDENCE", "MERGE", "MANUAL"
        ] = "LAST_WRITE_WINS",
        semantic_threshold: float = 0.80,
        conflict_judge_fn: ConflictJudgeFn | None = None,
        merge_fn: MergeFn | None = None,
    ):
        self._strategy = ConflictStrategy(strategy)
        self._semantic_threshold = semantic_threshold
        self._judge_fn = conflict_judge_fn
        self._merge_fn = merge_fn

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
        Uses the Rust ``detect_contradictions`` batch kernel when available.
        """
        from contextidx._core import detect_contradictions

        eligible = [
            u for u in existing_units
            if not u.is_superseded and u.matches_scope(new_unit.scope)
        ]
        if not eligible:
            return []

        contents = [u.content for u in eligible]
        flags = detect_contradictions(new_unit.content, contents)
        return [u for u, is_conflict in zip(eligible, flags) if is_conflict]

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

    async def aresolve(
        self,
        new_unit: ContextUnit,
        conflicting: list[ContextUnit],
    ) -> ConflictResult:
        """Async resolve — uses merge_fn when strategy is MERGE and a merge_fn is configured."""
        if not conflicting:
            return ConflictResult(winner=new_unit, superseded=[])

        if self._strategy == ConflictStrategy.MERGE and self._merge_fn is not None:
            return await self._resolve_merge_llm(new_unit, conflicting)

        return self.resolve(new_unit, conflicting)

    def detect_semantic_conflicts(
        self,
        new_unit: ContextUnit,
        existing_units: list[ContextUnit],
    ) -> list[ContextUnit]:
        """Semantic conflict detection via embedding cosine similarity.

        Two units in the same scope with high embedding similarity are
        treated as being about the same topic/attribute.  When that
        happens the newer unit supersedes the older one — no explicit
        negation pattern is required.

        Uses the Rust ``batch_cosine_similarity`` kernel for a single
        PyO3 crossing instead of per-unit Python calls.
        """
        if not new_unit.embedding:
            return []

        from contextidx._core import batch_cosine_similarity

        eligible = [
            u for u in existing_units
            if not u.is_superseded
            and u.matches_scope(new_unit.scope)
            and u.embedding
            and u.id != new_unit.id
        ]
        if not eligible:
            return []

        dim = len(new_unit.embedding)
        flat: list[float] = []
        for u in eligible:
            flat.extend(u.embedding)  # type: ignore[arg-type]

        sims = batch_cosine_similarity(new_unit.embedding, flat, dim)
        return [u for u, sim in zip(eligible, sims) if sim >= self._semantic_threshold]

    async def detect_llm_conflicts(
        self,
        new_unit: ContextUnit,
        existing_units: list[ContextUnit],
    ) -> list[ContextUnit]:
        """LLM-assisted conflict detection for highest accuracy.

        First narrows candidates using embedding similarity (same as
        ``detect_semantic_conflicts``), then asks the configured
        ``_judge_fn`` to confirm each candidate.  If no judge function
        is configured, falls back to pure semantic detection.
        """
        semantic_candidates = self.detect_semantic_conflicts(new_unit, existing_units)
        if not semantic_candidates or self._judge_fn is None:
            return semantic_candidates

        confirmed: list[ContextUnit] = []
        for candidate in semantic_candidates:
            try:
                is_conflict = await self._judge_fn(new_unit.content, candidate.content)
                if is_conflict:
                    confirmed.append(candidate)
            except Exception:
                confirmed.append(candidate)
        return confirmed

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

    def _resolve_merge(
        self,
        new_unit: ContextUnit,
        conflicting: list[ContextUnit],
    ) -> ConflictResult:
        if self._merge_fn is not None:
            logger.warning(
                "merge_fn requires aresolve(); falling back to default "
                "merge in sync resolve()"
            )
        return self._resolve_merge_concat(new_unit, conflicting)

    @staticmethod
    def _resolve_merge_concat(
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

    async def _resolve_merge_llm(
        self,
        new_unit: ContextUnit,
        conflicting: list[ContextUnit],
    ) -> ConflictResult:
        if self._merge_fn is None:
            return self._resolve_merge_concat(new_unit, conflicting)

        old_contents = [u.content for u in conflicting]
        try:
            merged_content = await self._merge_fn(new_unit.content, old_contents)
        except Exception:
            return self._resolve_merge_concat(new_unit, conflicting)

        merged = new_unit.model_copy(
            update={
                "id": generate_unit_id(),
                "content": merged_content,
                "confidence": new_unit.confidence,
            }
        )
        return ConflictResult(
            winner=merged,
            superseded=[new_unit, *conflicting],
            needs_review=False,
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
