from __future__ import annotations

import math
from datetime import datetime

from contextidx.core.context_unit import ContextUnit

SECONDS_PER_DAY = 86400.0

DEFAULT_WEIGHTS: dict[str, float] = {
    "semantic": 0.30,
    "bm25": 0.10,
    "recency": 0.25,
    "confidence": 0.20,
    "decay": 0.10,
    "reinforcement": 0.05,
}


class ScoringEngine:
    """Fuses up to six signals into a composite retrieval score.

    When ``bm25_score`` is not supplied at scoring time the BM25 weight is
    redistributed proportionally across the remaining signals, so vector-only
    backends produce identical rankings to the v0.1 engine.
    """

    def __init__(self, weights: dict[str, float] | None = None):
        self._weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self._weights.update(weights)
        self._weights.setdefault("bm25", 0.0)
        total = sum(self._weights.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            for k in self._weights:
                self._weights[k] /= total

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def compute_score(
        self,
        unit: ContextUnit,
        semantic_score: float,
        query_time: datetime,
        decay_score: float,
        reinforcement_count: int,
        bm25_score: float | None = None,
    ) -> float:
        """Compute composite score for a context unit.

        Args:
            unit: The context unit being scored.
            semantic_score: Cosine similarity from vector search, in [0, 1].
            query_time: The time at which retrieval is happening.
            decay_score: Pre-computed decay score from DecayEngine.
            reinforcement_count: Number of times this unit was reinforced.
            bm25_score: Optional BM25 relevance score from hybrid backends, in [0, 1].

        Returns:
            Composite score in [0, 1].
        """
        recency = self._recency_score(unit.timestamp, query_time)
        reinforcement = self._reinforcement_score(reinforcement_count)

        signals = {
            "semantic": semantic_score,
            "recency": recency,
            "confidence": unit.confidence,
            "decay": decay_score,
            "reinforcement": reinforcement,
        }

        if bm25_score is not None:
            signals["bm25"] = bm25_score
            weights = self._weights
        else:
            weights = self._redistribute_without_bm25()

        score = sum(weights.get(k, 0.0) * v for k, v in signals.items())
        return max(0.0, min(1.0, score))

    def _redistribute_without_bm25(self) -> dict[str, float]:
        """Drop the bm25 weight and renormalize the rest to sum to 1.0."""
        remaining = {k: v for k, v in self._weights.items() if k != "bm25"}
        total = sum(remaining.values())
        if total == 0:
            return remaining
        return {k: v / total for k, v in remaining.items()}

    @staticmethod
    def _recency_score(created_at: datetime, query_time: datetime) -> float:
        """Exponential recency with 30-day half-life."""
        age_days = (query_time - created_at).total_seconds() / SECONDS_PER_DAY
        if age_days < 0:
            return 1.0
        return math.exp(-age_days * math.log(2) / 30.0)

    @staticmethod
    def _reinforcement_score(reinforcement_count: int, saturation: int = 20) -> float:
        """Logarithmic reinforcement signal that saturates."""
        if reinforcement_count <= 0:
            return 0.0
        return min(1.0, math.log1p(reinforcement_count) / math.log1p(saturation))

    def batch_compute_score(
        self,
        units: list[ContextUnit],
        semantic_scores: list[float],
        query_time: datetime,
        decay_scores: list[float],
        reinforcement_counts: list[int],
        bm25_scores: list[float] | None = None,
    ) -> list[float]:
        """Compute composite scores for multiple units at once.

        Delegates to the Rust extension when available, otherwise uses the
        pure-Python fallback.
        """
        from contextidx._core import batch_score

        n = len(units)
        if n == 0:
            return []

        recency = [self._recency_score(u.timestamp, query_time) for u in units]
        confidences = [u.confidence for u in units]
        reinforcement = [self._reinforcement_score(rc) for rc in reinforcement_counts]
        bm25 = bm25_scores if bm25_scores is not None else [0.0] * n

        has_bm25 = bm25_scores is not None
        weights = self._weights if has_bm25 else self._redistribute_without_bm25()

        weight_vec = [
            weights.get("semantic", 0.0),
            weights.get("bm25", 0.0),
            weights.get("recency", 0.0),
            weights.get("confidence", 0.0),
            weights.get("decay", 0.0),
            weights.get("reinforcement", 0.0),
        ]

        return batch_score(
            semantic_scores, recency, confidences,
            decay_scores, reinforcement, bm25, weight_vec,
        )
