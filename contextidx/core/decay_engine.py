from __future__ import annotations

import math
from datetime import datetime

from contextidx.core.context_unit import ContextUnit

SECONDS_PER_DAY = 86400.0


def _prepare_batch_inputs(
    units: list[ContextUnit],
    t: datetime,
    reinforcement_counts: list[int],
) -> tuple[list[float], list[float], list[float], list[int]]:
    """Extract parallel arrays from a list of ContextUnits for batch decay."""
    confidences: list[float] = []
    decay_rates: list[float] = []
    age_days: list[float] = []
    for unit in units:
        confidences.append(unit.confidence)
        decay_rates.append(unit.decay_rate)
        age_seconds = max(0.0, (t - unit.timestamp).total_seconds())
        age_days.append(age_seconds / SECONDS_PER_DAY)
    return confidences, decay_rates, age_days, reinforcement_counts


class DecayEngine:
    """Computes temporal relevance scores for ContextUnits using configurable decay models."""

    def __init__(self, reinforcement_factor: float = 0.5):
        """
        Args:
            reinforcement_factor: How much each reinforcement resets the decay clock.
                0.5 means each reinforcement halves the effective age.
        """
        self._reinforcement_factor = reinforcement_factor

    def compute_decay(
        self,
        unit: ContextUnit,
        t: datetime,
        reinforcement_count: int = 0,
    ) -> float:
        """Compute the decay score d(eᵢ, t) for a unit at time t.

        Returns a value in [0, unit.confidence]. Higher = more relevant.
        """
        age_seconds = (t - unit.timestamp).total_seconds()
        if age_seconds < 0:
            return unit.confidence

        effective_age = self._apply_reinforcement(age_seconds, reinforcement_count)
        age_days = effective_age / SECONDS_PER_DAY

        if unit.decay_model == "exponential":
            return self._exponential(unit.confidence, unit.decay_rate, age_days)
        elif unit.decay_model == "linear":
            return self._linear(unit.confidence, unit.decay_rate, age_days)
        elif unit.decay_model == "step":
            return self._step(unit, t)
        else:
            raise ValueError(f"Unknown decay model: {unit.decay_model}")

    def _apply_reinforcement(self, age_seconds: float, reinforcement_count: int) -> float:
        """Reduce effective age based on reinforcement count."""
        if reinforcement_count <= 0:
            return age_seconds
        return age_seconds * (self._reinforcement_factor ** reinforcement_count)

    @staticmethod
    def _exponential(confidence: float, decay_rate: float, age_days: float) -> float:
        """d = c · exp(-λ · t)"""
        return confidence * math.exp(-decay_rate * age_days)

    @staticmethod
    def _linear(confidence: float, decay_rate: float, age_days: float) -> float:
        """d = c · max(0, 1 - t / h) where h = ln(2)/λ (half-life in days)."""
        half_life = math.log(2) / decay_rate if decay_rate > 0 else float("inf")
        return confidence * max(0.0, 1.0 - age_days / half_life)

    @staticmethod
    def _step(unit: ContextUnit, t: datetime) -> float:
        """d = c if t < expires_at else 0."""
        if unit.expires_at is None:
            return unit.confidence
        return unit.confidence if t < unit.expires_at else 0.0

    def batch_compute_decay(
        self,
        units: list[ContextUnit],
        t: datetime,
        reinforcement_counts: list[int] | None = None,
    ) -> list[float]:
        """Compute decay scores for multiple units at once.

        Delegates to the Rust extension when available, otherwise uses the
        pure-Python fallback.  All units must share the same ``decay_model``.
        """
        from contextidx._core import batch_decay

        if not units:
            return []
        if reinforcement_counts is None:
            reinforcement_counts = [0] * len(units)

        confidences, decay_rates, age_days, rc = _prepare_batch_inputs(
            units, t, reinforcement_counts,
        )
        model = units[0].decay_model
        return batch_decay(
            confidences, decay_rates, age_days, rc, model, self._reinforcement_factor,
        )
