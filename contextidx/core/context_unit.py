from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def generate_unit_id() -> str:
    return f"ctx_{uuid.uuid4().hex[:16]}"


class ContextUnit(BaseModel):
    """Atomic unit of temporal context storage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=generate_unit_id)
    content: str
    embedding: list[float] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    decay_rate: float = Field(gt=0.0, default=0.023)  # ~30 day half-life for exponential
    decay_model: Literal["exponential", "linear", "step"] = "exponential"
    scope: dict[str, str] = Field(default_factory=dict)
    source: str = "unknown"
    superseded_by: str | None = None
    version: int = 1
    expires_at: datetime | None = None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    @property
    def is_expired(self) -> bool:
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) >= self.expires_at

    def is_expired_at(self, t: datetime) -> bool:
        if self.expires_at is None:
            return False
        return t >= self.expires_at

    def matches_scope(self, scope: dict[str, str]) -> bool:
        """Check if this unit's scope is a superset of the query scope."""
        return all(self.scope.get(k) == v for k, v in scope.items())

    @staticmethod
    def decay_rate_from_half_life(half_life_days: float) -> float:
        """Convert half-life in days to exponential decay rate lambda.

        For exponential decay: d(t) = c * exp(-λ * t)
        Half-life: d(h) = c/2  =>  λ = ln(2) / h
        """
        import math
        return math.log(2) / half_life_days
