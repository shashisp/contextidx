"""Consolidation logic: merge redundant ContextUnits and detect long lineage chains.

Used by the state-path background loop to reduce storage bloat and keep
retrieval fast.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.core.temporal_graph import TemporalGraph


def find_redundant_pairs(
    units: list[ContextUnit],
    threshold: float = 0.92,
) -> list[tuple[str, str]]:
    """Find pairs of units with embedding similarity above *threshold*.

    Both units must have embeddings and share the same scope.
    Returns ``(keeper_id, absorbed_id)`` pairs where the keeper is the
    higher-confidence or more recent unit.
    """
    pairs: list[tuple[str, str]] = []
    n = len(units)
    for i in range(n):
        if not units[i].embedding:
            continue
        for j in range(i + 1, n):
            if not units[j].embedding:
                continue
            if units[i].scope != units[j].scope:
                continue
            sim = _cosine_similarity(units[i].embedding, units[j].embedding)
            if sim >= threshold:
                keeper, absorbed = _pick_keeper(units[i], units[j])
                pairs.append((keeper.id, absorbed.id))
    return pairs


def merge_units(
    keeper: ContextUnit,
    absorbed: ContextUnit,
) -> ContextUnit:
    """Produce an updated keeper that incorporates the absorbed unit.

    The keeper's confidence is boosted slightly (average of the two, capped
    at 1.0).  The absorbed unit should be marked as superseded by the
    caller.
    """
    merged_confidence = min(1.0, (keeper.confidence + absorbed.confidence) / 2.0)
    return keeper.model_copy(
        update={"confidence": merged_confidence, "version": keeper.version + 1},
    )


def should_summarize_lineage(
    unit_id: str,
    graph: TemporalGraph,
    max_chain: int = 10,
) -> bool:
    """Return True if the lineage chain for *unit_id* exceeds *max_chain*."""
    chain = graph.get_lineage(unit_id)
    return len(chain) > max_chain


def _pick_keeper(a: ContextUnit, b: ContextUnit) -> tuple[ContextUnit, ContextUnit]:
    """Pick which unit to keep: prefer higher confidence, then more recent."""
    if a.confidence > b.confidence:
        return a, b
    if b.confidence > a.confidence:
        return b, a
    if a.timestamp >= b.timestamp:
        return a, b
    return b, a


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
