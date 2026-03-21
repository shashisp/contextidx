"""Consolidation logic: merge redundant ContextUnits and detect long lineage chains.

Used by the state-path background loop to reduce storage bloat and keep
retrieval fast.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import datetime, timezone
from typing import Any

from contextidx.core.context_unit import ContextUnit
from contextidx.core.temporal_graph import TemporalGraph
from contextidx.utils.math_utils import cosine_similarity

# Type alias for the ANN search function injected from the backend.
# Signature: (embedding, top_k, filters) -> list of (id, score) tuples.
AnnSearchFn = Callable[
    [list[float], int, dict | None],
    Coroutine[Any, Any, list[Any]],  # returns list[SearchResult]
]

_ANN_CANDIDATES = 20  # number of ANN neighbours to fetch per unit


async def find_redundant_pairs(
    units: list[ContextUnit],
    threshold: float = 0.92,
    ann_search_fn: AnnSearchFn | None = None,
) -> list[tuple[str, str]]:
    """Find pairs of units with embedding similarity above *threshold*.

    Both units must have embeddings and share the same scope.
    Returns ``(keeper_id, absorbed_id)`` pairs where the keeper is the
    higher-confidence or more recent unit.

    When *ann_search_fn* is provided the function uses ANN pre-filtering:
    for each unit it fetches the top-``_ANN_CANDIDATES`` approximate nearest
    neighbours from the backend and applies exact cosine similarity only to
    those candidates.  This reduces the worst-case complexity from O(N²) to
    O(N · k) where k = ``_ANN_CANDIDATES``.

    When *ann_search_fn* is ``None`` the original O(N²) exact comparison is
    used (safe for small corpora and always correct).
    """
    if ann_search_fn is not None:
        return await _find_redundant_pairs_ann(units, threshold, ann_search_fn)
    return _find_redundant_pairs_exact(units, threshold)


def _find_redundant_pairs_exact(
    units: list[ContextUnit],
    threshold: float,
) -> list[tuple[str, str]]:
    """O(N²) exact pairwise comparison — used as fallback or for small N."""
    pairs: list[tuple[str, str]] = []
    n = len(units)
    seen: set[frozenset[str]] = set()
    for i in range(n):
        if not units[i].embedding:
            continue
        for j in range(i + 1, n):
            if not units[j].embedding:
                continue
            if units[i].scope != units[j].scope:
                continue
            sim = cosine_similarity(units[i].embedding, units[j].embedding)
            if sim >= threshold:
                pair_key = frozenset({units[i].id, units[j].id})
                if pair_key not in seen:
                    seen.add(pair_key)
                    keeper, absorbed = _pick_keeper(units[i], units[j])
                    pairs.append((keeper.id, absorbed.id))
    return pairs


async def _find_redundant_pairs_ann(
    units: list[ContextUnit],
    threshold: float,
    ann_search_fn: AnnSearchFn,
) -> list[tuple[str, str]]:
    """O(N·k) ANN pre-filtered comparison."""
    unit_by_id = {u.id: u for u in units}
    pairs: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()

    for unit in units:
        if not unit.embedding:
            continue
        # Fetch approximate nearest neighbours from the backend
        try:
            results = await ann_search_fn(
                unit.embedding,
                _ANN_CANDIDATES + 1,  # +1 because the unit itself may appear
                unit.scope or None,
            )
        except Exception:
            continue

        for result in results:
            candidate_id = result.id
            if candidate_id == unit.id:
                continue
            pair_key = frozenset({unit.id, candidate_id})
            if pair_key in seen:
                continue
            candidate = unit_by_id.get(candidate_id)
            if candidate is None or not candidate.embedding:
                continue
            if unit.scope != candidate.scope:
                continue
            sim = cosine_similarity(unit.embedding, candidate.embedding)
            if sim >= threshold:
                seen.add(pair_key)
                keeper, absorbed = _pick_keeper(unit, candidate)
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


