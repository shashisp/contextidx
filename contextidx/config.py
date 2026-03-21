"""ContextIdxConfig — centralised home for every tunable constant.

Pass a ``ContextIdxConfig`` instance to ``ContextIdx`` to override any default
without touching individual sub-system constructors.

Example::

    from contextidx import ContextIdx
    from contextidx.config import ContextIdxConfig

    cfg = ContextIdxConfig(
        wal_retention_hours=12,
        overfetch_factor=5,
        consolidation_threshold=0.88,
    )
    idx = ContextIdx(backend=..., config=cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextIdxConfig:
    """Tunable constants for the contextidx runtime.

    All values have sensible defaults that match the previous hard-coded
    behaviour, so existing code continues to work without changes.
    """

    # ── Scoring ──────────────────────────────────────────────────────────────

    recency_half_life_days: float = 30.0
    """Half-life for the exponential recency signal (days).
    A unit created ``recency_half_life_days`` ago scores 0.5 on recency."""

    reinforcement_saturation: int = 20
    """Reinforcement count at which the log-saturation signal reaches 1.0.
    Higher values require more reinforcement hits to reach full score."""

    scoring_weights: dict[str, float] = field(default_factory=dict)
    """Override default scoring weights.  Any key not supplied keeps its
    default value.  Valid keys: semantic, bm25, recency, confidence, decay,
    reinforcement."""

    # ── Retrieval ────────────────────────────────────────────────────────────

    overfetch_factor: int = 3
    """Multiplier applied to ``top_k`` when fetching from the vector backend.
    A higher value gives the scoring engine more candidates to re-rank at the
    cost of slightly more backend latency."""

    graph_expansion_default_score: float = 0.5
    """Composite score assigned to units that enter the result set via graph
    expansion (RELATES_TO edges) rather than vector search.  Set lower to
    deprioritise graph-expanded results relative to direct hits."""

    # ── Consolidation ────────────────────────────────────────────────────────

    consolidation_threshold: float = 0.92
    """Cosine similarity threshold above which two units are considered
    redundant and eligible for merging."""

    max_lineage_depth: int = 10
    """Maximum VERSION_OF chain length before a lineage-summarisation is
    triggered.  Longer chains are a symptom of runaway versioning."""

    # ── WAL ──────────────────────────────────────────────────────────────────

    wal_retention_hours: int = 24
    """How long *applied* WAL entries are retained before compaction.
    Lower values reduce SQLite bloat; higher values extend the audit window."""

    wal_max_age_hours: int = 48
    """Maximum age for *pending* (unapplied) WAL entries.  Entries older than
    this are considered unrecoverable and dropped with a warning during the
    WAL compaction tick.  Set to 0 to disable age-based dropping."""

    max_wal_entries: int = 10_000
    """Circuit-breaker threshold: ``astore()`` raises ``BackendError`` when
    the number of pending WAL entries reaches this limit, protecting the
    system from unbounded WAL growth when the backend is unavailable for
    extended periods.  Set to 0 to disable the circuit-breaker."""
