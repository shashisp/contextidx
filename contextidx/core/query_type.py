"""Query-type detection for dynamic scoring weight adjustment.

Classifies a retrieval query into one of three types and returns a weight
override dict that tilts the scoring engine toward the signals most useful
for that query class.

Query types
-----------
``temporal``
    Queries that ask *when* something happened or the chronological order of
    events ("when did …", "before …", "after …", "most recent …").
    Recency and decay signals are boosted; semantic weight is reduced slightly.

``multi_hop``
    Queries that require linking multiple facts ("what is the relationship
    between X and Y", "how does … relate to …", "which … connects to …").
    The default weights work well; a mild semantic boost is applied.

``factual``
    Simple look-up questions that are best answered by direct vector
    similarity ("what is X's job", "who is …", "what does … prefer").
    Semantic weight is boosted and recency / decay are reduced because
    the fact itself is more important than its age.

Usage::

    from contextidx.core.query_type import detect_query_type, weights_for_query

    qtype = detect_query_type("When did Alice join the team?")
    # → "temporal"
    overrides = weights_for_query(qtype)
    # → {"recency": 0.35, "decay": 0.15, "semantic": 0.25, …}
"""

from __future__ import annotations

import re

QueryType = str  # "temporal" | "multi_hop" | "factual"

# ── Temporal cues ────────────────────────────────────────────────────────────

_TEMPORAL_PATTERNS = [
    re.compile(r"\bwhen\b", re.IGNORECASE),
    re.compile(r"\bbefore\b", re.IGNORECASE),
    re.compile(r"\bafter\b", re.IGNORECASE),
    re.compile(r"\bmost recent(ly)?\b", re.IGNORECASE),
    re.compile(r"\blatest\b", re.IGNORECASE),
    re.compile(r"\bearlier\b", re.IGNORECASE),
    re.compile(r"\blast (time|week|month|year|session)\b", re.IGNORECASE),
    re.compile(r"\bused to\b", re.IGNORECASE),
    re.compile(r"\boriginally\b", re.IGNORECASE),
    re.compile(r"\brecently\b", re.IGNORECASE),
    re.compile(r"\bhistor(y|ically)\b", re.IGNORECASE),
    re.compile(r"\btimeline\b", re.IGNORECASE),
    re.compile(r"\bchronolog\b", re.IGNORECASE),
    re.compile(r"\bfirst time\b", re.IGNORECASE),
    re.compile(r"\bdate\b", re.IGNORECASE),
]

# ── Multi-hop / relational cues ───────────────────────────────────────────────

_MULTI_HOP_PATTERNS = [
    re.compile(r"\brelationship between\b", re.IGNORECASE),
    re.compile(r"\brelates? to\b", re.IGNORECASE),
    re.compile(r"\bconnect(s|ed)?\b", re.IGNORECASE),
    re.compile(r"\bhow does .+ relate\b", re.IGNORECASE),
    re.compile(r"\bwhy did\b", re.IGNORECASE),
    re.compile(r"\btransit(ion|ively)\b", re.IGNORECASE),
    re.compile(r"\bchain\b", re.IGNORECASE),
    re.compile(r"\bthrough\b.+\band\b", re.IGNORECASE),
    re.compile(r"\bboth\b", re.IGNORECASE),
]

# ── Weight presets ────────────────────────────────────────────────────────────

# Temporal: boost recency + decay, reduce semantic slightly
_TEMPORAL_WEIGHTS: dict[str, float] = {
    "semantic": 0.25,
    "bm25": 0.08,
    "recency": 0.35,
    "confidence": 0.15,
    "decay": 0.12,
    "reinforcement": 0.05,
}

# Multi-hop: small semantic boost; keep other defaults mostly intact
_MULTI_HOP_WEIGHTS: dict[str, float] = {
    "semantic": 0.35,
    "bm25": 0.10,
    "recency": 0.20,
    "confidence": 0.18,
    "decay": 0.12,
    "reinforcement": 0.05,
}

# Factual / single-hop: strong semantic, deprioritise recency + decay
_FACTUAL_WEIGHTS: dict[str, float] = {
    "semantic": 0.45,
    "bm25": 0.12,
    "recency": 0.15,
    "confidence": 0.18,
    "decay": 0.07,
    "reinforcement": 0.03,
}


def detect_query_type(query: str) -> QueryType:
    """Return the query type for *query* using lightweight pattern matching.

    Checks for temporal cues first (most distinctive), then multi-hop cues.
    Falls back to ``"factual"`` when neither pattern set matches.

    The classifier deliberately errs on the side of ``"factual"`` — temporal
    and multi-hop false-positives are more harmful to retrieval accuracy than
    false-negatives.
    """
    temporal_hits = sum(1 for p in _TEMPORAL_PATTERNS if p.search(query))
    if temporal_hits >= 2 or (temporal_hits == 1 and len(query.split()) <= 10):
        return "temporal"

    multi_hop_hits = sum(1 for p in _MULTI_HOP_PATTERNS if p.search(query))
    if multi_hop_hits >= 1:
        return "multi_hop"

    return "factual"


def weights_for_query(query_type: QueryType) -> dict[str, float]:
    """Return the scoring weight overrides for *query_type*.

    Returns a copy so callers can safely modify the dict.
    """
    if query_type == "temporal":
        return dict(_TEMPORAL_WEIGHTS)
    if query_type == "multi_hop":
        return dict(_MULTI_HOP_WEIGHTS)
    return dict(_FACTUAL_WEIGHTS)
