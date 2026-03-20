"""Pure-Python fallback implementations of the Rust-accelerated hot paths.

These mirror the exact signatures and semantics of the functions in
``src/lib.rs`` so that ``contextidx._core`` can transparently swap between
Rust and Python at import time.
"""

from __future__ import annotations

import math
import re


def batch_decay(
    confidences: list[float],
    decay_rates: list[float],
    age_days: list[float],
    reinforcement_counts: list[int],
    model: str,
    reinforcement_factor: float = 0.5,
) -> list[float]:
    """Batch-compute decay scores for N units."""
    n = len(confidences)
    out: list[float] = []
    for i in range(n):
        effective_age = age_days[i]
        if reinforcement_counts[i] > 0:
            effective_age = age_days[i] * (reinforcement_factor ** reinforcement_counts[i])

        if model == "exponential":
            score = confidences[i] * math.exp(-decay_rates[i] * effective_age)
        elif model == "linear":
            half_life = math.log(2) / decay_rates[i] if decay_rates[i] > 0 else float("inf")
            score = confidences[i] * max(0.0, 1.0 - effective_age / half_life)
        elif model == "step":
            half_life = math.log(2) / decay_rates[i] if decay_rates[i] > 0 else float("inf")
            score = confidences[i] if effective_age < half_life else 0.0
        else:
            score = confidences[i]
        out.append(score)
    return out


def batch_score(
    semantic_scores: list[float],
    recency_scores: list[float],
    confidences: list[float],
    decay_scores: list[float],
    reinforcement_scores: list[float],
    bm25_scores: list[float],
    weights: list[float],
) -> list[float]:
    """Batch-compute composite retrieval scores for N candidates.

    ``weights`` order: [semantic, bm25, recency, confidence, decay, reinforcement]
    """
    w_sem, w_bm25, w_rec, w_conf, w_dec, w_reinf = weights
    n = len(semantic_scores)
    out: list[float] = []
    for i in range(n):
        raw = (
            w_sem * semantic_scores[i]
            + w_bm25 * bm25_scores[i]
            + w_rec * recency_scores[i]
            + w_conf * confidences[i]
            + w_dec * decay_scores[i]
            + w_reinf * reinforcement_scores[i]
        )
        out.append(max(0.0, min(1.0, raw)))
    return out


_NEGATION_WORDS = re.compile(r"\bnot\b|\bno\b|\bnever\b|\bdon'?t\b", re.IGNORECASE)

_VERB_PAIRS: list[tuple[str, str]] = [
    ("prefers ", "does not prefer "),
    ("likes ", "does not like "),
    ("wants ", "does not want "),
    ("is a ", "is no longer a "),
]


def detect_contradictions(
    new_content: str,
    existing_contents: list[str],
) -> list[bool]:
    """Batch rule-based contradiction detection.

    Returns a list of bools where ``True`` means a contradiction was
    detected between *new_content* and the corresponding existing content.
    """
    new_lower = new_content.lower()
    new_words = set(new_lower.split())
    new_has_neg = bool(_NEGATION_WORDS.search(new_lower))

    out: list[bool] = []
    for existing in existing_contents:
        ex_lower = existing.lower()

        # Verb-pair check
        found = False
        for pos, neg in _VERB_PAIRS:
            a_pos = pos in new_lower
            b_neg = neg in ex_lower
            if a_pos and b_neg:
                found = True
                break
            a_neg = neg in new_lower
            b_pos = pos in ex_lower
            if a_neg and b_pos:
                found = True
                break
        if found:
            out.append(True)
            continue

        # High overlap + opposite negation
        ex_words = set(ex_lower.split())
        overlap = len(new_words & ex_words)
        max_len = max(len(new_words), len(ex_words))
        if max_len > 0:
            ratio = overlap / max_len
            ex_has_neg = bool(_NEGATION_WORDS.search(ex_lower))
            if ratio > 0.5 and new_has_neg != ex_has_neg:
                out.append(True)
                continue

        out.append(False)
    return out
