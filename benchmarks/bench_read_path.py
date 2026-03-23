"""Benchmark the read path (aretrieve) at various scales.

Includes per-unit baselines, simple batch, and the production aretrieve
pattern (model-grouped batch decay + bm25-split batch scoring).
"""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine


# ── Per-unit baseline (old aretrieve pattern) ────────────────────────────────

def test_scoring_pipeline_1k(benchmark, units_1k, query_embedding):
    """Measure the scoring-only portion of the read path with 1k candidates."""
    engine = ScoringEngine()
    decay_engine = DecayEngine()
    now = datetime.now(timezone.utc)

    def run():
        for u in units_1k:
            d = decay_engine.compute_decay(u, now)
            engine.compute_score(
                unit=u,
                semantic_score=0.7,
                query_time=now,
                decay_score=d,
                reinforcement_count=0,
            )

    benchmark(run)


# ── Simple batch (no model grouping) ────────────────────────────────────────

def test_scoring_pipeline_batch_1k(benchmark, units_1k, query_embedding):
    """Measure the batch scoring pipeline with 1k candidates."""
    engine = ScoringEngine()
    decay_engine = DecayEngine()
    now = datetime.now(timezone.utc)

    def run():
        decays = decay_engine.batch_compute_decay(units_1k, now)
        sem = [0.7] * len(units_1k)
        rcs = [0] * len(units_1k)
        engine.batch_compute_score(units_1k, sem, now, decays, rcs)

    benchmark(run)


def test_scoring_pipeline_batch_10k(benchmark, units_10k, query_embedding):
    """Measure the batch scoring pipeline with 10k candidates."""
    engine = ScoringEngine()
    decay_engine = DecayEngine()
    now = datetime.now(timezone.utc)

    def run():
        decays = decay_engine.batch_compute_decay(units_10k, now)
        sem = [0.7] * len(units_10k)
        rcs = [0] * len(units_10k)
        engine.batch_compute_score(units_10k, sem, now, decays, rcs)

    benchmark(run)


# ── Production aretrieve pattern (model-grouped decay + bm25-split score) ───

def _aretrieve_scoring(units, sem_scores, bm25_raw, reinforcement_counts, query_time):
    """Mirror the actual aretrieve batch scoring logic."""
    decay_engine = DecayEngine()
    scoring_engine = ScoringEngine()
    n = len(units)

    all_decay = [0.0] * n
    model_groups: dict[str, list[int]] = {}
    for i, u in enumerate(units):
        model_groups.setdefault(u.decay_model, []).append(i)
    for _model, indices in model_groups.items():
        group_units = [units[i] for i in indices]
        group_rcs = [reinforcement_counts[i] for i in indices]
        group_decays = decay_engine.batch_compute_decay(group_units, query_time, group_rcs)
        for idx, ds in zip(indices, group_decays):
            all_decay[idx] = ds

    has_any_bm25 = any(b is not None for b in bm25_raw)
    if not has_any_bm25:
        scoring_engine.batch_compute_score(
            units, sem_scores, query_time, all_decay, reinforcement_counts,
            bm25_scores=None,
        )
    else:
        bm25_idx = [i for i, b in enumerate(bm25_raw) if b is not None]
        no_bm25_idx = [i for i, b in enumerate(bm25_raw) if b is None]
        if bm25_idx:
            scoring_engine.batch_compute_score(
                [units[i] for i in bm25_idx],
                [sem_scores[i] for i in bm25_idx],
                query_time,
                [all_decay[i] for i in bm25_idx],
                [reinforcement_counts[i] for i in bm25_idx],
                bm25_scores=[bm25_raw[i] for i in bm25_idx],
            )
        if no_bm25_idx:
            scoring_engine.batch_compute_score(
                [units[i] for i in no_bm25_idx],
                [sem_scores[i] for i in no_bm25_idx],
                query_time,
                [all_decay[i] for i in no_bm25_idx],
                [reinforcement_counts[i] for i in no_bm25_idx],
                bm25_scores=None,
            )


def test_aretrieve_scoring_1k(benchmark, units_1k):
    """Production aretrieve pattern at 1k candidates, no bm25."""
    now = datetime.now(timezone.utc)
    sem = [0.7] * len(units_1k)
    bm25 = [None] * len(units_1k)
    rcs = [0] * len(units_1k)
    benchmark(_aretrieve_scoring, units_1k, sem, bm25, rcs, now)


def test_aretrieve_scoring_10k(benchmark, units_10k):
    """Production aretrieve pattern at 10k candidates, no bm25."""
    now = datetime.now(timezone.utc)
    sem = [0.7] * len(units_10k)
    bm25 = [None] * len(units_10k)
    rcs = [0] * len(units_10k)
    benchmark(_aretrieve_scoring, units_10k, sem, bm25, rcs, now)


def test_aretrieve_scoring_hybrid_1k(benchmark, units_1k):
    """Production aretrieve pattern at 1k with mixed bm25 (50% hybrid)."""
    now = datetime.now(timezone.utc)
    n = len(units_1k)
    sem = [random.random() for _ in range(n)]
    bm25 = [random.random() if i % 2 == 0 else None for i in range(n)]
    rcs = [random.randint(0, 3) for _ in range(n)]
    benchmark(_aretrieve_scoring, units_1k, sem, bm25, rcs, now)


def test_aretrieve_scoring_hybrid_10k(benchmark, units_10k):
    """Production aretrieve pattern at 10k with mixed bm25 (50% hybrid)."""
    now = datetime.now(timezone.utc)
    n = len(units_10k)
    sem = [random.random() for _ in range(n)]
    bm25 = [random.random() if i % 2 == 0 else None for i in range(n)]
    rcs = [random.randint(0, 3) for _ in range(n)]
    benchmark(_aretrieve_scoring, units_10k, sem, bm25, rcs, now)
