"""Benchmark scoring engine: single vs batch computation."""

from __future__ import annotations

import random
from datetime import datetime, timezone

import pytest

from contextidx.core.scoring_engine import ScoringEngine


def test_batch_score_1k(benchmark, units_1k):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)
    n = len(units_1k)
    sem = [random.random() for _ in range(n)]
    decay = [random.random() for _ in range(n)]
    rcs = [random.randint(0, 5) for _ in range(n)]

    benchmark(engine.batch_compute_score, units_1k, sem, now, decay, rcs)


def test_batch_score_10k(benchmark, units_10k):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)
    n = len(units_10k)
    sem = [random.random() for _ in range(n)]
    decay = [random.random() for _ in range(n)]
    rcs = [random.randint(0, 5) for _ in range(n)]

    benchmark(engine.batch_compute_score, units_10k, sem, now, decay, rcs)


def test_batch_score_100k(benchmark, units_100k):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)
    n = len(units_100k)
    sem = [random.random() for _ in range(n)]
    decay = [random.random() for _ in range(n)]
    rcs = [random.randint(0, 5) for _ in range(n)]

    benchmark(engine.batch_compute_score, units_100k, sem, now, decay, rcs)


def test_single_score_1k(benchmark, units_1k):
    engine = ScoringEngine()
    now = datetime.now(timezone.utc)

    def run():
        for u in units_1k:
            engine.compute_score(
                unit=u,
                semantic_score=random.random(),
                query_time=now,
                decay_score=random.random(),
                reinforcement_count=0,
            )

    benchmark(run)
