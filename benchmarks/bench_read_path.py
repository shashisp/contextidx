"""Benchmark the read path (aretrieve) at various scales."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine


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
