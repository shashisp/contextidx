"""Benchmark decay engine: single vs batch computation."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from contextidx.core.decay_engine import DecayEngine


def test_batch_decay_1k(benchmark, units_1k):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)
    rcs = [0] * len(units_1k)

    benchmark(engine.batch_compute_decay, units_1k, now, rcs)


def test_batch_decay_10k(benchmark, units_10k):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)
    rcs = [0] * len(units_10k)

    benchmark(engine.batch_compute_decay, units_10k, now, rcs)


def test_batch_decay_100k(benchmark, units_100k):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)
    rcs = [0] * len(units_100k)

    benchmark(engine.batch_compute_decay, units_100k, now, rcs)


def test_single_decay_1k(benchmark, units_1k):
    engine = DecayEngine()
    now = datetime.now(timezone.utc)

    def run():
        for u in units_1k:
            engine.compute_decay(u, now)

    benchmark(run)
