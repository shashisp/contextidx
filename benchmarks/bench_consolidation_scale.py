"""Consolidation benchmarks at 50K and 100K scale.

``find_redundant_pairs`` is O(N²) — these benchmarks make that visible so any
refactor targeting the bottleneck can measure its impact.

Mark these as ``slow`` so they run in nightly CI but not on every push::

    pytest benchmarks/bench_consolidation_scale.py --benchmark-only -v -m slow

Or run them explicitly::

    pytest benchmarks/bench_consolidation_scale.py --benchmark-only -v
"""

from __future__ import annotations

import pytest

from benchmarks.conftest import _make_units
from contextidx.core.consolidation import find_redundant_pairs

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def units_50k():
    return _make_units(50_000)


@pytest.fixture(scope="module")
def units_100k():
    return _make_units(100_000)


class TestConsolidationAtScale:
    """O(N²) bottleneck: these benchmarks expose absolute latency at scale.

    A healthy system should complete 50K in < 60 s and 100K in < 300 s.
    After ANN-based pre-filtering (improvement #8), both should drop to < 2 s.
    """

    def test_consolidation_50k(self, benchmark, units_50k):
        """find_redundant_pairs at 50K: expect to be slow before ANN optimization."""
        result = benchmark.pedantic(
            find_redundant_pairs,
            args=(units_50k, 0.92),
            rounds=1,
            iterations=1,
        )
        assert isinstance(result, list)

    def test_consolidation_100k(self, benchmark, units_100k):
        """find_redundant_pairs at 100K: quadratic behavior becomes obvious."""
        result = benchmark.pedantic(
            find_redundant_pairs,
            args=(units_100k, 0.92),
            rounds=1,
            iterations=1,
        )
        assert isinstance(result, list)

    def test_consolidation_threshold_sensitivity_50k(self, benchmark, units_50k):
        """Lower threshold → more pairs found, different performance profile."""
        result = benchmark.pedantic(
            find_redundant_pairs,
            args=(units_50k, 0.80),
            rounds=1,
            iterations=1,
        )
        assert isinstance(result, list)
