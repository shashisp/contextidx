"""Tests for the _core module: import routing, RUST_AVAILABLE flag, fallback correctness."""

from __future__ import annotations

import math

import pytest


class TestCoreImport:
    def test_module_imports_successfully(self):
        from contextidx._core import RUST_AVAILABLE, batch_decay, batch_score, detect_contradictions

        assert isinstance(RUST_AVAILABLE, bool)
        assert callable(batch_decay)
        assert callable(batch_score)
        assert callable(detect_contradictions)

    def test_rust_available_is_bool(self):
        from contextidx._core import RUST_AVAILABLE
        assert RUST_AVAILABLE is True or RUST_AVAILABLE is False


class TestBatchDecayFallback:
    def test_exponential_single(self):
        from contextidx._core import batch_decay

        result = batch_decay(
            confidences=[0.8],
            decay_rates=[0.02],
            age_days=[10.0],
            reinforcement_counts=[0],
            model="exponential",
            reinforcement_factor=0.5,
        )
        expected = 0.8 * math.exp(-0.02 * 10.0)
        assert len(result) == 1
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_linear_single(self):
        from contextidx._core import batch_decay

        result = batch_decay(
            confidences=[1.0],
            decay_rates=[0.02],
            age_days=[5.0],
            reinforcement_counts=[0],
            model="linear",
            reinforcement_factor=0.5,
        )
        half_life = math.log(2) / 0.02
        expected = max(0.0, 1.0 - 5.0 / half_life)
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_step_within_halflife(self):
        from contextidx._core import batch_decay

        result = batch_decay(
            confidences=[0.9],
            decay_rates=[0.02],
            age_days=[1.0],
            reinforcement_counts=[0],
            model="step",
            reinforcement_factor=0.5,
        )
        assert result[0] == pytest.approx(0.9)

    def test_step_beyond_halflife(self):
        from contextidx._core import batch_decay

        half_life = math.log(2) / 0.02
        result = batch_decay(
            confidences=[0.9],
            decay_rates=[0.02],
            age_days=[half_life + 1.0],
            reinforcement_counts=[0],
            model="step",
            reinforcement_factor=0.5,
        )
        assert result[0] == 0.0

    def test_reinforcement_reduces_effective_age(self):
        from contextidx._core import batch_decay

        no_reinforce = batch_decay([0.8], [0.02], [20.0], [0], "exponential", 0.5)
        with_reinforce = batch_decay([0.8], [0.02], [20.0], [2], "exponential", 0.5)
        assert with_reinforce[0] > no_reinforce[0]

    def test_batch_multiple_items(self):
        from contextidx._core import batch_decay

        result = batch_decay(
            confidences=[0.8, 0.6, 1.0],
            decay_rates=[0.02, 0.05, 0.01],
            age_days=[10.0, 20.0, 5.0],
            reinforcement_counts=[0, 1, 0],
            model="exponential",
            reinforcement_factor=0.5,
        )
        assert len(result) == 3
        assert all(0.0 <= v <= 1.0 for v in result)


class TestBatchScoreFallback:
    def test_basic_scoring(self):
        from contextidx._core import batch_score

        result = batch_score(
            semantic_scores=[0.8],
            recency_scores=[0.5],
            confidences=[0.9],
            decay_scores=[0.7],
            reinforcement_scores=[0.3],
            bm25_scores=[0.0],
            weights=[0.3, 0.1, 0.25, 0.2, 0.1, 0.05],
        )
        assert len(result) == 1
        expected = 0.3 * 0.8 + 0.1 * 0.0 + 0.25 * 0.5 + 0.2 * 0.9 + 0.1 * 0.7 + 0.05 * 0.3
        assert result[0] == pytest.approx(expected, rel=1e-6)

    def test_score_clamped_to_0_1(self):
        from contextidx._core import batch_score

        result = batch_score(
            semantic_scores=[1.0],
            recency_scores=[1.0],
            confidences=[1.0],
            decay_scores=[1.0],
            reinforcement_scores=[1.0],
            bm25_scores=[1.0],
            weights=[0.3, 0.1, 0.25, 0.2, 0.1, 0.05],
        )
        assert result[0] <= 1.0

    def test_batch_multiple(self):
        from contextidx._core import batch_score

        result = batch_score(
            semantic_scores=[0.8, 0.5],
            recency_scores=[0.5, 0.3],
            confidences=[0.9, 0.7],
            decay_scores=[0.7, 0.4],
            reinforcement_scores=[0.3, 0.1],
            bm25_scores=[0.0, 0.6],
            weights=[0.3, 0.1, 0.25, 0.2, 0.1, 0.05],
        )
        assert len(result) == 2
        assert all(0.0 <= v <= 1.0 for v in result)


class TestDetectContradictionsFallback:
    def test_verb_pair_contradiction(self):
        from contextidx._core import detect_contradictions

        result = detect_contradictions(
            "User prefers dark mode",
            ["User does not prefer dark mode"],
        )
        assert result == [True]

    def test_negation_overlap_contradiction(self):
        from contextidx._core import detect_contradictions

        result = detect_contradictions(
            "The system is fast and reliable",
            ["The system is not fast and reliable"],
        )
        assert result == [True]

    def test_no_contradiction(self):
        from contextidx._core import detect_contradictions

        result = detect_contradictions(
            "I like cats",
            ["The weather is nice today"],
        )
        assert result == [False]

    def test_multiple_existing(self):
        from contextidx._core import detect_contradictions

        result = detect_contradictions(
            "User likes pizza",
            [
                "User does not like pizza",
                "The server runs on port 8080",
                "User likes pasta",
            ],
        )
        assert result[0] is True
        assert result[1] is False
        assert result[2] is False
