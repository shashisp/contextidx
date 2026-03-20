"""Tests for ContextIdx constructor validation."""

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.exceptions import ConfigurationError
from tests.conftest import InMemoryVectorBackend


class TestConstructorValidation:
    def test_rejects_zero_half_life(self):
        with pytest.raises(ConfigurationError, match="half_life_days"):
            ContextIdx(backend=InMemoryVectorBackend(), half_life_days=0)

    def test_rejects_negative_half_life(self):
        with pytest.raises(ConfigurationError, match="half_life_days"):
            ContextIdx(backend=InMemoryVectorBackend(), half_life_days=-5)

    def test_rejects_decay_threshold_above_one(self):
        with pytest.raises(ConfigurationError, match="decay_threshold"):
            ContextIdx(backend=InMemoryVectorBackend(), decay_threshold=1.5)

    def test_rejects_negative_decay_threshold(self):
        with pytest.raises(ConfigurationError, match="decay_threshold"):
            ContextIdx(backend=InMemoryVectorBackend(), decay_threshold=-0.1)

    def test_rejects_zero_state_path_interval(self):
        with pytest.raises(ConfigurationError, match="state_path_interval"):
            ContextIdx(backend=InMemoryVectorBackend(), state_path_interval=0)

    def test_rejects_zero_batch_size(self):
        with pytest.raises(ConfigurationError, match="batch_size"):
            ContextIdx(backend=InMemoryVectorBackend(), batch_size=0)

    def test_rejects_negative_batch_flush_interval(self):
        with pytest.raises(ConfigurationError, match="batch_flush_interval"):
            ContextIdx(backend=InMemoryVectorBackend(), batch_flush_interval=-1)

    def test_rejects_zero_reconcile_ticks(self):
        with pytest.raises(ConfigurationError, match="reconcile_every_n_ticks"):
            ContextIdx(backend=InMemoryVectorBackend(), reconcile_every_n_ticks=0)

    def test_rejects_zero_consolidation_ticks(self):
        with pytest.raises(ConfigurationError, match="consolidation_every_n_ticks"):
            ContextIdx(
                backend=InMemoryVectorBackend(), consolidation_every_n_ticks=0
            )

    def test_rejects_zero_wal_compact_ticks(self):
        with pytest.raises(ConfigurationError, match="wal_compact_every_n_ticks"):
            ContextIdx(
                backend=InMemoryVectorBackend(), wal_compact_every_n_ticks=0
            )

    def test_rejects_postgres_without_dsn(self):
        with pytest.raises(ConfigurationError, match="internal_store_dsn"):
            ContextIdx(
                backend=InMemoryVectorBackend(), internal_store_type="postgres"
            )

    def test_rejects_redis_without_url(self):
        with pytest.raises(ConfigurationError, match="redis_url"):
            ContextIdx(
                backend=InMemoryVectorBackend(), pending_buffer_type="redis"
            )

    def test_accepts_valid_defaults(self):
        idx = ContextIdx(backend=InMemoryVectorBackend(), openai_api_key="test")
        assert idx._half_life_days == 30.0
        assert idx._decay_threshold == 0.01

    def test_accepts_valid_custom_values(self):
        idx = ContextIdx(
            backend=InMemoryVectorBackend(),
            openai_api_key="test",
            half_life_days=7.0,
            decay_threshold=0.05,
            batch_size=50,
            state_path_interval=120.0,
        )
        assert idx._half_life_days == 7.0
        assert idx._decay_threshold == 0.05

    def test_accepts_boundary_decay_threshold(self):
        idx = ContextIdx(
            backend=InMemoryVectorBackend(),
            openai_api_key="test",
            decay_threshold=0.0,
        )
        assert idx._decay_threshold == 0.0

        idx2 = ContextIdx(
            backend=InMemoryVectorBackend(),
            openai_api_key="test",
            decay_threshold=1.0,
        )
        assert idx2._decay_threshold == 1.0
