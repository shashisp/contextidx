"""Unit tests for RedisPendingBuffer — uses mocks to avoid requiring a real Redis instance."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextidx.core.context_unit import ContextUnit, generate_unit_id


def _redis_available() -> bool:
    try:
        import redis.asyncio
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _redis_available(), reason="redis not installed"
)


def _make_unit(**overrides) -> ContextUnit:
    defaults = {
        "id": generate_unit_id(),
        "content": "test context",
        "scope": {"user_id": "u1"},
        "confidence": 0.9,
        "source": "test",
        "decay_model": "exponential",
        "decay_rate": 0.02,
    }
    defaults.update(overrides)
    return ContextUnit(**defaults)


class TestRedisPendingBufferImport:
    def test_can_import(self):
        from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
        assert RedisPendingBuffer is not None

    def test_serialization_roundtrip(self):
        from contextidx.utils.redis_pending_buffer import _deserialize_unit, _serialize_unit
        unit = _make_unit()
        unit.embedding = [0.1, 0.2, 0.3]
        raw = _serialize_unit(unit)
        restored = _deserialize_unit(raw)
        assert restored.id == unit.id
        assert restored.content == unit.content
        assert restored.embedding == [0.1, 0.2, 0.3]
        assert restored.scope == unit.scope

    def test_serialization_no_embedding(self):
        from contextidx.utils.redis_pending_buffer import _deserialize_unit, _serialize_unit
        unit = _make_unit()
        raw = _serialize_unit(unit)
        restored = _deserialize_unit(raw)
        assert restored.embedding is None

    def test_serialization_bytes_input(self):
        from contextidx.utils.redis_pending_buffer import _deserialize_unit, _serialize_unit
        unit = _make_unit()
        raw = _serialize_unit(unit).encode("utf-8")
        restored = _deserialize_unit(raw)
        assert restored.id == unit.id

    def test_hash_scope_deterministic(self):
        from contextidx.utils.redis_pending_buffer import _hash_scope
        scope = {"user_id": "u1", "session_id": "s1"}
        assert _hash_scope(scope) == _hash_scope(scope)

    def test_hash_scope_different_scopes(self):
        from contextidx.utils.redis_pending_buffer import _hash_scope
        h1 = _hash_scope({"user_id": "u1"})
        h2 = _hash_scope({"user_id": "u2"})
        assert h1 != h2


class TestRedisPendingBufferMocked:
    @pytest.fixture
    def mock_buffer(self):
        from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
        buf = RedisPendingBuffer.__new__(RedisPendingBuffer)
        buf._ttl = 30
        buf._max = 50
        buf._redis = AsyncMock()
        pipe = AsyncMock()
        pipe.execute = AsyncMock(return_value=[True, True])
        buf._redis.pipeline.return_value = pipe
        buf._redis.zcard = AsyncMock(return_value=1)
        return buf

    async def test_add(self, mock_buffer):
        unit = _make_unit()
        await mock_buffer.add(unit)
        mock_buffer._redis.pipeline.assert_called_once()

    async def test_get_returns_empty(self, mock_buffer):
        mock_buffer._redis.zremrangebyscore = AsyncMock()
        mock_buffer._redis.zrangebyscore = AsyncMock(return_value=[])
        result = await mock_buffer.get({"user_id": "u1"})
        assert result == []

    async def test_get_returns_units(self, mock_buffer):
        from contextidx.utils.redis_pending_buffer import _serialize_unit
        unit = _make_unit()
        raw = _serialize_unit(unit).encode("utf-8")
        mock_buffer._redis.zremrangebyscore = AsyncMock()
        mock_buffer._redis.zrangebyscore = AsyncMock(return_value=[raw])
        result = await mock_buffer.get({"user_id": "u1"})
        assert len(result) == 1
        assert result[0].id == unit.id

    async def test_clear(self, mock_buffer):
        async def _scan_iter_empty(**kwargs):
            return
            yield  # noqa: unreachable — makes this an async generator

        mock_buffer._redis.scan_iter = _scan_iter_empty
        await mock_buffer.clear()

    async def test_close(self, mock_buffer):
        mock_buffer._redis.aclose = AsyncMock()
        await mock_buffer.close()
        mock_buffer._redis.aclose.assert_called_once()
