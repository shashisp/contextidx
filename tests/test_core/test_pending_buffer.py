import asyncio
import time
from datetime import datetime, timezone

import pytest

from contextidx.core.context_unit import ContextUnit
from contextidx.utils.pending_buffer import PendingBuffer


def _unit(uid: str = "ctx_t1", scope: dict | None = None) -> ContextUnit:
    return ContextUnit(
        id=uid,
        content="test",
        scope=scope or {"user_id": "u1"},
    )


class TestPendingBuffer:
    async def test_add_and_get(self):
        buf = PendingBuffer()
        unit = _unit()
        await buf.add(unit)
        results = await buf.get({"user_id": "u1"})
        assert len(results) == 1
        assert results[0].id == "ctx_t1"

    async def test_scope_isolation(self):
        buf = PendingBuffer()
        await buf.add(_unit("a", {"user_id": "u1"}))
        await buf.add(_unit("b", {"user_id": "u2"}))
        assert len(await buf.get({"user_id": "u1"})) == 1
        assert len(await buf.get({"user_id": "u2"})) == 1

    async def test_ttl_expiry(self):
        buf = PendingBuffer(ttl_seconds=0)
        await buf.add(_unit())
        await asyncio.sleep(0.05)
        results = await buf.get({"user_id": "u1"})
        assert len(results) == 0

    async def test_max_units_eviction(self):
        buf = PendingBuffer(max_units_per_scope=3)
        for i in range(5):
            await buf.add(_unit(f"ctx_{i}"))
        results = await buf.get({"user_id": "u1"})
        assert len(results) == 3
        ids = [u.id for u in results]
        assert "ctx_2" in ids
        assert "ctx_3" in ids
        assert "ctx_4" in ids

    async def test_remove(self):
        buf = PendingBuffer()
        await buf.add(_unit("a"))
        await buf.add(_unit("b"))
        buf.remove("a")
        results = await buf.get({"user_id": "u1"})
        assert len(results) == 1
        assert results[0].id == "b"

    async def test_flush_expired(self):
        buf = PendingBuffer(ttl_seconds=0)
        await buf.add(_unit("a"))
        await asyncio.sleep(0.05)
        expired = buf.flush_expired()
        assert len(expired) == 1
        assert expired[0].id == "a"

    async def test_clear(self):
        buf = PendingBuffer()
        await buf.add(_unit())
        buf.clear()
        assert len(await buf.get({"user_id": "u1"})) == 0

    async def test_add_is_awaitable(self):
        """PendingBuffer.add() must be a coroutine (matches RedisPendingBuffer interface)."""
        buf = PendingBuffer()
        coro = buf.add(_unit())
        assert asyncio.iscoroutine(coro)
        await coro

    async def test_get_is_awaitable(self):
        """PendingBuffer.get() must be a coroutine (matches RedisPendingBuffer interface)."""
        buf = PendingBuffer()
        coro = buf.get({"user_id": "u1"})
        assert asyncio.iscoroutine(coro)
        await coro
