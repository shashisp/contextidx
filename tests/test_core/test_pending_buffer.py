import time
from datetime import datetime, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.utils.pending_buffer import PendingBuffer


def _unit(uid: str = "ctx_t1", scope: dict | None = None) -> ContextUnit:
    return ContextUnit(
        id=uid,
        content="test",
        scope=scope or {"user_id": "u1"},
    )


class TestPendingBuffer:
    def test_add_and_get(self):
        buf = PendingBuffer()
        unit = _unit()
        buf.add(unit)
        results = buf.get({"user_id": "u1"})
        assert len(results) == 1
        assert results[0].id == "ctx_t1"

    def test_scope_isolation(self):
        buf = PendingBuffer()
        buf.add(_unit("a", {"user_id": "u1"}))
        buf.add(_unit("b", {"user_id": "u2"}))
        assert len(buf.get({"user_id": "u1"})) == 1
        assert len(buf.get({"user_id": "u2"})) == 1

    def test_ttl_expiry(self):
        buf = PendingBuffer(ttl_seconds=0)
        buf.add(_unit())
        time.sleep(0.05)
        results = buf.get({"user_id": "u1"})
        assert len(results) == 0

    def test_max_units_eviction(self):
        buf = PendingBuffer(max_units_per_scope=3)
        for i in range(5):
            buf.add(_unit(f"ctx_{i}"))
        results = buf.get({"user_id": "u1"})
        assert len(results) == 3
        ids = [u.id for u in results]
        assert "ctx_2" in ids
        assert "ctx_3" in ids
        assert "ctx_4" in ids

    def test_remove(self):
        buf = PendingBuffer()
        buf.add(_unit("a"))
        buf.add(_unit("b"))
        buf.remove("a")
        results = buf.get({"user_id": "u1"})
        assert len(results) == 1
        assert results[0].id == "b"

    def test_flush_expired(self):
        buf = PendingBuffer(ttl_seconds=0)
        buf.add(_unit("a"))
        time.sleep(0.05)
        expired = buf.flush_expired()
        assert len(expired) == 1
        assert expired[0].id == "a"

    def test_clear(self):
        buf = PendingBuffer()
        buf.add(_unit())
        buf.clear()
        assert len(buf.get({"user_id": "u1"})) == 0
