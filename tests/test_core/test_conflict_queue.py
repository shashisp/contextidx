"""Tests for ConflictQueue: enqueue, drain, status transitions."""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from contextidx.core.conflict_resolver import ConflictResolver
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.utils.conflict_queue import ConflictQueue, EntryStatus


def _unit(content: str, embedding: list[float] | None = None, **kw) -> ContextUnit:
    defaults = {
        "id": generate_unit_id(),
        "content": content,
        "scope": {"user_id": "u1"},
        "confidence": 0.8,
        "source": "test",
        "decay_model": "exponential",
        "decay_rate": 0.02,
    }
    defaults.update(kw)
    u = ContextUnit(**defaults)
    if embedding is not None:
        u.embedding = embedding
    return u


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestConflictQueue:
    @pytest.fixture
    def resolver(self):
        return ConflictResolver(strategy="LAST_WRITE_WINS", semantic_threshold=0.85)

    @pytest.fixture
    def queue(self, resolver):
        return ConflictQueue(resolver)

    async def test_enqueue_increases_pending_count(self, queue):
        u1 = _unit("unit a")
        u2 = _unit("unit b")
        await queue.enqueue(u1, [u2])
        assert queue.pending_count == 1
        assert queue.size == 1

    async def test_drain_with_no_conflicts(self, queue):
        """Orthogonal embeddings + no negation => no conflict found."""
        u1 = _unit("I like cats", embedding=_normalized([1.0, 0.0, 0.0]))
        u2 = _unit("The weather is nice", embedding=_normalized([0.0, 1.0, 0.0]))

        await queue.enqueue(u1, [u2])

        callback_calls = []

        async def callback(new_unit, conflicts):
            callback_calls.append((new_unit, conflicts))

        applied = await queue.drain(callback)
        assert applied == 0
        assert queue.pending_count == 0
        assert len(callback_calls) == 0

    async def test_drain_with_semantic_conflict(self, queue):
        """Same embedding + contradictory negation => conflict detected."""
        emb = _normalized([1.0, 0.0, 0.0])
        u_new = _unit("I like dogs", embedding=emb)
        u_existing = _unit("I do not like dogs", embedding=emb)

        await queue.enqueue(u_new, [u_existing])

        resolved = []

        async def callback(new_unit, conflicts):
            resolved.append((new_unit.id, [c.id for c in conflicts]))

        applied = await queue.drain(callback)
        assert applied == 1
        assert len(resolved) == 1
        assert resolved[0][1] == [u_existing.id]
        assert queue.pending_count == 0

    async def test_drain_removes_applied_entries(self, queue):
        u1 = _unit("hello", embedding=_normalized([1.0, 0.0, 0.0]))
        u2 = _unit("world", embedding=_normalized([0.0, 1.0, 0.0]))
        await queue.enqueue(u1, [u2])

        async def noop(new_unit, conflicts):
            pass

        await queue.drain(noop)
        assert queue.size == 0

    async def test_multiple_enqueues(self, queue):
        for i in range(5):
            await queue.enqueue(
                _unit(f"item {i}", embedding=_normalized([1.0, 0.0])),
                [_unit(f"other {i}", embedding=_normalized([0.0, 1.0]))],
            )
        assert queue.pending_count == 5

    async def test_callback_exception_does_not_crash_drain(self, queue):
        emb = _normalized([1.0, 0.0, 0.0])
        u_new = _unit("I like cats", embedding=emb)
        u_old = _unit("I do not like cats", embedding=emb)
        await queue.enqueue(u_new, [u_old])

        async def bad_callback(new_unit, conflicts):
            raise RuntimeError("kaboom")

        applied = await queue.drain(bad_callback)
        assert applied == 0
