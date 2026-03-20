"""Async queue for deferred semantic conflict detection.

In ``tiered`` mode, rule-based detection runs inline while semantic
detection is queued here and drained by the state-path background loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from contextidx.core.conflict_resolver import ConflictResolver
from contextidx.core.context_unit import ContextUnit

logger = logging.getLogger("contextidx.conflict_queue")


class EntryStatus(str, Enum):
    PENDING = "pending"
    RESOLVED = "resolved"
    APPLIED = "applied"


@dataclass
class QueueEntry:
    new_unit: ContextUnit
    candidates: list[ContextUnit]
    status: EntryStatus = EntryStatus.PENDING
    conflicts: list[ContextUnit] = field(default_factory=list)
    queued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConflictQueue:
    """In-memory async queue of ``(new_unit, candidates)`` pending semantic resolution."""

    def __init__(self, resolver: ConflictResolver):
        self._resolver = resolver
        self._queue: list[QueueEntry] = []
        self._lock = asyncio.Lock()

    async def enqueue(self, new_unit: ContextUnit, candidates: list[ContextUnit]) -> None:
        """Add a pair for deferred semantic conflict resolution."""
        async with self._lock:
            self._queue.append(QueueEntry(new_unit=new_unit, candidates=candidates))

    async def drain(
        self,
        resolve_callback,
    ) -> int:
        """Process all pending entries.

        For each entry, runs semantic conflict detection.  If conflicts are
        found, calls ``resolve_callback(new_unit, conflicts)`` which should
        handle superseding in the store / graph.

        Returns the number of entries that found conflicts.
        """
        async with self._lock:
            pending = [e for e in self._queue if e.status == EntryStatus.PENDING]

        applied = 0
        for entry in pending:
            conflicts = self._resolver.detect_semantic_conflicts(
                entry.new_unit, entry.candidates,
            )
            entry.conflicts = conflicts
            entry.status = EntryStatus.RESOLVED

            if conflicts:
                try:
                    await resolve_callback(entry.new_unit, conflicts)
                    entry.status = EntryStatus.APPLIED
                    applied += 1
                except Exception:
                    logger.exception(
                        "Failed to apply semantic conflict resolution for unit %s",
                        entry.new_unit.id,
                    )
            else:
                entry.status = EntryStatus.APPLIED

        async with self._lock:
            self._queue = [e for e in self._queue if e.status != EntryStatus.APPLIED]

        return applied

    @property
    def pending_count(self) -> int:
        return sum(1 for e in self._queue if e.status == EntryStatus.PENDING)

    @property
    def size(self) -> int:
        return len(self._queue)
