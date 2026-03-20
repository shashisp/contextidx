from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from contextidx.core.context_unit import ContextUnit


def _hash_scope(scope: dict[str, str]) -> str:
    canonical = json.dumps(scope, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class PendingBuffer:
    """In-memory scoped buffer for read-after-write consistency.

    Writes land here immediately so that a subsequent retrieve in the same
    session can see them before async conflict resolution + persistence finishes.
    """

    def __init__(self, ttl_seconds: int = 30, max_units_per_scope: int = 50):
        self._ttl = ttl_seconds
        self._max = max_units_per_scope
        self._buffer: dict[str, list[tuple[ContextUnit, datetime]]] = {}

    def add(self, unit: ContextUnit) -> None:
        key = _hash_scope(unit.scope)
        bucket = self._buffer.setdefault(key, [])
        bucket.append((unit, datetime.now(timezone.utc)))
        if len(bucket) > self._max:
            bucket.pop(0)

    def get(self, scope: dict[str, str]) -> list[ContextUnit]:
        """Return non-expired pending units for *scope*."""
        key = _hash_scope(scope)
        bucket = self._buffer.get(key, [])
        now = datetime.now(timezone.utc)
        alive = [
            (unit, ts) for unit, ts in bucket
            if (now - ts).total_seconds() < self._ttl
        ]
        self._buffer[key] = alive
        return [unit for unit, _ in alive]

    def remove(self, unit_id: str) -> None:
        """Remove a specific unit (e.g. after persistence completes)."""
        for key, bucket in self._buffer.items():
            self._buffer[key] = [(u, ts) for u, ts in bucket if u.id != unit_id]

    def flush_expired(self) -> list[ContextUnit]:
        """Remove and return all expired pending units."""
        expired: list[ContextUnit] = []
        now = datetime.now(timezone.utc)
        for key in list(self._buffer):
            bucket = self._buffer[key]
            still_alive = []
            for unit, ts in bucket:
                if (now - ts).total_seconds() >= self._ttl:
                    expired.append(unit)
                else:
                    still_alive.append((unit, ts))
            self._buffer[key] = still_alive
        return expired

    def clear(self) -> None:
        self._buffer.clear()
