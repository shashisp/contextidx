from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from contextidx.store.base import Store


@dataclass
class WALEntry:
    seq: int
    unit_id: str
    operation: str
    store_target: str
    payload: dict
    written_at: datetime
    applied_at: datetime | None = None
    status: str = "pending"


class WAL:
    """Write-Ahead Log backed by the internal Store.

    Every write appends to the WAL *before* touching either store.
    On startup, pending entries are replayed to recover from crashes.
    """

    def __init__(self, store: Store):
        self._store = store

    async def append(
        self,
        unit_id: str,
        operation: str,
        store_target: str,
        payload: dict,
    ) -> int:
        """Append an entry and return its sequence number."""
        seq = await self._store.append_wal(
            unit_id=unit_id,
            operation=operation,
            store_target=store_target,
            payload=payload,
            written_at=datetime.now(timezone.utc),
        )
        return seq

    async def replay_pending(self) -> list[WALEntry]:
        """Return all pending entries, oldest first."""
        rows = await self._store.get_pending_wal()
        return [
            WALEntry(
                seq=r["seq"],
                unit_id=r["unit_id"],
                operation=r["operation"],
                store_target=r["store_target"],
                payload=r["payload"],
                written_at=r["written_at"],
                status=r["status"],
            )
            for r in rows
        ]

    async def mark_applied(self, seq: int) -> None:
        await self._store.mark_wal_applied(seq, datetime.now(timezone.utc))

    async def mark_failed(self, seq: int) -> None:
        await self._store.mark_wal_failed(seq)

    async def compact(self, retention_hours: int = 24) -> int:
        """Remove applied WAL entries older than *retention_hours*.

        Returns the number of entries removed.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=retention_hours)
        return await self._store.compact_wal(before=cutoff)
