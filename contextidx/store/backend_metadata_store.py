"""Store implementation that delegates ContextUnit metadata to a VectorBackend.

Used when the vector backend has ``supports_metadata_store == True`` (e.g.
Weaviate, which stores metadata as native object properties).  Graph edges,
WAL, and checkpoint tables remain in a lightweight SQLite database for
reliability.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from contextidx.backends.base import VectorBackend
from contextidx.core.context_unit import ContextUnit
from contextidx.store.base import Store
from contextidx.store.sqlite_store import SQLiteStore

logger = logging.getLogger("contextidx.store.backend_metadata")


class BackendMetadataStore(Store):
    """Delegates ContextUnit CRUD to the vector backend.

    Graph, WAL, decay-state and checkpoint operations are handled by an
    internal ``SQLiteStore`` that is auto-provisioned.
    """

    def __init__(
        self,
        backend: VectorBackend,
        *,
        graph_store_path: str | Path = ".contextidx/graph.db",
    ):
        self._backend = backend
        self._sqlite = SQLiteStore(path=graph_store_path)
        self._units_cache: dict[str, ContextUnit] = {}

    async def initialize(self) -> None:
        await self._sqlite.initialize()

    async def close(self) -> None:
        await self._sqlite.close()

    # ── ContextUnit CRUD ──
    # Store units in the backend AND keep a local SQLite row for graph/decay
    # queries that need to iterate over units.

    async def create_unit(self, unit: ContextUnit) -> None:
        self._units_cache[unit.id] = unit
        await self._sqlite.create_unit(unit)

    async def get_unit(self, unit_id: str) -> ContextUnit | None:
        if unit_id in self._units_cache:
            return self._units_cache[unit_id]
        return await self._sqlite.get_unit(unit_id)

    async def get_units_batch(self, unit_ids: list[str]) -> dict[str, ContextUnit]:
        result: dict[str, ContextUnit] = {}
        missing: list[str] = []
        for uid in unit_ids:
            if uid in self._units_cache:
                result[uid] = self._units_cache[uid]
            else:
                missing.append(uid)
        if missing:
            from_db = await self._sqlite.get_units_batch(missing)
            result.update(from_db)
        return result

    async def update_unit(self, unit_id: str, updates: dict) -> None:
        await self._sqlite.update_unit(unit_id, updates)
        await self._backend.update_metadata(unit_id, updates)
        if unit_id in self._units_cache:
            unit = self._units_cache[unit_id]
            for k, v in updates.items():
                if hasattr(unit, k):
                    object.__setattr__(unit, k, v)

    async def delete_unit(self, unit_id: str) -> None:
        await self._sqlite.delete_unit(unit_id)
        self._units_cache.pop(unit_id, None)
        try:
            await self._backend.delete(unit_id)
        except Exception:
            logger.warning("Failed to delete unit %s from backend", unit_id)

    async def find_units_in_scope(
        self,
        scope: dict[str, str],
        include_superseded: bool = False,
        include_archived: bool = False,
    ) -> list[ContextUnit]:
        return await self._sqlite.find_units_in_scope(
            scope, include_superseded=include_superseded, include_archived=include_archived
        )

    async def find_active_units(
        self,
        since: datetime | None = None,
    ) -> list[ContextUnit]:
        return await self._sqlite.find_active_units(since=since)

    # ── Graph — delegated to SQLite ──

    async def add_graph_edge(
        self, from_id: str, to_id: str, relationship: str, created_at: datetime
    ) -> None:
        await self._sqlite.add_graph_edge(from_id, to_id, relationship, created_at)

    async def get_graph_edges(
        self, unit_id: str
    ) -> list[tuple[str, str, str, datetime]]:
        return await self._sqlite.get_graph_edges(unit_id)

    async def get_all_graph_edges(self) -> list[tuple[str, str, str, datetime]]:
        return await self._sqlite.get_all_graph_edges()

    # ── Decay state — delegated to SQLite ──

    async def upsert_decay_state(
        self,
        unit_id: str,
        current_score: float,
        last_updated: datetime,
        reinforcement_count: int,
    ) -> None:
        await self._sqlite.upsert_decay_state(
            unit_id, current_score, last_updated, reinforcement_count
        )

    async def get_decay_state(
        self, unit_id: str
    ) -> tuple[float, datetime, int] | None:
        return await self._sqlite.get_decay_state(unit_id)

    async def get_decay_states_batch(
        self, unit_ids: list[str]
    ) -> dict[str, tuple[float, datetime, int]]:
        return await self._sqlite.get_decay_states_batch(unit_ids)

    async def upsert_decay_states_batch(
        self,
        states: list[tuple[str, float, datetime, int]],
    ) -> None:
        await self._sqlite.upsert_decay_states_batch(states)

    async def find_expired_units(self, now: datetime) -> list[str]:
        return await self._sqlite.find_expired_units(now)

    async def increment_reinforcement(self, unit_id: str) -> int:
        return await self._sqlite.increment_reinforcement(unit_id)

    # ── WAL — delegated to SQLite ──

    async def append_wal(
        self,
        unit_id: str,
        operation: str,
        store_target: str,
        payload: dict,
        written_at: datetime,
    ) -> int:
        return await self._sqlite.append_wal(
            unit_id, operation, store_target, payload, written_at
        )

    async def get_pending_wal(self) -> list[dict]:
        return await self._sqlite.get_pending_wal()

    async def mark_wal_applied(self, seq: int, applied_at: datetime) -> None:
        await self._sqlite.mark_wal_applied(seq, applied_at)

    async def mark_wal_failed(self, seq: int) -> None:
        await self._sqlite.mark_wal_failed(seq)

    async def compact_wal(self, before: datetime) -> int:
        return await self._sqlite.compact_wal(before)

    # ── Checkpoints — delegated to SQLite ──

    async def update_checkpoint(
        self, store_name: str, last_synced_at: datetime, units_synced: int
    ) -> None:
        await self._sqlite.update_checkpoint(store_name, last_synced_at, units_synced)

    async def get_checkpoint(
        self, store_name: str
    ) -> tuple[datetime, int] | None:
        return await self._sqlite.get_checkpoint(store_name)

    # ── Reconciliation helper ──

    async def find_active_without_vector(self, since: datetime) -> list[str]:
        """Return IDs of units present in SQLite but not yet verified in the backend."""
        return await self._sqlite.find_active_without_vector(since)
