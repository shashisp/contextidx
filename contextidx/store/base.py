from __future__ import annotations

import re
from abc import ABC, abstractmethod
from datetime import datetime

from contextidx.core.context_unit import ContextUnit

_SAFE_SCOPE_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_scope_keys(scope: dict[str, str]) -> None:
    """Raise ``ValueError`` if any scope key contains unsafe characters.

    Scope keys are interpolated into SQL identifiers (json path expressions),
    so they must be restricted to alphanumeric characters and underscores.
    """
    for key in scope:
        if not _SAFE_SCOPE_KEY.match(key):
            raise ValueError(
                f"Invalid scope key {key!r}: must match [a-zA-Z_][a-zA-Z0-9_]*"
            )


class Store(ABC):
    """Abstract interface for the contextidx internal metadata store."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables / indexes if they don't exist."""

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""

    # ── ContextUnit CRUD ──

    @abstractmethod
    async def create_unit(self, unit: ContextUnit) -> None: ...

    @abstractmethod
    async def get_unit(self, unit_id: str) -> ContextUnit | None: ...

    async def get_units_batch(self, unit_ids: list[str]) -> dict[str, ContextUnit]:
        """Return units for the given IDs as a {id: unit} mapping.

        Subclasses should override with an efficient single-query implementation.
        The default falls back to sequential ``get_unit`` calls.
        """
        result: dict[str, ContextUnit] = {}
        for uid in unit_ids:
            unit = await self.get_unit(uid)
            if unit is not None:
                result[uid] = unit
        return result

    @abstractmethod
    async def update_unit(self, unit_id: str, updates: dict) -> None: ...

    @abstractmethod
    async def delete_unit(self, unit_id: str) -> None: ...

    @abstractmethod
    async def find_units_in_scope(
        self,
        scope: dict[str, str],
        include_superseded: bool = False,
        include_archived: bool = False,
    ) -> list[ContextUnit]: ...

    @abstractmethod
    async def find_active_units(
        self,
        since: datetime | None = None,
    ) -> list[ContextUnit]: ...

    # ── Graph ──

    @abstractmethod
    async def add_graph_edge(
        self, from_id: str, to_id: str, relationship: str, created_at: datetime
    ) -> None: ...

    @abstractmethod
    async def get_graph_edges(
        self, unit_id: str
    ) -> list[tuple[str, str, str, datetime]]: ...

    # ── Decay state ──

    @abstractmethod
    async def upsert_decay_state(
        self,
        unit_id: str,
        current_score: float,
        last_updated: datetime,
        reinforcement_count: int,
    ) -> None: ...

    @abstractmethod
    async def get_decay_state(
        self, unit_id: str
    ) -> tuple[float, datetime, int] | None: ...

    async def get_decay_states_batch(
        self, unit_ids: list[str]
    ) -> dict[str, tuple[float, datetime, int]]:
        """Return decay states for the given IDs as a {id: state} mapping.

        Subclasses should override with an efficient single-query implementation.
        The default falls back to sequential ``get_decay_state`` calls.
        """
        result: dict[str, tuple[float, datetime, int]] = {}
        for uid in unit_ids:
            state = await self.get_decay_state(uid)
            if state is not None:
                result[uid] = state
        return result

    async def upsert_decay_states_batch(
        self,
        states: list[tuple[str, float, datetime, int]],
    ) -> None:
        """Upsert multiple decay states in a single operation.

        ``states`` is a list of ``(unit_id, current_score, last_updated, reinforcement_count)``.
        Subclasses should override with an efficient bulk implementation.
        The default falls back to sequential ``upsert_decay_state`` calls.
        """
        for unit_id, score, last_updated, rc in states:
            await self.upsert_decay_state(unit_id, score, last_updated, rc)

    @abstractmethod
    async def find_expired_units(self, now: datetime) -> list[str]:
        """Return IDs of active units whose ``expires_at`` is <= *now*.

        Implementations must use a SQL index on ``expires_at`` rather than
        loading all units into Python for a full table scan.
        """

    @abstractmethod
    async def get_all_graph_edges(self) -> list[tuple[str, str, str, datetime]]:
        """Return all edges in the graph as ``(from_id, to_id, relationship, created_at)``.

        Used at startup to bulk-load the in-memory TemporalGraph in a single
        query instead of one ``get_graph_edges()`` call per active unit.
        """

    @abstractmethod
    async def increment_reinforcement(self, unit_id: str) -> int: ...

    # ── WAL ──

    @abstractmethod
    async def append_wal(
        self,
        unit_id: str,
        operation: str,
        store_target: str,
        payload: dict,
        written_at: datetime,
    ) -> int: ...

    @abstractmethod
    async def get_pending_wal(self) -> list[dict]: ...

    @abstractmethod
    async def mark_wal_applied(self, seq: int, applied_at: datetime) -> None: ...

    @abstractmethod
    async def mark_wal_failed(self, seq: int) -> None: ...

    @abstractmethod
    async def compact_wal(self, before: datetime) -> int:
        """Delete applied WAL entries older than *before*.

        Returns the number of rows removed.
        """

    # ── Reconciliation ──

    @abstractmethod
    async def find_active_without_vector(self, since: datetime) -> list[str]:
        """Return IDs of active units created since *since*.

        Used by checkpoint reconciliation to find units that may be missing
        from the vector backend.
        """

    # ── Checkpoints ──

    @abstractmethod
    async def update_checkpoint(
        self, store_name: str, last_synced_at: datetime, units_synced: int
    ) -> None: ...

    @abstractmethod
    async def get_checkpoint(
        self, store_name: str
    ) -> tuple[datetime, int] | None: ...
