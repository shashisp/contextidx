from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from contextidx.core.context_unit import ContextUnit


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
