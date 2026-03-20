"""Async PostgreSQL-backed metadata store for contextidx.

Requires optional dependency: pip install contextidx[postgres]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from contextidx.core.context_unit import ContextUnit
from contextidx.store.base import Store, validate_scope_keys
from contextidx.store.postgres_schema import INDEXES_SQL, SCHEMA_SQL

try:
    import asyncpg
except ImportError as exc:
    raise ImportError(
        "PostgresStore requires asyncpg. Install with: pip install contextidx[postgres]"
    ) from exc


def _row_to_unit(row: asyncpg.Record) -> ContextUnit:
    emb_raw = row["embedding"]
    embedding = json.loads(emb_raw) if emb_raw else None
    scope = row["scope"] if isinstance(row["scope"], dict) else json.loads(row["scope"])
    return ContextUnit(
        id=row["id"],
        content=row["content"],
        embedding=embedding,
        scope=scope,
        confidence=float(row["confidence"]),
        decay_rate=float(row["decay_rate"]),
        decay_model=row["decay_model"],
        version=row["version"],
        source=row["source"],
        superseded_by=row["superseded_by"],
        timestamp=row["created_at"],
        expires_at=row["expires_at"],
    )


class PostgresStore(Store):
    """Async PostgreSQL-backed metadata store for contextidx.

    Uses asyncpg with connection pooling for production-scale workloads.
    Schema is equivalent to SQLiteStore -- migration is a connection string swap.
    """

    def __init__(self, dsn: str, *, min_pool_size: int = 2, max_pool_size: int = 10):
        self._dsn = dsn
        self._min_pool = min_pool_size
        self._max_pool = max_pool_size
        self._pool: asyncpg.Pool | None = None

    async def initialize(self) -> None:
        self._pool = await asyncpg.create_pool(
            self._dsn, min_size=self._min_pool, max_size=self._max_pool,
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
            for stmt in INDEXES_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    await conn.execute(stmt)

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    @property
    def _conn_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._pool

    # ── ContextUnit CRUD ──

    async def create_unit(self, unit: ContextUnit) -> None:
        await self._conn_pool.execute(
            """INSERT INTO context_units
               (id, content, embedding, scope, confidence, decay_rate, decay_model,
                version, source, superseded_by, created_at, expires_at, archived_at)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)""",
            unit.id,
            unit.content,
            json.dumps(unit.embedding) if unit.embedding else None,
            json.dumps(unit.scope),
            unit.confidence,
            unit.decay_rate,
            unit.decay_model,
            unit.version,
            unit.source,
            unit.superseded_by,
            unit.timestamp,
            unit.expires_at,
            None,
        )

    async def get_unit(self, unit_id: str) -> ContextUnit | None:
        row = await self._conn_pool.fetchrow(
            "SELECT * FROM context_units WHERE id = $1", unit_id,
        )
        if row is None:
            return None
        return _row_to_unit(row)

    async def get_units_batch(self, unit_ids: list[str]) -> dict[str, ContextUnit]:
        if not unit_ids:
            return {}
        rows = await self._conn_pool.fetch(
            "SELECT * FROM context_units WHERE id = ANY($1::text[])", unit_ids,
        )
        return {row["id"]: _row_to_unit(row) for row in rows}

    async def update_unit(self, unit_id: str, updates: dict) -> None:
        if not updates:
            return
        set_parts: list[str] = []
        values: list = []
        idx = 1
        for key, val in updates.items():
            col = "created_at" if key == "timestamp" else key
            if key == "scope":
                val = json.dumps(val)
            set_parts.append(f"{col} = ${idx}")
            values.append(val)
            idx += 1
        values.append(unit_id)
        sql = f"UPDATE context_units SET {', '.join(set_parts)} WHERE id = ${idx}"
        await self._conn_pool.execute(sql, *values)

    async def delete_unit(self, unit_id: str) -> None:
        async with self._conn_pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("DELETE FROM decay_state WHERE unit_id = $1", unit_id)
                await conn.execute(
                    "DELETE FROM context_graph WHERE from_id = $1 OR to_id = $1", unit_id,
                )
                await conn.execute("DELETE FROM context_units WHERE id = $1", unit_id)

    async def find_units_in_scope(
        self,
        scope: dict[str, str],
        include_superseded: bool = False,
        include_archived: bool = False,
    ) -> list[ContextUnit]:
        validate_scope_keys(scope)
        conditions = ["1=1"]
        params: list = []
        idx = 1
        for key, val in scope.items():
            conditions.append(f"scope->>'{key}' = ${idx}")
            params.append(val)
            idx += 1
        if not include_superseded:
            conditions.append("superseded_by IS NULL")
        if not include_archived:
            conditions.append("archived_at IS NULL")
        query = f"SELECT * FROM context_units WHERE {' AND '.join(conditions)}"
        rows = await self._conn_pool.fetch(query, *params)
        return [_row_to_unit(r) for r in rows]

    async def find_active_units(
        self,
        since: datetime | None = None,
    ) -> list[ContextUnit]:
        if since:
            rows = await self._conn_pool.fetch(
                "SELECT * FROM context_units "
                "WHERE superseded_by IS NULL AND archived_at IS NULL AND created_at >= $1",
                since,
            )
        else:
            rows = await self._conn_pool.fetch(
                "SELECT * FROM context_units "
                "WHERE superseded_by IS NULL AND archived_at IS NULL",
            )
        return [_row_to_unit(r) for r in rows]

    # ── Graph ──

    async def add_graph_edge(
        self, from_id: str, to_id: str, relationship: str, created_at: datetime
    ) -> None:
        await self._conn_pool.execute(
            """INSERT INTO context_graph (from_id, to_id, relationship, created_at)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT DO NOTHING""",
            from_id, to_id, relationship, created_at,
        )

    async def get_graph_edges(
        self, unit_id: str
    ) -> list[tuple[str, str, str, datetime]]:
        rows = await self._conn_pool.fetch(
            "SELECT from_id, to_id, relationship, created_at "
            "FROM context_graph WHERE from_id = $1 OR to_id = $1",
            unit_id,
        )
        return [(r["from_id"], r["to_id"], r["relationship"], r["created_at"]) for r in rows]

    # ── Decay state ──

    async def upsert_decay_state(
        self,
        unit_id: str,
        current_score: float,
        last_updated: datetime,
        reinforcement_count: int,
    ) -> None:
        await self._conn_pool.execute(
            """INSERT INTO decay_state (unit_id, current_score, last_updated, reinforcement_count)
               VALUES ($1, $2, $3, $4)
               ON CONFLICT(unit_id) DO UPDATE SET
                   current_score = EXCLUDED.current_score,
                   last_updated = EXCLUDED.last_updated,
                   reinforcement_count = EXCLUDED.reinforcement_count""",
            unit_id, current_score, last_updated, reinforcement_count,
        )

    async def get_decay_state(
        self, unit_id: str
    ) -> tuple[float, datetime, int] | None:
        row = await self._conn_pool.fetchrow(
            "SELECT current_score, last_updated, reinforcement_count "
            "FROM decay_state WHERE unit_id = $1",
            unit_id,
        )
        if row is None:
            return None
        return (float(row["current_score"]), row["last_updated"], row["reinforcement_count"])

    async def get_decay_states_batch(
        self, unit_ids: list[str]
    ) -> dict[str, tuple[float, datetime, int]]:
        if not unit_ids:
            return {}
        rows = await self._conn_pool.fetch(
            "SELECT unit_id, current_score, last_updated, reinforcement_count "
            "FROM decay_state WHERE unit_id = ANY($1::text[])",
            unit_ids,
        )
        return {
            r["unit_id"]: (float(r["current_score"]), r["last_updated"], r["reinforcement_count"])
            for r in rows
        }

    async def increment_reinforcement(self, unit_id: str) -> int:
        await self._conn_pool.execute(
            """UPDATE decay_state
               SET reinforcement_count = reinforcement_count + 1,
                   last_updated = $1
               WHERE unit_id = $2""",
            datetime.now(timezone.utc), unit_id,
        )
        state = await self.get_decay_state(unit_id)
        return state[2] if state else 0

    # ── Reconciliation ──

    async def find_active_without_vector(self, since: datetime) -> list[str]:
        rows = await self._conn_pool.fetch(
            "SELECT id FROM context_units "
            "WHERE superseded_by IS NULL AND archived_at IS NULL AND created_at >= $1",
            since,
        )
        return [r["id"] for r in rows]

    # ── WAL ──

    async def append_wal(
        self,
        unit_id: str,
        operation: str,
        store_target: str,
        payload: dict,
        written_at: datetime,
    ) -> int:
        row = await self._conn_pool.fetchrow(
            """INSERT INTO wal (unit_id, operation, store_target, payload, written_at, status)
               VALUES ($1, $2, $3, $4, $5, 'pending')
               RETURNING seq""",
            unit_id, operation, store_target, json.dumps(payload), written_at,
        )
        return row["seq"]  # type: ignore[index]

    async def get_pending_wal(self) -> list[dict]:
        rows = await self._conn_pool.fetch(
            "SELECT * FROM wal WHERE status = 'pending' ORDER BY seq"
        )
        return [
            {
                "seq": r["seq"],
                "unit_id": r["unit_id"],
                "operation": r["operation"],
                "store_target": r["store_target"],
                "payload": json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"],
                "written_at": r["written_at"],
                "status": r["status"],
            }
            for r in rows
        ]

    async def mark_wal_applied(self, seq: int, applied_at: datetime) -> None:
        await self._conn_pool.execute(
            "UPDATE wal SET status = 'applied', applied_at = $1 WHERE seq = $2",
            applied_at, seq,
        )

    async def mark_wal_failed(self, seq: int) -> None:
        await self._conn_pool.execute(
            "UPDATE wal SET status = 'failed' WHERE seq = $1", seq,
        )

    async def compact_wal(self, before: datetime) -> int:
        result = await self._conn_pool.execute(
            "DELETE FROM wal WHERE status = 'applied' AND applied_at < $1", before,
        )
        # asyncpg returns "DELETE N"
        return int(result.split()[-1])

    # ── Checkpoints ──

    async def update_checkpoint(
        self, store_name: str, last_synced_at: datetime, units_synced: int
    ) -> None:
        await self._conn_pool.execute(
            """INSERT INTO sync_checkpoints (store_name, last_synced_at, units_synced)
               VALUES ($1, $2, $3)
               ON CONFLICT(store_name) DO UPDATE SET
                   last_synced_at = EXCLUDED.last_synced_at,
                   units_synced = EXCLUDED.units_synced""",
            store_name, last_synced_at, units_synced,
        )

    async def get_checkpoint(
        self, store_name: str
    ) -> tuple[datetime, int] | None:
        row = await self._conn_pool.fetchrow(
            "SELECT last_synced_at, units_synced FROM sync_checkpoints WHERE store_name = $1",
            store_name,
        )
        if row is None:
            return None
        return (row["last_synced_at"], row["units_synced"])
