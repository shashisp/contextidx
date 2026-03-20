from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

from contextidx.core.context_unit import ContextUnit
from contextidx.store.base import Store
from contextidx.store.schema import INDEXES_SQL, SCHEMA_SQL

_ISO = "%Y-%m-%dT%H:%M:%S.%f%z"


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


def _row_to_unit(row: aiosqlite.Row) -> ContextUnit:
    emb_raw = row["embedding"]
    embedding = json.loads(emb_raw) if emb_raw else None
    return ContextUnit(
        id=row["id"],
        content=row["content"],
        embedding=embedding,
        scope=json.loads(row["scope"]),
        confidence=row["confidence"],
        decay_rate=row["decay_rate"],
        decay_model=row["decay_model"],
        version=row["version"],
        source=row["source"],
        superseded_by=row["superseded_by"],
        timestamp=_str_to_dt(row["created_at"]),  # type: ignore[arg-type]
        expires_at=_str_to_dt(row["expires_at"]),
    )


class SQLiteStore(Store):
    """Async SQLite-backed metadata store for contextidx."""

    def __init__(self, path: str | Path = ".contextidx/meta.db"):
        self._path = Path(path)
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(str(self._path))
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(SCHEMA_SQL)
        await self._db.executescript(INDEXES_SQL)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def _conn(self) -> aiosqlite.Connection:
        if self._db is None:
            raise RuntimeError("Store not initialized. Call initialize() first.")
        return self._db

    # ── ContextUnit CRUD ──

    async def create_unit(self, unit: ContextUnit) -> None:
        await self._conn.execute(
            """INSERT INTO context_units
               (id, content, embedding, scope, confidence, decay_rate, decay_model,
                version, source, superseded_by, created_at, expires_at, archived_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
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
                _dt_to_str(unit.timestamp),
                _dt_to_str(unit.expires_at) if unit.expires_at else None,
                None,
            ),
        )
        await self._conn.commit()

    async def get_unit(self, unit_id: str) -> ContextUnit | None:
        cursor = await self._conn.execute(
            "SELECT * FROM context_units WHERE id = ?", (unit_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_unit(row)

    async def update_unit(self, unit_id: str, updates: dict) -> None:
        set_parts: list[str] = []
        values: list = []
        for key, val in updates.items():
            col = key
            if key == "timestamp":
                col = "created_at"
            if key == "scope":
                val = json.dumps(val)
            elif isinstance(val, datetime):
                val = _dt_to_str(val)
            set_parts.append(f"{col} = ?")
            values.append(val)
        if not set_parts:
            return
        values.append(unit_id)
        sql = f"UPDATE context_units SET {', '.join(set_parts)} WHERE id = ?"
        await self._conn.execute(sql, values)
        await self._conn.commit()

    async def delete_unit(self, unit_id: str) -> None:
        await self._conn.execute("DELETE FROM decay_state WHERE unit_id = ?", (unit_id,))
        await self._conn.execute(
            "DELETE FROM context_graph WHERE from_id = ? OR to_id = ?",
            (unit_id, unit_id),
        )
        await self._conn.execute("DELETE FROM context_units WHERE id = ?", (unit_id,))
        await self._conn.commit()

    async def find_units_in_scope(
        self,
        scope: dict[str, str],
        include_superseded: bool = False,
        include_archived: bool = False,
    ) -> list[ContextUnit]:
        query = "SELECT * FROM context_units WHERE 1=1"
        params: list = []

        for key, val in scope.items():
            query += f" AND json_extract(scope, '$.{key}') = ?"
            params.append(val)

        if not include_superseded:
            query += " AND superseded_by IS NULL"
        if not include_archived:
            query += " AND archived_at IS NULL"

        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_unit(r) for r in rows]

    async def find_active_units(
        self,
        since: datetime | None = None,
    ) -> list[ContextUnit]:
        query = "SELECT * FROM context_units WHERE superseded_by IS NULL AND archived_at IS NULL"
        params: list = []
        if since:
            query += " AND created_at >= ?"
            params.append(_dt_to_str(since))
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [_row_to_unit(r) for r in rows]

    # ── Graph ──

    async def add_graph_edge(
        self, from_id: str, to_id: str, relationship: str, created_at: datetime
    ) -> None:
        await self._conn.execute(
            """INSERT OR IGNORE INTO context_graph (from_id, to_id, relationship, created_at)
               VALUES (?, ?, ?, ?)""",
            (from_id, to_id, relationship, _dt_to_str(created_at)),
        )
        await self._conn.commit()

    async def get_graph_edges(
        self, unit_id: str
    ) -> list[tuple[str, str, str, datetime]]:
        cursor = await self._conn.execute(
            "SELECT from_id, to_id, relationship, created_at FROM context_graph WHERE from_id = ? OR to_id = ?",
            (unit_id, unit_id),
        )
        rows = await cursor.fetchall()
        return [
            (r["from_id"], r["to_id"], r["relationship"], _str_to_dt(r["created_at"]))  # type: ignore[misc]
            for r in rows
        ]

    # ── Decay state ──

    async def upsert_decay_state(
        self,
        unit_id: str,
        current_score: float,
        last_updated: datetime,
        reinforcement_count: int,
    ) -> None:
        await self._conn.execute(
            """INSERT INTO decay_state (unit_id, current_score, last_updated, reinforcement_count)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(unit_id) DO UPDATE SET
                   current_score = excluded.current_score,
                   last_updated = excluded.last_updated,
                   reinforcement_count = excluded.reinforcement_count""",
            (unit_id, current_score, _dt_to_str(last_updated), reinforcement_count),
        )
        await self._conn.commit()

    async def get_decay_state(
        self, unit_id: str
    ) -> tuple[float, datetime, int] | None:
        cursor = await self._conn.execute(
            "SELECT current_score, last_updated, reinforcement_count FROM decay_state WHERE unit_id = ?",
            (unit_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return (row["current_score"], _str_to_dt(row["last_updated"]), row["reinforcement_count"])  # type: ignore[return-value]

    async def increment_reinforcement(self, unit_id: str) -> int:
        await self._conn.execute(
            """UPDATE decay_state
               SET reinforcement_count = reinforcement_count + 1,
                   last_updated = ?
               WHERE unit_id = ?""",
            (_dt_to_str(datetime.now(timezone.utc)), unit_id),
        )
        await self._conn.commit()
        state = await self.get_decay_state(unit_id)
        return state[2] if state else 0

    # ── Reconciliation ──

    async def find_active_without_vector(self, since: datetime) -> list[str]:
        query = (
            "SELECT id FROM context_units "
            "WHERE superseded_by IS NULL AND archived_at IS NULL AND created_at >= ?"
        )
        cursor = await self._conn.execute(query, (_dt_to_str(since),))
        rows = await cursor.fetchall()
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
        cursor = await self._conn.execute(
            """INSERT INTO wal (unit_id, operation, store_target, payload, written_at, status)
               VALUES (?, ?, ?, ?, ?, 'pending')""",
            (unit_id, operation, store_target, json.dumps(payload), _dt_to_str(written_at)),
        )
        await self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    async def get_pending_wal(self) -> list[dict]:
        cursor = await self._conn.execute(
            "SELECT * FROM wal WHERE status = 'pending' ORDER BY seq"
        )
        rows = await cursor.fetchall()
        return [
            {
                "seq": r["seq"],
                "unit_id": r["unit_id"],
                "operation": r["operation"],
                "store_target": r["store_target"],
                "payload": json.loads(r["payload"]),
                "written_at": _str_to_dt(r["written_at"]),
                "status": r["status"],
            }
            for r in rows
        ]

    async def mark_wal_applied(self, seq: int, applied_at: datetime) -> None:
        await self._conn.execute(
            "UPDATE wal SET status = 'applied', applied_at = ? WHERE seq = ?",
            (_dt_to_str(applied_at), seq),
        )
        await self._conn.commit()

    async def mark_wal_failed(self, seq: int) -> None:
        await self._conn.execute(
            "UPDATE wal SET status = 'failed' WHERE seq = ?", (seq,)
        )
        await self._conn.commit()

    async def compact_wal(self, before: datetime) -> int:
        cursor = await self._conn.execute(
            "DELETE FROM wal WHERE status = 'applied' AND applied_at < ?",
            (_dt_to_str(before),),
        )
        await self._conn.commit()
        return cursor.rowcount

    # ── Checkpoints ──

    async def update_checkpoint(
        self, store_name: str, last_synced_at: datetime, units_synced: int
    ) -> None:
        await self._conn.execute(
            """INSERT INTO sync_checkpoints (store_name, last_synced_at, units_synced)
               VALUES (?, ?, ?)
               ON CONFLICT(store_name) DO UPDATE SET
                   last_synced_at = excluded.last_synced_at,
                   units_synced = excluded.units_synced""",
            (store_name, _dt_to_str(last_synced_at), units_synced),
        )
        await self._conn.commit()

    async def get_checkpoint(
        self, store_name: str
    ) -> tuple[datetime, int] | None:
        cursor = await self._conn.execute(
            "SELECT last_synced_at, units_synced FROM sync_checkpoints WHERE store_name = ?",
            (store_name,),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return (_str_to_dt(row["last_synced_at"]), row["units_synced"])  # type: ignore[return-value]
