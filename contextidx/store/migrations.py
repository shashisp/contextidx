"""SQLite-to-PostgreSQL migration tool for contextidx.

Reads all data from an existing SQLiteStore and writes it into a PostgresStore,
handling the type conversions between SQLite (TEXT dates, JSON-as-TEXT) and
Postgres (TIMESTAMPTZ, JSONB).

Usage::

    from contextidx.store.migrations import migrate_sqlite_to_postgres

    report = await migrate_sqlite_to_postgres(
        sqlite_path=".contextidx/meta.db",
        postgres_dsn="postgresql://user:pass@localhost:5432/contextidx",
    )
    print(report)

Requires optional dependency: pip install contextidx[postgres]
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger("contextidx.migrations")

try:
    import asyncpg
except ImportError as exc:
    raise ImportError(
        "Migration requires asyncpg. Install with: pip install contextidx[postgres]"
    ) from exc


def _str_to_dt(s: str | None) -> datetime | None:
    if s is None:
        return None
    return datetime.fromisoformat(s)


@dataclass
class TableCount:
    table: str
    source: int = 0
    target: int = 0

    @property
    def matches(self) -> bool:
        return self.source == self.target


@dataclass
class MigrationReport:
    tables: list[TableCount] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return all(t.matches for t in self.tables) and not self.errors

    def summary(self) -> str:
        lines = ["Migration Report", "=" * 40]
        for tc in self.tables:
            status = "OK" if tc.matches else "MISMATCH"
            lines.append(f"  {tc.table}: source={tc.source} target={tc.target} [{status}]")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
            for e in self.errors:
                lines.append(f"    - {e}")
        lines.append(f"  Result: {'SUCCESS' if self.success else 'FAILED'}")
        return "\n".join(lines)


_TABLES = ["context_units", "context_graph", "decay_state", "wal", "sync_checkpoints"]


async def _count_sqlite(conn: aiosqlite.Connection, table: str) -> int:
    cursor = await conn.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
    row = await cursor.fetchone()
    return row[0] if row else 0


async def _count_postgres(pool: asyncpg.Pool, table: str) -> int:
    return await pool.fetchval(f"SELECT COUNT(*) FROM {table}")  # noqa: S608


async def migrate_sqlite_to_postgres(
    sqlite_path: str | Path,
    postgres_dsn: str,
    *,
    batch_size: int = 500,
) -> MigrationReport:
    """Migrate all data from a SQLiteStore database to a PostgresStore database.

    The target Postgres database must already have the schema created (i.e.
    ``PostgresStore.initialize()`` must have been called). This function only
    copies data -- it does not create tables.

    Args:
        sqlite_path: Path to the SQLite database file.
        postgres_dsn: PostgreSQL connection string.
        batch_size: Number of rows to insert per batch.

    Returns:
        A ``MigrationReport`` with per-table counts and any errors.
    """
    report = MigrationReport()

    sqlite_conn = await aiosqlite.connect(str(sqlite_path))
    sqlite_conn.row_factory = aiosqlite.Row
    pg_pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=5)

    try:
        await _migrate_context_units(sqlite_conn, pg_pool, batch_size, report)
        await _migrate_context_graph(sqlite_conn, pg_pool, batch_size, report)
        await _migrate_decay_state(sqlite_conn, pg_pool, batch_size, report)
        await _migrate_wal(sqlite_conn, pg_pool, batch_size, report)
        await _migrate_sync_checkpoints(sqlite_conn, pg_pool, batch_size, report)

        # Reset the WAL serial sequence to continue after the max imported seq
        await _reset_wal_sequence(pg_pool)

    except Exception as exc:
        report.errors.append(f"Migration failed: {exc}")
        logger.exception("Migration failed")
    finally:
        await pg_pool.close()
        await sqlite_conn.close()

    if report.success:
        logger.info("Migration completed successfully")
    else:
        logger.warning("Migration completed with issues:\n%s", report.summary())

    return report


async def _migrate_context_units(
    sqlite: aiosqlite.Connection,
    pg: asyncpg.Pool,
    batch_size: int,
    report: MigrationReport,
) -> None:
    table = "context_units"
    source_count = await _count_sqlite(sqlite, table)
    cursor = await sqlite.execute("SELECT * FROM context_units")
    rows = await cursor.fetchall()

    records = []
    for r in rows:
        records.append((
            r["id"],
            r["content"],
            r["embedding"],
            json.dumps(json.loads(r["scope"])) if r["scope"] else "{}",
            float(r["confidence"]),
            float(r["decay_rate"]),
            r["decay_model"],
            r["version"],
            r["source"],
            r["superseded_by"],
            _str_to_dt(r["created_at"]),
            _str_to_dt(r["expires_at"]),
            _str_to_dt(r["archived_at"]),
        ))

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            await pg.executemany(
                """INSERT INTO context_units
                   (id, content, embedding, scope, confidence, decay_rate, decay_model,
                    version, source, superseded_by, created_at, expires_at, archived_at)
                   VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                   ON CONFLICT (id) DO NOTHING""",
                batch,
            )
        except Exception as exc:
            report.errors.append(f"{table} batch {i}: {exc}")

    target_count = await _count_postgres(pg, table)
    report.tables.append(TableCount(table=table, source=source_count, target=target_count))
    logger.info("Migrated %s: %d -> %d rows", table, source_count, target_count)


async def _migrate_context_graph(
    sqlite: aiosqlite.Connection,
    pg: asyncpg.Pool,
    batch_size: int,
    report: MigrationReport,
) -> None:
    table = "context_graph"
    source_count = await _count_sqlite(sqlite, table)
    cursor = await sqlite.execute("SELECT * FROM context_graph")
    rows = await cursor.fetchall()

    records = [
        (r["from_id"], r["to_id"], r["relationship"], _str_to_dt(r["created_at"]))
        for r in rows
    ]

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            await pg.executemany(
                """INSERT INTO context_graph (from_id, to_id, relationship, created_at)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT DO NOTHING""",
                batch,
            )
        except Exception as exc:
            report.errors.append(f"{table} batch {i}: {exc}")

    target_count = await _count_postgres(pg, table)
    report.tables.append(TableCount(table=table, source=source_count, target=target_count))
    logger.info("Migrated %s: %d -> %d rows", table, source_count, target_count)


async def _migrate_decay_state(
    sqlite: aiosqlite.Connection,
    pg: asyncpg.Pool,
    batch_size: int,
    report: MigrationReport,
) -> None:
    table = "decay_state"
    source_count = await _count_sqlite(sqlite, table)
    cursor = await sqlite.execute("SELECT * FROM decay_state")
    rows = await cursor.fetchall()

    records = [
        (r["unit_id"], float(r["current_score"]), _str_to_dt(r["last_updated"]), r["reinforcement_count"])
        for r in rows
    ]

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            await pg.executemany(
                """INSERT INTO decay_state (unit_id, current_score, last_updated, reinforcement_count)
                   VALUES ($1, $2, $3, $4)
                   ON CONFLICT (unit_id) DO NOTHING""",
                batch,
            )
        except Exception as exc:
            report.errors.append(f"{table} batch {i}: {exc}")

    target_count = await _count_postgres(pg, table)
    report.tables.append(TableCount(table=table, source=source_count, target=target_count))
    logger.info("Migrated %s: %d -> %d rows", table, source_count, target_count)


async def _migrate_wal(
    sqlite: aiosqlite.Connection,
    pg: asyncpg.Pool,
    batch_size: int,
    report: MigrationReport,
) -> None:
    table = "wal"
    source_count = await _count_sqlite(sqlite, table)
    cursor = await sqlite.execute("SELECT * FROM wal")
    rows = await cursor.fetchall()

    records = [
        (
            r["unit_id"],
            r["operation"],
            r["store_target"],
            r["payload"],  # already JSON text, Postgres JSONB accepts it
            _str_to_dt(r["written_at"]),
            _str_to_dt(r["applied_at"]),
            r["status"],
        )
        for r in rows
    ]

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            await pg.executemany(
                """INSERT INTO wal (unit_id, operation, store_target, payload, written_at, applied_at, status)
                   VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                batch,
            )
        except Exception as exc:
            report.errors.append(f"{table} batch {i}: {exc}")

    target_count = await _count_postgres(pg, table)
    report.tables.append(TableCount(table=table, source=source_count, target=target_count))
    logger.info("Migrated %s: %d -> %d rows", table, source_count, target_count)


async def _migrate_sync_checkpoints(
    sqlite: aiosqlite.Connection,
    pg: asyncpg.Pool,
    batch_size: int,
    report: MigrationReport,
) -> None:
    table = "sync_checkpoints"
    source_count = await _count_sqlite(sqlite, table)
    cursor = await sqlite.execute("SELECT * FROM sync_checkpoints")
    rows = await cursor.fetchall()

    records = [
        (r["store_name"], _str_to_dt(r["last_synced_at"]), r["units_synced"])
        for r in rows
    ]

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            await pg.executemany(
                """INSERT INTO sync_checkpoints (store_name, last_synced_at, units_synced)
                   VALUES ($1, $2, $3)
                   ON CONFLICT (store_name) DO NOTHING""",
                batch,
            )
        except Exception as exc:
            report.errors.append(f"{table} batch {i}: {exc}")

    target_count = await _count_postgres(pg, table)
    report.tables.append(TableCount(table=table, source=source_count, target=target_count))
    logger.info("Migrated %s: %d -> %d rows", table, source_count, target_count)


async def _reset_wal_sequence(pg: asyncpg.Pool) -> None:
    """Reset the WAL serial sequence to continue after the max imported seq."""
    max_seq = await pg.fetchval("SELECT COALESCE(MAX(seq), 0) FROM wal")
    if max_seq and max_seq > 0:
        await pg.execute(f"SELECT setval('wal_seq_seq', {max_seq})")


async def validate_migration(
    sqlite_path: str | Path,
    postgres_dsn: str,
) -> MigrationReport:
    """Compare row counts across all tables between SQLite and Postgres.

    Useful as a post-migration health check without modifying any data.
    """
    report = MigrationReport()

    sqlite_conn = await aiosqlite.connect(str(sqlite_path))
    sqlite_conn.row_factory = aiosqlite.Row
    pg_pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=2)

    try:
        for table in _TABLES:
            try:
                src = await _count_sqlite(sqlite_conn, table)
                tgt = await _count_postgres(pg_pool, table)
                report.tables.append(TableCount(table=table, source=src, target=tgt))
            except Exception as exc:
                report.errors.append(f"Validation error for {table}: {exc}")
    finally:
        await pg_pool.close()
        await sqlite_conn.close()

    return report
