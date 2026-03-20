"""Tests for the SQLite-to-Postgres migration tool.

Tests use a real SQLiteStore populated with data and mock the asyncpg pool
to verify the migration logic without requiring a running PostgreSQL instance.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.store.sqlite_store import SQLiteStore


def _asyncpg_available() -> bool:
    try:
        import asyncpg
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _asyncpg_available(), reason="asyncpg not installed"
)


def _make_unit(**overrides) -> ContextUnit:
    defaults = {
        "id": generate_unit_id(),
        "content": "migration test context",
        "scope": {"user_id": "u1"},
        "confidence": 0.85,
        "source": "test",
        "decay_model": "exponential",
        "decay_rate": 0.02,
    }
    defaults.update(overrides)
    return ContextUnit(**defaults)


class TestMigrationReport:
    def test_empty_report_is_success(self):
        from contextidx.store.migrations import MigrationReport
        report = MigrationReport()
        assert report.success is True

    def test_matching_counts_is_success(self):
        from contextidx.store.migrations import MigrationReport, TableCount
        report = MigrationReport(tables=[
            TableCount(table="context_units", source=10, target=10),
            TableCount(table="context_graph", source=5, target=5),
        ])
        assert report.success is True

    def test_mismatching_counts_is_failure(self):
        from contextidx.store.migrations import MigrationReport, TableCount
        report = MigrationReport(tables=[
            TableCount(table="context_units", source=10, target=8),
        ])
        assert report.success is False

    def test_errors_cause_failure(self):
        from contextidx.store.migrations import MigrationReport, TableCount
        report = MigrationReport(
            tables=[TableCount(table="context_units", source=10, target=10)],
            errors=["something went wrong"],
        )
        assert report.success is False

    def test_summary_contains_table_names(self):
        from contextidx.store.migrations import MigrationReport, TableCount
        report = MigrationReport(tables=[
            TableCount(table="context_units", source=10, target=10),
        ])
        summary = report.summary()
        assert "context_units" in summary
        assert "SUCCESS" in summary

    def test_summary_shows_mismatch(self):
        from contextidx.store.migrations import MigrationReport, TableCount
        report = MigrationReport(tables=[
            TableCount(table="wal", source=5, target=3),
        ])
        summary = report.summary()
        assert "MISMATCH" in summary
        assert "FAILED" in summary


class TestTableCount:
    def test_matches_true(self):
        from contextidx.store.migrations import TableCount
        tc = TableCount(table="test", source=10, target=10)
        assert tc.matches is True

    def test_matches_false(self):
        from contextidx.store.migrations import TableCount
        tc = TableCount(table="test", source=10, target=8)
        assert tc.matches is False


@pytest.fixture
async def populated_sqlite(tmp_path):
    """Create and populate a SQLiteStore with test data."""
    store = SQLiteStore(path=tmp_path / "migration_src.db")
    await store.initialize()

    unit1 = _make_unit(id="ctx_unit1")
    unit1.embedding = [0.1, 0.2, 0.3]
    unit2 = _make_unit(id="ctx_unit2", scope={"user_id": "u2"})

    await store.create_unit(unit1)
    await store.create_unit(unit2)

    now = datetime.now(timezone.utc)
    await store.add_graph_edge("ctx_unit2", "ctx_unit1", "supersedes", now)
    await store.upsert_decay_state("ctx_unit1", 0.9, now, 2)
    await store.upsert_decay_state("ctx_unit2", 0.8, now, 0)

    await store.append_wal("ctx_unit1", "store", "both", {"test": True}, now)
    await store.mark_wal_applied(1, now)
    await store.append_wal("ctx_unit2", "store", "both", {"test": True}, now)

    await store.update_checkpoint("vector_backend", now, 2)

    yield store, tmp_path / "migration_src.db"
    await store.close()


def _mock_pg_pool():
    """Create a mock asyncpg pool that tracks executemany calls."""
    pool = AsyncMock()
    pool.executemany = AsyncMock()
    pool.fetchval = AsyncMock(return_value=0)
    pool.execute = AsyncMock()
    pool.close = AsyncMock()
    return pool


class TestMigrateSqliteToPostgres:
    async def test_migration_calls_executemany_for_each_table(self, populated_sqlite):
        from contextidx.store.migrations import migrate_sqlite_to_postgres
        store, db_path = populated_sqlite

        mock_pool = _mock_pg_pool()
        # Return correct counts for validation after insert
        call_count = [0]
        async def _fake_fetchval(sql):
            call_count[0] += 1
            if "context_units" in sql:
                return 2
            if "context_graph" in sql:
                return 1
            if "decay_state" in sql:
                return 2
            if "wal" in sql:
                return 2
            if "sync_checkpoints" in sql:
                return 1
            return 0

        mock_pool.fetchval = _fake_fetchval

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await migrate_sqlite_to_postgres(db_path, "postgresql://fake")

        assert len(report.tables) == 5
        assert mock_pool.executemany.call_count >= 5

    async def test_migration_reports_correct_source_counts(self, populated_sqlite):
        from contextidx.store.migrations import migrate_sqlite_to_postgres
        store, db_path = populated_sqlite

        mock_pool = _mock_pg_pool()
        mock_pool.fetchval = AsyncMock(return_value=0)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await migrate_sqlite_to_postgres(db_path, "postgresql://fake")

        table_map = {t.table: t for t in report.tables}
        assert table_map["context_units"].source == 2
        assert table_map["context_graph"].source == 1
        assert table_map["decay_state"].source == 2
        assert table_map["wal"].source == 2
        assert table_map["sync_checkpoints"].source == 1

    async def test_migration_handles_executemany_error(self, populated_sqlite):
        from contextidx.store.migrations import migrate_sqlite_to_postgres
        _, db_path = populated_sqlite

        mock_pool = _mock_pg_pool()
        mock_pool.executemany = AsyncMock(side_effect=Exception("connection error"))
        mock_pool.fetchval = AsyncMock(return_value=0)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await migrate_sqlite_to_postgres(db_path, "postgresql://fake")

        assert len(report.errors) > 0

    async def test_migration_resets_wal_sequence(self, populated_sqlite):
        from contextidx.store.migrations import migrate_sqlite_to_postgres
        _, db_path = populated_sqlite

        mock_pool = _mock_pg_pool()
        mock_pool.fetchval = AsyncMock(return_value=2)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            await migrate_sqlite_to_postgres(db_path, "postgresql://fake")

        setval_calls = [
            c for c in mock_pool.execute.call_args_list
            if "setval" in str(c)
        ]
        assert len(setval_calls) == 1


class TestMigrateEmptyDatabase:
    async def test_empty_sqlite_produces_zero_counts(self, tmp_path):
        from contextidx.store.migrations import migrate_sqlite_to_postgres

        empty_store = SQLiteStore(path=tmp_path / "empty.db")
        await empty_store.initialize()
        await empty_store.close()

        mock_pool = _mock_pg_pool()
        mock_pool.fetchval = AsyncMock(return_value=0)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await migrate_sqlite_to_postgres(
                tmp_path / "empty.db", "postgresql://fake"
            )

        assert report.success is True
        for tc in report.tables:
            assert tc.source == 0
            assert tc.target == 0


class TestValidateMigration:
    async def test_validate_matching(self, tmp_path):
        from contextidx.store.migrations import validate_migration

        empty_store = SQLiteStore(path=tmp_path / "validate.db")
        await empty_store.initialize()
        await empty_store.close()

        mock_pool = _mock_pg_pool()
        mock_pool.fetchval = AsyncMock(return_value=0)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await validate_migration(
                tmp_path / "validate.db", "postgresql://fake"
            )

        assert report.success is True
        assert len(report.tables) == 5

    async def test_validate_mismatch(self, populated_sqlite):
        from contextidx.store.migrations import validate_migration
        _, db_path = populated_sqlite

        mock_pool = _mock_pg_pool()
        mock_pool.fetchval = AsyncMock(return_value=0)

        with patch("contextidx.store.migrations.asyncpg") as mock_asyncpg:
            mock_asyncpg.create_pool = AsyncMock(return_value=mock_pool)
            report = await validate_migration(db_path, "postgresql://fake")

        assert report.success is False
        table_map = {t.table: t for t in report.tables}
        assert table_map["context_units"].source == 2
        assert table_map["context_units"].target == 0
