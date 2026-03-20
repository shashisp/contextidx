"""Unit tests for PostgresStore using mocks — no real Postgres required.

Tests marked with @pytest.mark.postgres require a running PostgreSQL instance
and are skipped by default.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contextidx.core.context_unit import ContextUnit, generate_unit_id


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
        "content": "test context",
        "scope": {"user_id": "u1"},
        "confidence": 0.9,
        "source": "test",
        "decay_model": "exponential",
        "decay_rate": 0.02,
    }
    defaults.update(overrides)
    return ContextUnit(**defaults)


def _make_record(unit: ContextUnit) -> dict:
    """Create a dict that mimics an asyncpg.Record for row_to_unit conversion."""
    return {
        "id": unit.id,
        "content": unit.content,
        "embedding": json.dumps(unit.embedding) if unit.embedding else None,
        "scope": unit.scope,
        "confidence": unit.confidence,
        "decay_rate": unit.decay_rate,
        "decay_model": unit.decay_model,
        "version": unit.version,
        "source": unit.source,
        "superseded_by": unit.superseded_by,
        "created_at": unit.timestamp,
        "expires_at": unit.expires_at,
        "archived_at": None,
    }


class TestPostgresStoreImport:
    def test_can_import(self):
        from contextidx.store.postgres_store import PostgresStore
        assert PostgresStore is not None

    def test_schema_has_postgres_types(self):
        from contextidx.store.postgres_schema import SCHEMA_SQL
        assert "SERIAL" in SCHEMA_SQL
        assert "JSONB" in SCHEMA_SQL
        assert "TIMESTAMPTZ" in SCHEMA_SQL

    def test_schema_has_all_tables(self):
        from contextidx.store.postgres_schema import SCHEMA_SQL
        assert "context_units" in SCHEMA_SQL
        assert "context_graph" in SCHEMA_SQL
        assert "decay_state" in SCHEMA_SQL
        assert "wal" in SCHEMA_SQL
        assert "sync_checkpoints" in SCHEMA_SQL


class TestPostgresStoreInit:
    def test_constructor(self):
        from contextidx.store.postgres_store import PostgresStore
        store = PostgresStore(dsn="postgresql://localhost/test")
        assert store._dsn == "postgresql://localhost/test"
        assert store._pool is None

    def test_not_initialized_raises(self):
        from contextidx.store.postgres_store import PostgresStore
        store = PostgresStore(dsn="postgresql://localhost/test")
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = store._conn_pool


class TestPostgresStoreRowConversion:
    def test_row_to_unit(self):
        from contextidx.store.postgres_store import _row_to_unit
        unit = _make_unit()
        unit.embedding = [0.1, 0.2, 0.3]
        record = _make_record(unit)
        result = _row_to_unit(record)
        assert result.id == unit.id
        assert result.content == unit.content
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.scope == unit.scope

    def test_row_to_unit_no_embedding(self):
        from contextidx.store.postgres_store import _row_to_unit
        unit = _make_unit()
        record = _make_record(unit)
        result = _row_to_unit(record)
        assert result.embedding is None

    def test_row_to_unit_json_scope(self):
        from contextidx.store.postgres_store import _row_to_unit
        unit = _make_unit()
        record = _make_record(unit)
        record["scope"] = json.dumps({"user_id": "u1"})
        result = _row_to_unit(record)
        assert result.scope == {"user_id": "u1"}


class TestPostgresStoreMocked:
    """Tests with a mocked asyncpg pool."""

    @pytest.fixture
    def mock_store(self):
        from contextidx.store.postgres_store import PostgresStore
        store = PostgresStore(dsn="postgresql://localhost/test")
        store._pool = MagicMock()
        store._pool.execute = AsyncMock()
        store._pool.fetch = AsyncMock(return_value=[])
        store._pool.fetchrow = AsyncMock(return_value=None)
        store._pool.close = AsyncMock()
        return store

    async def test_create_unit_calls_execute(self, mock_store):
        unit = _make_unit()
        await mock_store.create_unit(unit)
        mock_store._pool.execute.assert_called_once()
        call_sql = mock_store._pool.execute.call_args[0][0]
        assert "INSERT INTO context_units" in call_sql

    async def test_get_unit_returns_none(self, mock_store):
        result = await mock_store.get_unit("nonexistent")
        assert result is None

    async def test_get_unit_returns_unit(self, mock_store):
        unit = _make_unit()
        unit.embedding = [0.1, 0.2]
        mock_store._pool.fetchrow = AsyncMock(return_value=_make_record(unit))
        result = await mock_store.get_unit(unit.id)
        assert result is not None
        assert result.id == unit.id

    async def test_update_unit_empty_updates(self, mock_store):
        await mock_store.update_unit("u1", {})
        mock_store._pool.execute.assert_not_called()

    async def test_update_unit_with_updates(self, mock_store):
        await mock_store.update_unit("u1", {"confidence": 0.5})
        mock_store._pool.execute.assert_called_once()

    async def test_find_units_in_scope_empty(self, mock_store):
        result = await mock_store.find_units_in_scope({"user_id": "u1"})
        assert result == []

    async def test_append_wal(self, mock_store):
        mock_store._pool.fetchrow = AsyncMock(return_value={"seq": 42})
        seq = await mock_store.append_wal(
            unit_id="u1",
            operation="store",
            store_target="both",
            payload={"test": True},
            written_at=datetime.now(timezone.utc),
        )
        assert seq == 42

    async def test_compact_wal(self, mock_store):
        mock_store._pool.execute = AsyncMock(return_value="DELETE 5")
        removed = await mock_store.compact_wal(datetime.now(timezone.utc))
        assert removed == 5

    async def test_close(self, mock_store):
        await mock_store.close()
        mock_store._pool.close.assert_called_once()
        assert mock_store._pool is None

    async def test_upsert_decay_state(self, mock_store):
        now = datetime.now(timezone.utc)
        await mock_store.upsert_decay_state("u1", 0.8, now, 3)
        mock_store._pool.execute.assert_called_once()
        call_sql = mock_store._pool.execute.call_args[0][0]
        assert "ON CONFLICT" in call_sql

    async def test_update_checkpoint(self, mock_store):
        now = datetime.now(timezone.utc)
        await mock_store.update_checkpoint("vector_backend", now, 100)
        mock_store._pool.execute.assert_called_once()
