"""Tests for scope key validation (SQL injection prevention)."""

import pytest

from contextidx.core.context_unit import ContextUnit
from contextidx.store.base import validate_scope_keys
from contextidx.store.sqlite_store import SQLiteStore


@pytest.fixture
async def store(tmp_path):
    s = SQLiteStore(path=tmp_path / "scope_test.db")
    await s.initialize()
    yield s
    await s.close()


def _unit(uid: str = "u1", scope: dict | None = None) -> ContextUnit:
    return ContextUnit(
        id=uid,
        content="test",
        scope=scope or {"user_id": "u1"},
        confidence=0.9,
        source="test",
    )


class TestValidateScopeKeys:
    def test_valid_simple_key(self):
        validate_scope_keys({"user_id": "u1"})

    def test_valid_multiple_keys(self):
        validate_scope_keys({"user_id": "u1", "session_id": "s1", "agent_id": "a1"})

    def test_valid_underscore_prefix(self):
        validate_scope_keys({"_private": "val"})

    def test_valid_mixed_case(self):
        validate_scope_keys({"UserId": "val", "sessionID": "val"})

    def test_rejects_sql_injection_semicolon(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"'; DROP TABLE context_units; --": "val"})

    def test_rejects_dot_traversal(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"../etc": "val"})

    def test_rejects_spaces(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"user id": "val"})

    def test_rejects_quotes(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"user'id": "val"})

    def test_rejects_dash(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"user-id": "val"})

    def test_rejects_empty_key(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"": "val"})

    def test_rejects_numeric_start(self):
        with pytest.raises(ValueError, match="Invalid scope key"):
            validate_scope_keys({"1invalid": "val"})

    def test_empty_scope_passes(self):
        validate_scope_keys({})


class TestScopeValidationInStore:
    async def test_find_units_rejects_malicious_scope(self, store):
        await store.create_unit(_unit())
        with pytest.raises(ValueError, match="Invalid scope key"):
            await store.find_units_in_scope({"'; DROP TABLE x;--": "val"})

    async def test_find_units_accepts_valid_scope(self, store):
        await store.create_unit(_unit())
        results = await store.find_units_in_scope({"user_id": "u1"})
        assert len(results) == 1
