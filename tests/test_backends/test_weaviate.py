"""Unit tests for WeaviateBackend — uses mocks to avoid requiring a real Weaviate instance."""

from __future__ import annotations

import pytest

from contextidx.backends.weaviate import WeaviateBackend, _hash_scope


def _weaviate_available() -> bool:
    try:
        import weaviate
        return True
    except ImportError:
        return False


class TestScopeToTenant:
    def test_user_id_maps_to_user_tenant(self):
        backend = WeaviateBackend()
        assert backend._scope_to_tenant({"user_id": "u123"}) == "user_u123"

    def test_empty_scope_maps_to_default(self):
        backend = WeaviateBackend()
        assert backend._scope_to_tenant({}) == "default"
        assert backend._scope_to_tenant(None) == "default"

    def test_scope_without_user_id_uses_hash(self):
        backend = WeaviateBackend()
        tenant = backend._scope_to_tenant({"session_id": "s1"})
        assert tenant.startswith("scope_")
        assert len(tenant) > len("scope_")

    def test_deterministic_hash(self):
        scope = {"agent_id": "a1", "project": "p2"}
        assert _hash_scope(scope) == _hash_scope(scope)

    def test_different_scopes_different_hashes(self):
        h1 = _hash_scope({"a": "1"})
        h2 = _hash_scope({"a": "2"})
        assert h1 != h2


class TestCapabilityFlags:
    def test_supports_metadata_store(self):
        backend = WeaviateBackend()
        assert backend.supports_metadata_store is True

    def test_supports_hybrid_search(self):
        backend = WeaviateBackend()
        assert backend.supports_hybrid_search is True


class TestImportGuard:
    """Verify WeaviateBackend raises ImportError when weaviate-client is missing."""

    @pytest.mark.skipif(
        _weaviate_available(), reason="weaviate-client is installed"
    )
    async def test_initialize_raises_without_weaviate(self):
        backend = WeaviateBackend()
        with pytest.raises(ImportError, match="weaviate-client"):
            await backend.initialize()
