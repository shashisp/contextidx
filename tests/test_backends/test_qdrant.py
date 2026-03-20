"""Unit tests for QdrantBackend — uses mocks to avoid requiring a real Qdrant instance."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _qdrant_available() -> bool:
    try:
        import qdrant_client
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _qdrant_available(), reason="qdrant-client not installed"
)


class TestQdrantBackendImport:
    def test_can_import(self):
        from contextidx.backends.qdrant import QdrantBackend
        assert QdrantBackend is not None

    def test_filter_builder(self):
        from contextidx.backends.qdrant import _build_qdrant_filter
        f = _build_qdrant_filter({"user_id": "u1"})
        assert f.must is not None
        assert len(f.must) == 1

    def test_filter_builder_multi_key(self):
        from contextidx.backends.qdrant import _build_qdrant_filter
        f = _build_qdrant_filter({"user_id": "u1", "session_id": "s1"})
        assert len(f.must) == 2


class TestQdrantBackendCapabilities:
    def test_supports_metadata_store_false(self):
        from contextidx.backends.qdrant import QdrantBackend
        backend = QdrantBackend()
        assert backend.supports_metadata_store is False

    def test_supports_hybrid_search_false(self):
        from contextidx.backends.qdrant import QdrantBackend
        backend = QdrantBackend()
        assert backend.supports_hybrid_search is False

    def test_not_initialized_raises(self):
        from contextidx.backends.qdrant import QdrantBackend
        backend = QdrantBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend._get_client()


class TestQdrantBackendMocked:
    @pytest.fixture
    def mock_backend(self):
        from contextidx.backends.qdrant import QdrantBackend
        backend = QdrantBackend(url="http://localhost:6333", collection_name="test")
        backend._client = AsyncMock()
        return backend

    async def test_store(self, mock_backend):
        result = await mock_backend.store("id1", [0.1, 0.2], {"key": "val"})
        assert result == "id1"
        mock_backend._client.upsert.assert_called_once()

    async def test_search(self, mock_backend):
        mock_hit = MagicMock()
        mock_hit.id = "id1"
        mock_hit.score = 0.95
        mock_hit.payload = {"key": "val"}
        mock_backend._client.search = AsyncMock(return_value=[mock_hit])

        results = await mock_backend.search([0.1, 0.2], top_k=5)
        assert len(results) == 1
        assert results[0].id == "id1"
        assert results[0].score == 0.95

    async def test_search_with_filter(self, mock_backend):
        mock_backend._client.search = AsyncMock(return_value=[])
        results = await mock_backend.search([0.1, 0.2], top_k=5, filters={"user_id": "u1"})
        assert results == []
        call_kwargs = mock_backend._client.search.call_args
        assert call_kwargs.kwargs.get("query_filter") is not None or call_kwargs[1].get("query_filter") is not None

    async def test_delete(self, mock_backend):
        await mock_backend.delete("id1")
        mock_backend._client.delete.assert_called_once()

    async def test_update_metadata(self, mock_backend):
        await mock_backend.update_metadata("id1", {"new_key": "new_val"})
        mock_backend._client.set_payload.assert_called_once()

    async def test_close(self, mock_backend):
        await mock_backend.close()
        assert mock_backend._client is None
