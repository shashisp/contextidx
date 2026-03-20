"""Unit tests for PineconeBackend — uses mocks to avoid requiring a real Pinecone account."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _pinecone_available() -> bool:
    try:
        import pinecone
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _pinecone_available(), reason="pinecone not installed"
)


class TestPineconeBackendImport:
    def test_can_import(self):
        from contextidx.backends.pinecone import PineconeBackend
        assert PineconeBackend is not None

    def test_filter_builder_single(self):
        from contextidx.backends.pinecone import _build_pinecone_filter
        result = _build_pinecone_filter({"user_id": "u1"})
        assert result == {"user_id": {"$eq": "u1"}}

    def test_filter_builder_multi(self):
        from contextidx.backends.pinecone import _build_pinecone_filter
        result = _build_pinecone_filter({"user_id": "u1", "session_id": "s1"})
        assert "$and" in result
        assert len(result["$and"]) == 2


class TestPineconeBackendCapabilities:
    def test_supports_metadata_store_false(self):
        from contextidx.backends.pinecone import PineconeBackend
        backend = PineconeBackend(api_key="test-key")
        assert backend.supports_metadata_store is False

    def test_supports_hybrid_search_false(self):
        from contextidx.backends.pinecone import PineconeBackend
        backend = PineconeBackend(api_key="test-key")
        assert backend.supports_hybrid_search is False

    def test_not_initialized_raises(self):
        from contextidx.backends.pinecone import PineconeBackend
        backend = PineconeBackend(api_key="test-key")
        with pytest.raises(RuntimeError, match="not initialized"):
            backend._get_index()


class TestPineconeBackendMocked:
    @pytest.fixture
    def mock_backend(self):
        from contextidx.backends.pinecone import PineconeBackend
        backend = PineconeBackend(api_key="test-key", index_name="test")
        backend._index = MagicMock()
        return backend

    async def test_store(self, mock_backend):
        result = await mock_backend.store("id1", [0.1, 0.2], {"key": "val"})
        assert result == "id1"
        mock_backend._index.upsert.assert_called_once()
        call_kwargs = mock_backend._index.upsert.call_args
        vectors = call_kwargs.kwargs.get("vectors") or call_kwargs[1].get("vectors")
        assert vectors[0][0] == "id1"

    async def test_search(self, mock_backend):
        mock_backend._index.query.return_value = {
            "matches": [
                {"id": "id1", "score": 0.95, "metadata": {"k": "v"}},
                {"id": "id2", "score": 0.80, "metadata": {}},
            ]
        }
        results = await mock_backend.search([0.1, 0.2], top_k=5)
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == 0.95

    async def test_search_with_filter(self, mock_backend):
        mock_backend._index.query.return_value = {"matches": []}
        await mock_backend.search([0.1], top_k=5, filters={"user_id": "u1"})
        call_kwargs = mock_backend._index.query.call_args.kwargs
        assert "filter" in call_kwargs

    async def test_delete(self, mock_backend):
        await mock_backend.delete("id1")
        mock_backend._index.delete.assert_called_once()

    async def test_update_metadata(self, mock_backend):
        await mock_backend.update_metadata("id1", {"new": "val"})
        mock_backend._index.update.assert_called_once()

    async def test_close(self, mock_backend):
        await mock_backend.close()
        assert mock_backend._index is None
        assert mock_backend._pc is None
