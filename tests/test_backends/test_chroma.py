"""Unit tests for ChromaBackend — uses mocks to avoid requiring a real Chroma server."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _chroma_available() -> bool:
    try:
        import chromadb
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _chroma_available(), reason="chromadb not installed"
)


class TestChromaBackendImport:
    def test_can_import(self):
        from contextidx.backends.chroma import ChromaBackend
        assert ChromaBackend is not None

    def test_filter_builder_single(self):
        from contextidx.backends.chroma import _build_chroma_where
        result = _build_chroma_where({"user_id": "u1"})
        assert result == {"user_id": {"$eq": "u1"}}

    def test_filter_builder_multi(self):
        from contextidx.backends.chroma import _build_chroma_where
        result = _build_chroma_where({"user_id": "u1", "session_id": "s1"})
        assert "$and" in result
        assert len(result["$and"]) == 2


class TestChromaBackendCapabilities:
    def test_supports_metadata_store_false(self):
        from contextidx.backends.chroma import ChromaBackend
        backend = ChromaBackend()
        assert backend.supports_metadata_store is False

    def test_supports_hybrid_search_false(self):
        from contextidx.backends.chroma import ChromaBackend
        backend = ChromaBackend()
        assert backend.supports_hybrid_search is False

    def test_not_initialized_raises(self):
        from contextidx.backends.chroma import ChromaBackend
        backend = ChromaBackend()
        with pytest.raises(RuntimeError, match="not initialized"):
            backend._get_collection()


class TestChromaBackendMocked:
    @pytest.fixture
    def mock_backend(self):
        from contextidx.backends.chroma import ChromaBackend
        backend = ChromaBackend(collection_name="test")
        backend._collection = MagicMock()
        return backend

    async def test_store(self, mock_backend):
        result = await mock_backend.store("id1", [0.1, 0.2], {"key": "val"})
        assert result == "id1"
        mock_backend._collection.upsert.assert_called_once()

    async def test_search(self, mock_backend):
        mock_backend._collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"key": "v1"}, {"key": "v2"}]],
        }
        results = await mock_backend.search([0.1, 0.2], top_k=5)
        assert len(results) == 2
        assert results[0].id == "id1"
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.7)

    async def test_search_with_filter(self, mock_backend):
        mock_backend._collection.query.return_value = {
            "ids": [[]], "distances": [[]], "metadatas": [[]],
        }
        results = await mock_backend.search([0.1], top_k=5, filters={"user_id": "u1"})
        assert results == []
        call_kwargs = mock_backend._collection.query.call_args
        assert call_kwargs.kwargs.get("where") is not None or call_kwargs[1].get("where") is not None

    async def test_delete(self, mock_backend):
        await mock_backend.delete("id1")
        mock_backend._collection.delete.assert_called_once_with(ids=["id1"])

    async def test_update_metadata(self, mock_backend):
        await mock_backend.update_metadata("id1", {"new": "val"})
        mock_backend._collection.update.assert_called_once()

    async def test_close(self, mock_backend):
        await mock_backend.close()
        assert mock_backend._collection is None
        assert mock_backend._client is None
