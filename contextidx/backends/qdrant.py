"""Qdrant backend adapter for contextidx.

Requires optional dependency: pip install contextidx[qdrant]
"""

from __future__ import annotations

from contextidx.backends.base import SearchResult, VectorBackend

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )
except ImportError as exc:
    raise ImportError(
        "Qdrant backend requires qdrant-client. "
        "Install with: pip install contextidx[qdrant]"
    ) from exc


class QdrantBackend(VectorBackend):
    """Qdrant vector database backend.

    Uses the async gRPC client for best throughput. Each contextidx deployment
    maps to a single Qdrant collection.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "contextidx",
        dimensions: int = 1536,
        *,
        api_key: str | None = None,
        prefer_grpc: bool = True,
    ):
        self._url = url
        self._collection = collection_name
        self._dimensions = dimensions
        self._api_key = api_key
        self._prefer_grpc = prefer_grpc
        self._client: AsyncQdrantClient | None = None

    async def initialize(self) -> None:
        self._client = AsyncQdrantClient(
            url=self._url,
            api_key=self._api_key,
            prefer_grpc=self._prefer_grpc,
        )
        collections = await self._client.get_collections()
        names = [c.name for c in collections.collections]
        if self._collection not in names:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._dimensions,
                    distance=Distance.COSINE,
                ),
            )

    async def close(self) -> None:
        if self._client:
            await self._client.close()
            self._client = None

    def _get_client(self) -> AsyncQdrantClient:
        if self._client is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._client

    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        point = PointStruct(
            id=id,
            vector=embedding,
            payload=metadata or {},
        )
        await self._get_client().upsert(
            collection_name=self._collection,
            points=[point],
        )
        return id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        qfilter = _build_qdrant_filter(filters) if filters else None
        hits = await self._get_client().search(
            collection_name=self._collection,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=qfilter,
        )
        return [
            SearchResult(
                id=str(h.id),
                score=h.score,
                metadata=h.payload or {},
            )
            for h in hits
        ]

    async def delete(self, id: str) -> None:
        from qdrant_client.models import PointIdsList

        await self._get_client().delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=[id]),
        )

    async def update_metadata(self, id: str, metadata: dict) -> None:
        await self._get_client().set_payload(
            collection_name=self._collection,
            payload=metadata,
            points=[id],
        )


def _build_qdrant_filter(filters: dict) -> Filter:
    must = [
        FieldCondition(key=k, match=MatchValue(value=v))
        for k, v in filters.items()
    ]
    return Filter(must=must)
