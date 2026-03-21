"""Pinecone backend adapter for contextidx.

Requires optional dependency: pip install contextidx[pinecone]
"""

from __future__ import annotations

from contextidx.backends.base import SearchResult, VectorBackend

try:
    from pinecone import Pinecone, ServerlessSpec
except ImportError as exc:
    raise ImportError(
        "Pinecone backend requires the pinecone client v3+. "
        "Install with: pip install contextidx[pinecone]"
    ) from exc


class PineconeBackend(VectorBackend):
    """Pinecone vector database backend.

    Uses Pinecone's serverless or pod-based indexes via the v3 Python SDK.
    The index must be pre-created or auto-created via ``initialize()``.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "contextidx",
        *,
        namespace: str = "",
        cloud: str = "aws",
        region: str = "us-east-1",
        dimensions: int = 1536,
        metric: str = "cosine",
        create_if_missing: bool = True,
    ):
        self._api_key = api_key
        self._index_name = index_name
        self._namespace = namespace
        self._cloud = cloud
        self._region = region
        self._dimensions = dimensions
        self._metric = metric
        self._create_if_missing = create_if_missing
        self._pc: Pinecone | None = None
        self._index = None  # pinecone.Index

    async def initialize(self) -> None:
        self._pc = Pinecone(api_key=self._api_key)
        existing = [idx.name for idx in self._pc.list_indexes()]
        if self._index_name not in existing and self._create_if_missing:
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dimensions,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        self._index = self._pc.Index(self._index_name)

    async def close(self) -> None:
        self._index = None
        self._pc = None

    def _get_index(self):
        if self._index is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._index

    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        self._get_index().upsert(
            vectors=[(id, embedding, metadata or {})],
            namespace=self._namespace,
        )
        return id

    async def store_batch(
        self,
        items: list[tuple[str, list[float], dict | None]],
    ) -> list[str]:
        if not items:
            return []
        vectors = [(id_, emb, meta or {}) for id_, emb, meta in items]
        self._get_index().upsert(vectors=vectors, namespace=self._namespace)
        return [id_ for id_, _, _ in items]

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        kwargs: dict = {
            "vector": query_embedding,
            "top_k": top_k,
            "include_metadata": True,
            "namespace": self._namespace,
        }
        if filters:
            kwargs["filter"] = _build_pinecone_filter(filters)
        resp = self._get_index().query(**kwargs)
        return [
            SearchResult(
                id=m["id"],
                score=m["score"],
                metadata=m.get("metadata", {}),
            )
            for m in resp.get("matches", [])
        ]

    async def delete(self, id: str) -> None:
        self._get_index().delete(ids=[id], namespace=self._namespace)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        self._get_index().update(
            id=id,
            set_metadata=metadata,
            namespace=self._namespace,
        )


def _build_pinecone_filter(filters: dict) -> dict:
    """Convert scope dict to Pinecone metadata filter.

    Each key is matched with ``$eq``; multiple keys are implicitly AND-ed
    by Pinecone when placed in a flat dict with ``$and``.
    """
    if len(filters) == 1:
        key, val = next(iter(filters.items()))
        return {key: {"$eq": val}}
    return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
