"""ChromaDB backend adapter for contextidx.

Requires optional dependency: pip install contextidx[chroma]
"""

from __future__ import annotations

from contextidx.backends.base import SearchResult, VectorBackend

try:
    import chromadb
    from chromadb.api.models.Collection import Collection
except ImportError as exc:
    raise ImportError(
        "Chroma backend requires chromadb. "
        "Install with: pip install contextidx[chroma]"
    ) from exc


class ChromaBackend(VectorBackend):
    """ChromaDB vector database backend.

    Uses the Chroma HTTP client for client/server deployments. Falls back
    to an ephemeral in-process client when ``host`` is ``None``.
    """

    def __init__(
        self,
        collection_name: str = "contextidx",
        *,
        host: str | None = None,
        port: int = 8000,
        ssl: bool = False,
        tenant: str = chromadb.DEFAULT_TENANT,
        database: str = chromadb.DEFAULT_DATABASE,
    ):
        self._collection_name = collection_name
        self._host = host
        self._port = port
        self._ssl = ssl
        self._tenant = tenant
        self._database = database
        self._client: chromadb.ClientAPI | None = None
        self._collection: Collection | None = None

    async def initialize(self) -> None:
        if self._host:
            self._client = chromadb.HttpClient(
                host=self._host,
                port=self._port,
                ssl=self._ssl,
                tenant=self._tenant,
                database=self._database,
            )
        else:
            self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def close(self) -> None:
        self._collection = None
        self._client = None

    def _get_collection(self) -> Collection:
        if self._collection is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._collection

    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        self._get_collection().upsert(
            ids=[id],
            embeddings=[embedding],
            metadatas=[metadata or {}],
        )
        return id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        where = _build_chroma_where(filters) if filters else None
        results = self._get_collection().query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        out: list[SearchResult] = []
        for i, doc_id in enumerate(ids):
            score = 1.0 - distances[i] if distances else 0.0
            out.append(
                SearchResult(
                    id=doc_id,
                    score=max(0.0, score),
                    metadata=metadatas[i] if metadatas else {},
                )
            )
        return out

    async def delete(self, id: str) -> None:
        self._get_collection().delete(ids=[id])

    async def update_metadata(self, id: str, metadata: dict) -> None:
        self._get_collection().update(ids=[id], metadatas=[metadata])


def _build_chroma_where(filters: dict) -> dict:
    """Convert scope dict to Chroma ``where`` clause.

    Single key maps directly; multiple keys are combined with ``$and``.
    """
    if len(filters) == 1:
        key, val = next(iter(filters.items()))
        return {key: {"$eq": val}}
    return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
