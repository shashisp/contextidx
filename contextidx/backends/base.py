from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    """Single result from a vector similarity search."""

    id: str
    score: float
    metadata: dict = field(default_factory=dict)


class VectorBackend(ABC):
    """Abstract interface for vector storage backends.

    All vector DB operations are abstracted behind these four methods plus
    one capability flag. Custom backends require implementing only this.
    """

    @property
    def supports_metadata_store(self) -> bool:
        """Whether this backend can store contextidx metadata natively.

        If True, contextidx *may* use the backend itself for metadata
        instead of an external SQLite/Postgres store.
        """
        return False

    @abstractmethod
    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        """Persist an embedding vector. Returns the stored ID."""

    @abstractmethod
    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar vectors, scored in [0, 1]."""

    @abstractmethod
    async def delete(self, id: str) -> None:
        """Remove a vector by ID."""

    @abstractmethod
    async def update_metadata(self, id: str, metadata: dict) -> None:
        """Patch metadata for an existing vector."""

    @property
    def supports_hybrid_search(self) -> bool:
        """Whether this backend natively supports BM25 + vector hybrid search."""
        return False

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Hybrid BM25 + vector search.

        Default falls back to vector-only search. Backends that support hybrid
        search (e.g. Weaviate) should override this.

        Args:
            query: Raw text query for BM25 matching.
            query_embedding: Pre-computed query embedding for vector search.
            top_k: Number of results.
            filters: Scope/metadata filters.
            alpha: Balance between vector (1.0) and BM25 (0.0).
        """
        return await self.search(query_embedding, top_k, filters)

    async def initialize(self) -> None:
        """Optional setup hook (create tables, extensions, etc.)."""

    async def close(self) -> None:
        """Optional teardown hook."""
