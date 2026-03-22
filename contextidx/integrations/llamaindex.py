"""LlamaIndex integration for contextidx.

Install extras:
    pip install contextidx[llamaindex]

Usage::

    from contextidx import ContextIdx
    from contextidx.backends.qdrant import QdrantBackend
    from contextidx.integrations.llamaindex import ContextIdxRetriever
    from llama_index.core import QueryEngine

    async def main():
        ctx = ContextIdx(backend=QdrantBackend(...))
        await ctx.ainitialize()

        retriever = ContextIdxRetriever(ctx=ctx, scope={"user": "u_42"}, top_k=8)
        nodes = await retriever.aretrieve("What are the project deadlines?")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

try:
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "LlamaIndex is not installed. Run: pip install contextidx[llamaindex]"
    ) from exc

if TYPE_CHECKING:
    from contextidx.contextidx import ContextIdx


class ContextIdxRetriever(BaseRetriever):
    """LlamaIndex ``BaseRetriever`` backed by contextidx temporal storage.

    Wraps ``ContextIdx.aretrieve()`` and maps each ``ContextUnit`` to a
    ``NodeWithScore``, preserving temporal scores for downstream rerankers.

    Parameters
    ----------
    ctx:
        An initialised ``ContextIdx`` instance.
    scope:
        Scope dict used to filter retrieval to a specific session/user/thread.
    top_k:
        Number of context units to retrieve.
    score_threshold:
        Minimum composite score (0–1) for a unit to be included.
        Units below this threshold are filtered out. Default: ``0.0`` (no filter).
    """

    def __init__(
        self,
        ctx: ContextIdx,
        scope: Optional[dict[str, str]] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._ctx = ctx
        self._scope = scope or {}
        self._top_k = top_k
        self._score_threshold = score_threshold
        super().__init__(**kwargs)

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Synchronous retrieval — delegates to ``_aretrieve`` via asyncio."""
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            raise RuntimeError(
                "ContextIdxRetriever._retrieve() cannot be called from an async context. "
                "Use `await retriever.aretrieve(query)` instead."
            )
        except RuntimeError as exc:
            if "cannot be called" in str(exc):
                raise
            return asyncio.run(self._aretrieve(query_bundle))

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve contextidx units and map them to LlamaIndex ``NodeWithScore`` objects."""
        query = query_bundle.query_str
        units = await self._ctx.aretrieve(
            query, top_k=self._top_k, scope=self._scope
        )

        nodes: list[NodeWithScore] = []
        for unit in units:
            # Extract the composite score if present in metadata (set by scoring engine)
            score: float = unit.metadata.get("composite_score", unit.confidence)  # type: ignore[union-attr]
            if score < self._score_threshold:
                continue

            node = TextNode(
                text=unit.content,
                id_=unit.id,
                metadata={
                    "unit_id": unit.id,
                    "scope": unit.scope,
                    "confidence": unit.confidence,
                    "age_days": unit.age_days,
                    "created_at": unit.created_at.isoformat(),
                    **(unit.metadata or {}),
                },
            )
            nodes.append(NodeWithScore(node=node, score=score))

        return nodes
