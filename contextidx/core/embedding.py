"""Pluggable embedding interface for contextidx.

Users can provide any async callable that conforms to :class:`EmbeddingFunction`
to use custom embedding models (Sentence Transformers, Cohere, Azure OpenAI, etc.).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class EmbeddingFunction(Protocol):
    """Protocol for embedding providers.

    Any object with ``embed`` and ``embed_batch`` async methods satisfying
    these signatures can be used as an embedding function.

    Example::

        class MyEmbedder:
            async def embed(self, text: str) -> list[float]:
                return my_model.encode(text).tolist()

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [my_model.encode(t).tolist() for t in texts]

        idx = ContextIdx(backend=..., embedding_fn=MyEmbedder())
    """

    async def embed(self, text: str) -> list[float]: ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
