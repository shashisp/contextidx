"""contextidx - A temporal context layer for your existing vector database."""

from contextidx.contextidx import ContextIdx
from contextidx.core.context_unit import ContextUnit
from contextidx.backends.base import VectorBackend, SearchResult

__all__ = ["ContextIdx", "ContextUnit", "VectorBackend", "SearchResult"]
__version__ = "1.0.0"

# Conditionally export backends that have their dependencies installed
try:
    from contextidx.backends.pgvector import PGVectorBackend
    __all__.append("PGVectorBackend")
except ImportError:
    pass

try:
    from contextidx.backends.weaviate import WeaviateBackend
    __all__.append("WeaviateBackend")
except ImportError:
    pass

try:
    from contextidx.backends.qdrant import QdrantBackend
    __all__.append("QdrantBackend")
except ImportError:
    pass

try:
    from contextidx.backends.chroma import ChromaBackend
    __all__.append("ChromaBackend")
except ImportError:
    pass

try:
    from contextidx.backends.pinecone import PineconeBackend
    __all__.append("PineconeBackend")
except ImportError:
    pass

try:
    from contextidx.store.postgres_store import PostgresStore
    __all__.append("PostgresStore")
except ImportError:
    pass
