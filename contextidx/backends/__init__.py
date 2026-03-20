from contextidx.backends.base import VectorBackend, SearchResult

__all__ = ["VectorBackend", "SearchResult"]

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
