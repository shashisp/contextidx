# Backend Setup Guide

contextidx supports multiple vector database backends. Each implements the `VectorBackend` ABC and can be swapped in with a single-line change.

## Backend Comparison

| Backend | Hybrid Search | Native Metadata | Setup Complexity |
|---------|:---:|:---:|:---:|
| PGVector | No | Yes | Medium |
| Weaviate | Yes | No | Medium |
| Qdrant | No | No | Low |
| ChromaDB | No | No | Low |
| Pinecone | No | No | Low (managed) |

## PGVector

PostgreSQL with the `pgvector` extension. Best when you already run Postgres.

```bash
pip install contextidx[pgvector]
```

```python
from contextidx.backends.pgvector import PGVectorBackend

backend = PGVectorBackend(
    conn_string="postgresql://user:pass@localhost:5432/mydb",
    table_name="contextidx_vectors",  # default
    dimensions=1536,                   # must match embedding model
)
```

**Prerequisites:** PostgreSQL 15+ with `pgvector` extension installed.

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

PGVector is the only backend where `supports_metadata_store = True`, meaning contextidx can use the same Postgres instance for both vectors and metadata.

## Weaviate

Weaviate offers native BM25 + vector hybrid search and multi-tenancy.

```bash
pip install contextidx[weaviate]
```

```python
from contextidx.backends.weaviate import WeaviateBackend

backend = WeaviateBackend(
    url="http://localhost:8080",
    collection_name="ContextIdx",
    api_key=None,          # if auth is enabled
    multi_tenant=False,    # enable for per-user isolation
)
```

With hybrid search enabled, the scoring engine automatically incorporates BM25 keyword relevance alongside vector similarity.

## Qdrant

High-performance vector DB with gRPC support.

```bash
pip install contextidx[qdrant]
```

```python
from contextidx.backends.qdrant import QdrantBackend

backend = QdrantBackend(
    url="http://localhost:6333",
    collection_name="contextidx",
    dimensions=1536,
    api_key=None,           # for Qdrant Cloud
    prefer_grpc=True,       # faster than REST
)
```

**Local development:**

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

The collection is auto-created on `initialize()` if it doesn't exist, using cosine distance.

## ChromaDB

Lightweight, easy to get started. Good for development and small-scale deployments.

```bash
pip install contextidx[chroma]
```

```python
from contextidx.backends.chroma import ChromaBackend

# Ephemeral (in-process) — great for testing
backend = ChromaBackend(collection_name="contextidx")

# Client/server mode
backend = ChromaBackend(
    collection_name="contextidx",
    host="localhost",
    port=8000,
)
```

**Start the Chroma server:**

```bash
chroma run --path ./chroma-data
```

When `host` is `None`, Chroma runs in ephemeral in-process mode — no server needed.

## Pinecone

Fully managed vector database. No infrastructure to maintain.

```bash
pip install contextidx[pinecone]
```

```python
from contextidx.backends.pinecone import PineconeBackend

backend = PineconeBackend(
    api_key="your-api-key",
    index_name="contextidx",
    namespace="production",    # optional namespace isolation
    dimensions=1536,
    cloud="aws",               # or "gcp", "azure"
    region="us-east-1",
    create_if_missing=True,    # auto-create serverless index
)
```

Pinecone uses serverless indexes by default. The index is auto-created on `initialize()` if `create_if_missing=True`.

## Custom Backends

Implement the `VectorBackend` ABC:

```python
from contextidx.backends.base import VectorBackend, SearchResult

class MyBackend(VectorBackend):
    async def store(self, id: str, embedding: list[float], metadata: dict | None = None) -> str:
        ...

    async def search(self, query_embedding: list[float], top_k: int, filters: dict | None = None) -> list[SearchResult]:
        ...

    async def delete(self, id: str) -> None:
        ...

    async def update_metadata(self, id: str, metadata: dict) -> None:
        ...
```

Optional overrides:
- `supports_metadata_store` — return `True` if your backend can store contextidx metadata natively
- `supports_hybrid_search` — return `True` and override `hybrid_search()` for BM25+vector
- `initialize()` / `close()` — lifecycle hooks
