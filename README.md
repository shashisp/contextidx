# contextidx

> A temporal context layer for your existing vector DB — giving AI agents accurate context, not just similar context.

## Why contextidx?

Vector databases return **similar** results. But for AI agents, similarity isn't enough — context has a **shelf life**. User preferences change, facts get superseded, and stale information causes hallucinations.

contextidx wraps your existing vector DB and adds:

- **Temporal decay** — older context gradually loses relevance
- **Conflict resolution** — contradictory facts are detected and resolved
- **Scoped retrieval** — context is partitioned per user/session/agent
- **Reinforcement** — context that gets used stays fresh longer
- **Version lineage** — full history of how context evolved

## Backend Support

| Backend | Install Extra | Hybrid Search | Native Metadata |
|---------|--------------|:---:|:---:|
| PGVector | `pgvector` | — | Yes |
| Weaviate | `weaviate` | Yes | — |
| Qdrant | `qdrant` | — | — |
| ChromaDB | `chroma` | — | — |
| Pinecone | `pinecone` | — | — |

## Install

```bash
pip install contextidx                # core
pip install contextidx[pgvector]      # with PGVector backend
pip install contextidx[qdrant]        # with Qdrant backend
pip install contextidx[all]           # everything
```

## Quick Start

```python
import asyncio
from contextidx import ContextIdx
from contextidx.backends.pgvector import PGVectorBackend

async def main():
    idx = ContextIdx(
        backend=PGVectorBackend(conn_string="postgresql://..."),
        decay_model="exponential",
        half_life_days=30,
    )
    await idx.ainitialize()

    # Store context
    await idx.astore(
        content="User prefers async communication over calls",
        scope={"user_id": "u123"},
        confidence=0.9,
        source="conversation_turn_42",
    )

    # Retrieve — temporally scored, stale context filtered
    results = await idx.aretrieve(
        query="how does user prefer to communicate?",
        scope={"user_id": "u123"},
        top_k=5,
    )

    for unit in results:
        print(f"[{unit.confidence:.2f}] {unit.content}")

    await idx.aclose()

asyncio.run(main())
```

## Production Setup

For horizontal scaling, use PostgresStore + Redis pending buffer:

```python
idx = ContextIdx(
    backend=backend,
    internal_store_dsn="postgresql://user:pass@pghost:5432/ctxidx",
    pending_buffer_type="redis",
    redis_url="redis://redis-host:6379/0",
    conflict_detection="tiered",
    enable_batching=True,
)
```

## Documentation

- [Quickstart](docs/quickstart.md)
- [Backend Setup](docs/backends.md)
- [API Reference](docs/api_reference.md)
- [Configuration](docs/configuration.md)
- [Scaling Guide](docs/scaling.md)

## Architecture

```
┌─────────────┐
│  Your App   │
└──────┬──────┘
       │
┌──────▼──────┐     ┌────────────────┐
│  ContextIdx │────▶│ Vector Backend │ (pgvector / Qdrant / Weaviate / ...)
│             │     └────────────────┘
│  • Decay    │     ┌────────────────┐
│  • Scoring  │────▶│ Metadata Store │ (SQLite / PostgreSQL)
│  • Conflict │     └────────────────┘
│  • WAL      │     ┌────────────────┐
│  • Graph    │────▶│ Pending Buffer │ (Memory / Redis)
└─────────────┘     └────────────────┘
```

## License

MIT
