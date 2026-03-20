# Quickstart

Get up and running with contextidx in under 5 minutes.

## Installation

```bash
pip install contextidx
```

With a vector backend (pick one):

```bash
pip install contextidx[pgvector]   # PostgreSQL + pgvector
pip install contextidx[qdrant]     # Qdrant
pip install contextidx[chroma]     # ChromaDB
pip install contextidx[pinecone]   # Pinecone
pip install contextidx[weaviate]   # Weaviate (hybrid search)
```

For production stores:

```bash
pip install contextidx[postgres]   # PostgresStore (asyncpg)
pip install contextidx[redis]      # Redis pending buffer
```

Or install everything:

```bash
pip install contextidx[all]
```

## Basic Usage

```python
import asyncio
from contextidx import ContextIdx
from contextidx.backends.pgvector import PGVectorBackend

async def main():
    # 1. Create an instance with your vector backend
    idx = ContextIdx(
        backend=PGVectorBackend(conn_string="postgresql://localhost/mydb"),
        decay_model="exponential",
        half_life_days=30,
        openai_api_key="sk-...",
    )
    await idx.ainitialize()

    # 2. Store context
    unit_id = await idx.astore(
        content="User prefers async communication over calls",
        scope={"user_id": "u123", "session_id": "s456"},
        confidence=0.9,
        source="conversation_turn_42",
    )

    # 3. Retrieve — temporally scored, stale context filtered
    results = await idx.aretrieve(
        query="how does user prefer to communicate?",
        scope={"user_id": "u123"},
        top_k=5,
    )

    for unit in results:
        print(f"[{unit.confidence:.2f}] {unit.content}")

    # 4. Reinforce context that was useful
    await idx.areinforce(unit_id)

    # 5. View version history
    history = await idx.alineage(unit_id)

    # 6. Clean up
    await idx.aclose()

asyncio.run(main())
```

## Sync API

Every async method has a sync wrapper:

```python
idx = ContextIdx(backend=backend)
# These call asyncio.run() under the hood
unit_id = idx.store(content="...", scope={"user_id": "u1"})
results = idx.retrieve(query="...", scope={"user_id": "u1"})
idx.reinforce(unit_id)
```

## Key Concepts

| Concept | Description |
|---------|-------------|
| **ContextUnit** | The atomic unit of context — text + embedding + temporal metadata |
| **Scope** | Dict of key-value pairs that partition context (e.g. `{"user_id": "u1"}`) |
| **Decay** | Time-based relevance scoring — older context is less relevant |
| **Reinforcement** | Using context resets part of its decay clock |
| **Conflict Resolution** | Detects and resolves contradictory context |
| **Lineage** | Full version history of a context unit |

## Next Steps

- [Backend Setup Guide](backends.md) — detailed setup for each backend
- [Configuration Reference](configuration.md) — all configuration options
- [API Reference](api_reference.md) — full API documentation
- [Scaling Guide](scaling.md) — horizontal scaling patterns
