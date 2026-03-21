# contextidx

> A temporal context layer for your existing vector DB — giving AI agents accurate context, not just similar context.

> **Note:** contextidx is an experimental project under active development. APIs may change. Contributions and feedback are welcome.

### How it works

contextidx sits between your app and your vector DB. Every piece of stored context is wrapped in a **ContextUnit** that tracks when it was created, how it decays over time, and how it relates to other context via a **temporal graph**. On retrieval, contextidx fuses vector similarity with temporal decay, BM25 keyword matching, and graph-based expansion to return context that is both relevant *and* current — not just similar.

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
| PGVector | `pgvector` | Yes | Yes |
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

## Benchmark Results

Evaluated on the [LoCoMo](https://github.com/snap-research/locomo) long-conversation memory benchmark via [memorybench](https://github.com/supermemoryai/memorybench) (50 questions, GPT-4o judge).

### Comparison with other memory providers

| Provider | LoCoMo Accuracy | Type |
|----------|:-:|------|
| Zep | ~85% | Cloud |
| Letta/MemGPT | ~83% | Cloud |
| **contextidx** | **74%** | **Open-source** |
| Supermemory | ~70% | Cloud |
| Mem0 | ~58-66% | Cloud |

### contextidx breakdown

| Question Type | Accuracy | What helps |
|---------------|:-:|------------|
| Multi-hop | 83.3% | Temporal graph expansion across related chunks |
| Temporal | 85.7% | Chronological context ordering + date-aware prompting |
| Single-hop | 57.9% | Vector + BM25 hybrid search |

| Retrieval Metric | Score |
|------------------|:-----:|
| Hit@K | 92% |
| MRR | 0.717 |
| NDCG | 0.738 |

The **multi-hop score (83.3%)** is contextidx's standout — the temporal graph connects related chunks across sessions, enabling multi-hop reasoning that pure RAG systems miss. This is competitive with cloud providers costing significantly more.

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
