# Configuration Reference

All configuration is passed to the `ContextIdx` constructor. This page documents every parameter.

## Vector Backend

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `VectorBackend` | required | The vector database backend instance |

## Decay Model

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `decay_model` | `"exponential" \| "linear" \| "step"` | `"exponential"` | Global decay function |
| `half_life_days` | `float` | `30.0` | Time in days for relevance to halve |
| `decay_threshold` | `float` | `0.01` | Minimum decay score before filtering out |

### Decay Models Explained

- **Exponential** — `score = confidence * exp(-rate * age_days)`. Smooth, gradual decay. Best for most use cases.
- **Linear** — `score = confidence * max(0, 1 - rate * age_days)`. Reaches zero at `1/rate` days.
- **Step** — Full confidence until `expires_at`, then drops to zero. Use for time-bounded facts.

Individual units can override the global decay model via `astore(decay_model=..., decay_rate=...)`.

## Conflict Resolution

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conflict_detection` | `"rule_based" \| "semantic" \| "tiered"` | `"rule_based"` | Detection strategy |
| `conflict_strategy` | `"LAST_WRITE_WINS" \| "HIGHEST_CONFIDENCE" \| "MERGE" \| "MANUAL"` | `"LAST_WRITE_WINS"` | Resolution strategy |

### Detection Modes

- **rule_based** — Fast regex-based negation and verb-pair detection. Low latency, may miss implicit contradictions.
- **semantic** — Embedding cosine similarity + negation patterns. More accurate but slower.
- **tiered** — Rule-based inline (blocking) + semantic in background queue. Best balance of latency and accuracy.

## Scoring

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scoring_weights` | `dict[str, float] \| None` | `None` | Custom weights for composite scoring |

Default weights (when `None`):

```python
{
    "semantic": 0.30,
    "recency": 0.25,
    "confidence": 0.20,
    "decay": 0.10,
    "reinforcement": 0.05,
    "bm25": 0.10,
}
```

## Internal Metadata Store

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `internal_store` | `Store \| None` | `None` | Explicit store instance (overrides auto-routing) |
| `internal_store_path` | `str \| None` | `None` | Path for SQLite store |
| `internal_store_type` | `"sqlite" \| "postgres" \| "auto"` | `"auto"` | Store type selection |
| `internal_store_dsn` | `str \| None` | `None` | PostgreSQL connection string |

### Auto-Routing Logic

1. If `internal_store` is provided → use it directly
2. If `internal_store_dsn` starts with `"postgresql://"` → use PostgresStore
3. If `internal_store_type == "postgres"` → use PostgresStore
4. If `backend.supports_metadata_store == True` → use BackendMetadataStore
5. Otherwise → use SQLiteStore

## Embedding

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `openai_api_key` | `str \| None` | `None` | OpenAI API key (falls back to `OPENAI_API_KEY` env var) |
| `embedding_model` | `str` | `"text-embedding-3-small"` | OpenAI embedding model name |

## Pending Buffer

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `pending_buffer_type` | `"memory" \| "redis"` | `"memory"` | Buffer backend |
| `redis_url` | `str \| None` | `None` | Redis connection URL (required when `pending_buffer_type="redis"`) |

## Write Batching

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_batching` | `bool` | `False` | Enable automatic write batching |
| `batch_size` | `int` | `10` | Flush after this many buffered writes |
| `batch_flush_interval` | `float` | `0.5` | Max seconds between auto-flushes |

## Background State Path

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_path_interval` | `float` | `60.0` | Seconds between state path ticks |
| `reconcile_every_n_ticks` | `int` | `10` | Run reconciliation every N ticks |
| `consolidation_every_n_ticks` | `int` | `5` | Run consolidation every N ticks |
| `wal_compact_every_n_ticks` | `int` | `50` | Run WAL compaction every N ticks |

## Example: Production Configuration

```python
idx = ContextIdx(
    backend=QdrantBackend(url="http://qdrant:6333"),
    decay_model="exponential",
    half_life_days=14,
    conflict_detection="tiered",
    conflict_strategy="HIGHEST_CONFIDENCE",
    internal_store_dsn="postgresql://user:pass@pghost:5432/ctxidx",
    pending_buffer_type="redis",
    redis_url="redis://redis-host:6379/0",
    enable_batching=True,
    batch_size=50,
    batch_flush_interval=1.0,
    state_path_interval=120.0,
    reconcile_every_n_ticks=20,
    consolidation_every_n_ticks=10,
    wal_compact_every_n_ticks=100,
)
```
