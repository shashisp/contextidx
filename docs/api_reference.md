# API Reference

## ContextIdx

The main entry point for the library.

```python
from contextidx import ContextIdx
```

### Constructor

```python
ContextIdx(
    backend: VectorBackend,
    *,
    decay_model: Literal["exponential", "linear", "step"] = "exponential",
    half_life_days: float = 30.0,
    conflict_detection: Literal["rule_based", "semantic", "tiered"] = "rule_based",
    conflict_strategy: Literal["LAST_WRITE_WINS", "HIGHEST_CONFIDENCE", "MERGE", "MANUAL"] = "LAST_WRITE_WINS",
    scoring_weights: dict[str, float] | None = None,
    internal_store: Store | None = None,
    internal_store_path: str | None = None,
    internal_store_type: Literal["sqlite", "postgres", "auto"] = "auto",
    internal_store_dsn: str | None = None,
    openai_api_key: str | None = None,
    embedding_model: str = "text-embedding-3-small",
    state_path_interval: float = 60.0,
    decay_threshold: float = 0.01,
    reconcile_every_n_ticks: int = 10,
    enable_batching: bool = False,
    batch_size: int = 10,
    batch_flush_interval: float = 0.5,
    consolidation_every_n_ticks: int = 5,
    wal_compact_every_n_ticks: int = 50,
    pending_buffer_type: Literal["memory", "redis"] = "memory",
    redis_url: str | None = None,
)
```

See [Configuration Reference](configuration.md) for detailed parameter descriptions.

### Lifecycle

| Method | Description |
|--------|-------------|
| `await ainitialize()` | Initialize stores, replay WAL, start background tasks |
| `await aclose()` | Stop background tasks, flush buffers, close connections |

### Write Path

| Method | Description |
|--------|-------------|
| `await astore(content, scope, ...)` | Store a context unit (async) |
| `store(content, scope, ...)` | Store a context unit (sync wrapper) |
| `await astore_batch(items)` | Store multiple units with batched embedding |

#### `astore` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content` | `str` | required | Text content of the context |
| `scope` | `dict[str, str]` | required | Scope partition keys |
| `confidence` | `float` | `0.8` | Confidence score [0, 1] |
| `source` | `str` | `"unknown"` | Provenance identifier |
| `embedding` | `list[float] \| None` | `None` | Pre-computed embedding (skips API call) |
| `decay_model` | `str \| None` | `None` | Override per-unit decay model |
| `decay_rate` | `float \| None` | `None` | Override per-unit decay rate |
| `expires_at` | `datetime \| None` | `None` | Hard expiration time |

Returns the stored unit ID (`str`).

### Read Path

| Method | Description |
|--------|-------------|
| `await aretrieve(query, scope, ...)` | Retrieve temporally-scored context (async) |
| `retrieve(query, scope, ...)` | Retrieve temporally-scored context (sync) |

#### `aretrieve` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | required | Natural language query |
| `scope` | `dict[str, str]` | required | Scope filter |
| `top_k` | `int` | `5` | Number of results |
| `at` | `datetime \| None` | `None` | Time-travel query point |
| `query_embedding` | `list[float] \| None` | `None` | Pre-computed query embedding |
| `min_score` | `float` | `0.0` | Minimum composite score threshold |

Returns `list[ContextUnit]` sorted by composite score (highest first).

### Other Operations

| Method | Description |
|--------|-------------|
| `await areinforce(unit_id)` | Mark a unit as used, partially resetting decay |
| `await alineage(unit_id)` | Get full version history for a unit |
| `await adiff(scope, since)` | Get units created/modified since a time |
| `await areconcile()` | Detect and repair store drift |

---

## ContextUnit

The atomic unit of temporal context.

```python
from contextidx import ContextUnit
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Unique identifier (auto-generated) |
| `content` | `str` | Text content |
| `embedding` | `list[float] \| None` | Vector embedding |
| `scope` | `dict[str, str]` | Scope partition |
| `confidence` | `float` | Confidence [0, 1] |
| `decay_rate` | `float` | Decay rate per day |
| `decay_model` | `str` | `"exponential"`, `"linear"`, or `"step"` |
| `version` | `int` | Version counter |
| `source` | `str` | Provenance |
| `superseded_by` | `str \| None` | ID of superseding unit |
| `timestamp` | `datetime` | Creation time (UTC) |
| `expires_at` | `datetime \| None` | Hard expiration |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `is_expired` | `bool` | Whether the unit has passed its `expires_at` |
| `is_superseded` | `bool` | Whether another unit supersedes this one |
| `age_days` | `float` | Age in days since creation |

### Class Methods

| Method | Description |
|--------|-------------|
| `decay_rate_from_half_life(days)` | Convert half-life in days to a decay rate |

---

## VectorBackend

Abstract base class for vector storage backends.

```python
from contextidx.backends.base import VectorBackend
```

### Abstract Methods

| Method | Description |
|--------|-------------|
| `await store(id, embedding, metadata)` | Persist an embedding |
| `await search(query_embedding, top_k, filters)` | Similarity search |
| `await delete(id)` | Remove a vector |
| `await update_metadata(id, metadata)` | Patch metadata |

### Properties

| Property | Default | Description |
|----------|---------|-------------|
| `supports_metadata_store` | `False` | Can store contextidx metadata natively |
| `supports_hybrid_search` | `False` | Supports BM25 + vector hybrid search |

### Optional Hooks

| Method | Description |
|--------|-------------|
| `await initialize()` | Create tables, connections, etc. |
| `await close()` | Clean up resources |
| `await hybrid_search(...)` | BM25 + vector search (defaults to vector-only) |

---

## SearchResult

Return type from backend search operations.

```python
from contextidx.backends.base import SearchResult
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Vector ID |
| `score` | `float` | Similarity score [0, 1] |
| `metadata` | `dict` | Associated metadata |
