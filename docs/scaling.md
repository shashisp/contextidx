# Scaling Guide

This guide covers horizontal scaling patterns for production contextidx deployments.

## Architecture Overview

```
┌────────────────┐     ┌────────────────┐
│  Instance A    │     │  Instance B    │
│  ContextIdx    │     │  ContextIdx    │
└───────┬────────┘     └───────┬────────┘
        │                      │
        ▼                      ▼
┌────────────────────────────────────────┐
│           Redis Pending Buffer         │
│     (shared read-after-write state)    │
└───────────────────┬────────────────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│PostgreSQL│ │  Vector  │ │PostgreSQL│
│  Store   │ │ Backend  │ │ (pgvec)  │
└──────────┘ └──────────┘ └──────────┘
```

## Multi-Instance Setup

### 1. Switch to PostgresStore

Replace SQLite with PostgreSQL for the internal metadata store:

```python
idx = ContextIdx(
    backend=backend,
    internal_store_dsn="postgresql://user:pass@db-host:5432/contextidx",
)
```

PostgresStore uses `asyncpg` with connection pooling (2–10 connections by default). Multiple instances can share the same Postgres database safely.

### 2. Enable Redis Pending Buffer

The in-memory pending buffer only provides read-after-write consistency within a single process. For multi-instance deployments, switch to Redis:

```python
idx = ContextIdx(
    backend=backend,
    internal_store_dsn="postgresql://...",
    pending_buffer_type="redis",
    redis_url="redis://redis-host:6379/0",
)
```

The Redis buffer stores serialized ContextUnits in sorted sets keyed by scope hash. TTL is enforced via Redis score-based expiry.

### 3. Connection Pool Tuning

For PostgresStore:

```python
from contextidx.store.postgres_store import PostgresStore

store = PostgresStore(
    dsn="postgresql://...",
    min_pool_size=5,
    max_pool_size=20,
)
idx = ContextIdx(backend=backend, internal_store=store)
```

## Session Affinity

For best read-after-write consistency (even without Redis), route requests from the same user/session to the same instance using load-balancer session affinity (sticky sessions).

With Redis pending buffer, session affinity is optional but still reduces latency — local buffer hits are faster than Redis round-trips.

## State Path Coordination

The background state path (decay recalculation, consolidation, WAL compaction) runs in every instance. To avoid duplicate work:

1. **Stagger intervals** — use different `state_path_interval` values per instance
2. **Leader election** — use a distributed lock (e.g., Redis `SET NX EX`) to ensure only one instance runs the state path
3. **Increase tick intervals** — for reconciliation, consolidation, and WAL compaction, use larger `*_every_n_ticks` values to reduce overlap

```python
idx = ContextIdx(
    backend=backend,
    state_path_interval=120.0,              # 2 min instead of 1 min
    reconcile_every_n_ticks=20,             # reconcile every 40 min
    consolidation_every_n_ticks=10,         # consolidate every 20 min
    wal_compact_every_n_ticks=100,          # compact WAL every ~3.3 hours
)
```

## Write Batching at Scale

Enable write batching to amortize embedding API costs:

```python
idx = ContextIdx(
    backend=backend,
    enable_batching=True,
    batch_size=50,               # flush every 50 units
    batch_flush_interval=1.0,    # or every 1 second
)
```

For high-throughput scenarios, use `astore_batch()` directly to embed and store multiple units in one call:

```python
ids = await idx.astore_batch([
    {"content": "fact 1", "scope": {"user_id": "u1"}},
    {"content": "fact 2", "scope": {"user_id": "u1"}},
    {"content": "fact 3", "scope": {"user_id": "u1"}},
])
```

## Monitoring

Key metrics to track:

| Metric | Source | Threshold |
|--------|--------|-----------|
| WAL pending entries | `store.get_pending_wal()` | Alert if > 1000 |
| Reconciliation errors | `areconcile()` return value | Alert if errors > 0 |
| Decay tick duration | Logging | Alert if > 30s |
| Pending buffer size | Redis `DBSIZE` | Alert if growing steadily |
| Connection pool usage | asyncpg pool stats | Alert if near max |
