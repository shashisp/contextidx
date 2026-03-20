# contextidx

> A context index for your existing vector DB — giving AI agents accurate context, not just similar context.

---

## What is contextidx?

`contextidx` is an open-source temporal context layer that plugs into your existing vector database (pgvector, Weaviate, Qdrant, Chroma, Pinecone) and adds temporal awareness, decay, scoping, conflict resolution, and hybrid retrieval — without requiring you to migrate or replace anything.

**One-liner:** Like adding a database index to make queries faster and more accurate, you add `contextidx` to make context retrieval temporally accurate.

---

## The Problem: Context Drift

Vector databases have no model of time. A user preference stored 18 months ago receives equal retrieval weight to one stored yesterday. A fact explicitly superseded by newer information continues to surface. Contradictory context coexists without resolution.

This is **context drift** — a gradual divergence between the state of the world as represented in the vector store and the actual current state of the user or domain.

```
Without contextidx:
  Agent asks vector DB → returns stale/conflicting embeddings
  → outdated recommendations
  → contradictory behavior across sessions
  → no awareness of what's still true

With contextidx:
  Agent asks contextidx → returns current, scoped, conflict-free context
  → accurate retrieval
  → consistent agent behavior
  → context that reflects truth, not just similarity
```

Context drift has three distinct failure modes:

| Failure Mode | Description | Example |
|---|---|---|
| **Stale retrieval** | Outdated context surfaces with equal weight to current context | User changed fitness goal from weight loss to muscle building — both goals returned simultaneously |
| **Supersession failure** | Explicitly updated facts coexist with their predecessors | "Junior engineer" (Jan) and "Senior engineer" (Dec) both retrieved at equal weight |
| **Contradiction accumulation** | Conflicting context accumulates without resolution | "Prefers formal tone" and "prefers casual tone" coexist — inconsistent agent behavior |

Vector databases answer: *"What is similar?"*

AI agents need: *"What is relevant **right now**, given what happened over time?"*

---

## How It Works

`contextidx` sits between your AI application and your existing vector backend as a middleware layer. You keep your vector DB. You keep your embeddings pipeline. `contextidx` adds the intelligence layer.

```
┌─────────────────────────────────────────────┐
│           Your AI Agent / App               │
└──────────────────────┬──────────────────────┘
                       │
┌──────────────────────▼──────────────────────┐
│                 contextidx                  │
│                                             │
│  ┌────────────┐  ┌────────────┐  ┌───────┐  │
│  │ Write Path │  │ Read Path  │  │ State │  │
│  │            │  │            │  │ Path  │  │
│  │ 1. Extract │  │ 1. Embed   │  │ Decay │  │
│  │ 2. Enrich  │  │ 2. Fetch   │  │ Tick  │  │
│  │ 3. Conflict│  │ 3. Filter  │  │ Cons. │  │
│  │ 4. Version │  │ 4. Score   │  │ Reinf.│  │
│  │ 5. Store   │  │ 5. Return  │  │ Expiry│  │
│  └────────────┘  └────────────┘  └───────┘  │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │    Internal Store (SQLite / PG)     │    │
│  │  ContextUnits │ Graph │ DecayState  │    │
│  └─────────────────────────────────────┘    │
└──────────────────────┬──────────────────────┘
                       │  VectorBackend interface
                       ▼
          Your Existing Vector Store
          (pgvector / Weaviate / Qdrant / Chroma)
```

---

## Five Core Primitives

### 1. ContextUnit

The atomic unit of temporal context storage. Extends a standard vector embedding with temporal and provenance metadata.

```python
ContextUnit {
  id:             str          # unique identifier
  content:        str          # raw text content
  embedding:      List[float]  # stored in vector backend
  timestamp:      datetime     # creation time
  confidence:     float        # extraction confidence in [0, 1]
  decay_rate:     float        # λ parameter for decay function
  decay_model:    str          # exponential | linear | step
  scope:          dict         # {user_id, session_id, agent_id}
  source:         str          # provenance identifier
  superseded_by:  str | None   # id of superseding unit, if any
  version:        int          # version number within lineage
  expires_at:     datetime | None
}
```

The `embedding` is stored in the external vector backend. All other fields are maintained in `contextidx`'s internal SQLite store.

---

### 2. TemporalGraph

Maintains directed relationships between ContextUnits, enabling lineage tracking and supersession chains. Four relationship types:

| Relationship | Meaning |
|---|---|
| `SUPERSEDES` | A newer fact replaces an older one. Superseded units are excluded from active retrieval. |
| `RELATES_TO` | Bidirectional semantic relationship between context units that inform each other. |
| `VERSION_OF` | Incremental update to a fact over time — preserves full history. |
| `CAUSED_BY` | Causal provenance — which event or artifact triggered a context update. |

---

### 3. DecayEngine

Computes temporal relevance score `d(eᵢ, t)` for a ContextUnit at retrieval time `t`. Three models:

**Exponential decay** (default) — fast initial decay, long tail. Best for preferences and soft facts.
```
d(eᵢ, t) = cᵢ · exp(-λ(t - tᵢ))
```

**Linear decay** — steady reduction to zero by expiry. Best for skill levels and performance scores.
```
d(eᵢ, t) = cᵢ · max(0, 1 - (t - tᵢ) / h)
```

**Step decay** — full relevance until hard expiry, then zero. Best for session context and event-bound facts (e.g. injury status).
```
d(eᵢ, t) = cᵢ  if t < t_exp
           0    otherwise
```

**Reinforcement** — when a ContextUnit is retrieved and used, its decay clock resets partially. Frequently accessed context stays fresh longer.

---

### 4. ScoringEngine

Implements composite temporal retrieval by fusing five signals:

```
f(eᵢ, q, t) = w_s · semantic_similarity
             + w_r · recency_score
             + w_c · confidence
             + w_d · decay_score
             + w_u · reinforcement_weight
```

**Default weights:**

| Signal | Default Weight | Rationale |
|---|---|---|
| Semantic similarity | 0.35 | Core relevance signal from vector search |
| Recency | 0.25 | Recent context generally more reliable |
| Confidence | 0.20 | Extraction quality signal |
| Decay | 0.15 | Temporal relevance degradation |
| Reinforcement | 0.05 | Usage frequency as implicit relevance signal |

Weights are fully configurable. A medical AI system may weight confidence higher; a conversational assistant may weight recency higher.

---

### 5. ConflictResolver

Detects and resolves conflicting context units within the same scope.

**Detection methods:**

| Method | Complexity | Best For |
|---|---|---|
| Rule-based (default) | O(1) | Explicit contradictions, high-throughput writes |
| Semantic | O(n) | Implicit conflicts, higher accuracy |

**Resolution strategies:**

| Strategy | Behavior |
|---|---|
| `LAST_WRITE_WINS` (default) | Most recent context supersedes all prior conflicting units |
| `HIGHEST_CONFIDENCE` | Highest confidence score wins regardless of recency |
| `MERGE` | Conflicting units combined into one with reduced confidence, flagged as uncertain |
| `MANUAL` | Conflicts flagged for human review; no automatic resolution |

---

## Three Processing Paths

### Write Path

When new context arrives, `contextidx` processes it through five sequential stages:

1. **Extract** — parse raw input into a ContextUnit; assign scope, source, initial confidence
2. **Enrich** — generate embedding; assign timestamp, decay model, decay rate based on content type
3. **Conflict Check** — query existing active ContextUnits in scope; run ConflictResolver
4. **Version** — if updating an existing fact, create a `SUPERSEDES` edge in the TemporalGraph; mark prior unit as superseded
5. **Store** — write embedding to vector backend; write metadata, graph edges, and decay state to internal store

### Read Path

On retrieval, `contextidx` augments standard vector similarity search with temporal post-processing:

1. **Embed** — generate query embedding
2. **Fetch** — execute similarity search, retrieve top-k × 3 candidates (over-fetch to account for filtering)
3. **Filter** — remove superseded units, expired units, and out-of-scope units; discard below minimum decay threshold
4. **Score** — apply ScoringEngine composite `f(eᵢ, q, t)` to remaining candidates
5. **Rank & Return** — return top-k by composite score; optionally include lineage metadata

### State Path

Runs periodically or event-triggered to maintain context quality over time:

- **Decay Tick** — recalculate decay scores for all active ContextUnits; flag below-threshold units for archival
- **Consolidation** — merge semantically redundant ContextUnits; summarize long lineage chains, preserving graph history
- **Reinforcement Update** — update reinforcement scores based on recent retrieval usage
- **Expiry** — archive step-decay units past expiry; archived units remain available for time-travel queries but excluded from active retrieval

---

## Scope Model

Every ContextUnit is scoped. Retrieval automatically filters to the requested scope, preventing context leakage between users or sessions.

| Scope | Persists | Example |
|---|---|---|
| `user_id` | Across all sessions for a user | "User prefers async communication" |
| `session_id` | Within a single session only | "User is currently debugging the auth flow" |
| `agent_id` | Specific to an agent instance | "Agent has already suggested solution A" |

Scopes can be combined for fine-grained isolation.

---

## Backend Adapter Interface

All vector backend operations are abstracted behind a four-method interface:

```python
class VectorBackend(ABC):
    def store(id, embedding, metadata) -> str
    def search(query_embedding, top_k, filters) -> List[SearchResult]
    def delete(id) -> None
    def update_metadata(id, metadata) -> None
```

This minimal interface is satisfied by all major vector stores. Custom backends require implementing just four methods.

**Supported backends:**

| Backend | Status | Notes |
|---|---|---|
| pgvector | ✅ v0.1 | Default, recommended |
| Qdrant | 🔜 v0.2 | In progress |
| Weaviate | 🔜 v0.2 | Planned |
| Chroma | 🔜 v0.3 | Planned |
| Pinecone | 🔜 v0.3 | Planned |

---

## Quickstart

```bash
pip install contextidx
```

```python
from contextidx import ContextIdx
from contextidx.backends import PGVector

# Initialize with your existing vector DB
idx = ContextIdx(
    backend=PGVector(conn_string="postgresql://..."),
    decay_model="exponential",
    half_life_days=30
)

# Store context
idx.store(
    content="User prefers async communication over calls",
    scope={"user_id": "u123"},
    confidence=0.9,
    source="conversation_turn_42"
)

# Retrieve — temporally scored, stale context filtered
context = idx.retrieve(
    query="how does user prefer to communicate?",
    scope={"user_id": "u123"},
    top_k=5
)

# Reinforce — mark context as used, reset decay
idx.reinforce(unit_id=context[0].id)

# Time travel — what did the agent know on a specific date?
past = idx.retrieve(
    query="user communication style",
    scope={"user_id": "u123"},
    at="2025-01-01"
)

# Diff — what changed in the last 30 days?
delta = idx.diff(
    scope={"user_id": "u123"},
    since="30d"
)

# Lineage — full history of a context unit
history = idx.lineage(unit_id="ctx_abc123")
```

---

## Time Travel Queries

`contextidx` supports point-in-time retrieval by restoring the active context set to its state at any past timestamp.

Implementation:
- Include superseded units created before `t_query` and superseded after `t_query`
- Exclude units created after `t_query`
- Compute decay scores relative to `t_query`, not current time

**Use cases:**
- **Progress reports** — "What did the agent know about this user in January?"
- **Debugging** — "Why did the agent make this recommendation last week?"
- **Compliance auditing** — Full point-in-time reconstruction of agent context state

---

## Internal Data Model

`contextidx` maintains a lightweight metadata store (SQLite for v0, Postgres for v1+) alongside your vector DB. Separate from your embeddings.

```sql
-- Core context units
CREATE TABLE context_units (
    id           TEXT PRIMARY KEY,
    content      TEXT,
    scope        JSON,
    confidence   REAL,
    decay_rate   REAL,
    decay_model  TEXT,
    version      INTEGER,
    source       TEXT,
    superseded   BOOLEAN DEFAULT FALSE,
    created_at   TIMESTAMP,
    expires_at   TIMESTAMP,
    archived_at  TIMESTAMP
);

-- Temporal graph (supersession + relationships)
CREATE TABLE context_graph (
    from_id      TEXT REFERENCES context_units(id),
    to_id        TEXT REFERENCES context_units(id),
    relationship TEXT,   -- supersedes | relates_to | version_of | caused_by
    created_at   TIMESTAMP
);

-- Decay state (updated on access and on schedule)
CREATE TABLE decay_state (
    unit_id              TEXT REFERENCES context_units(id),
    current_score        REAL,
    last_updated         TIMESTAMP,
    reinforcement_count  INTEGER DEFAULT 0
);
```

---

## Project Structure

```
contextidx/
├── core/
│   ├── context_unit.py       # ContextUnit dataclass
│   ├── temporal_graph.py     # Graph operations
│   ├── decay_engine.py       # Decay models
│   ├── scoring_engine.py     # Composite scoring
│   └── conflict_resolver.py  # Conflict detection + resolution
├── backends/
│   ├── base.py               # VectorBackend ABC
│   ├── pgvector.py           # pgvector adapter
│   ├── weaviate.py           # Weaviate adapter
│   └── qdrant.py             # Qdrant adapter
├── store/
│   ├── sqlite_store.py       # Internal SQLite store
│   └── schema.py             # Schema definitions
└── contextidx.py             # Public API
```

---

## Comparison

| | Raw Vector DB | contextidx + Vector DB | HydraDB / Zep |
|---|---|---|---|
| Temporal awareness | ❌ | ✅ | ✅ |
| Context decay | ❌ | ✅ | ✅ |
| Supersession / versioning | ❌ | ✅ | ✅ |
| Conflict resolution | ❌ | ✅ | ✅ |
| User/session/agent scoping | Manual | ✅ Built-in | ✅ |
| Time travel queries | ❌ | ✅ | Partial |
| Works with existing stack | ✅ | ✅ | ❌ (requires migration) |
| Open source | — | ✅ | ❌ |
| Self-hostable | ✅ | ✅ | ❌ |

**The key difference:** HydraDB and Zep require you to replace or migrate your vector infrastructure. `contextidx` adds the same capabilities to the stack you already run.

---

## Use Cases

- **Conversational AI agents** — maintain accurate user context across sessions without stale or contradictory memory
- **Personalization pipelines** — surface what's true about a user right now, not what was true 6 months ago
- **Multi-agent systems** — scope context cleanly per agent, per session, per user; prevent context bleed
- **Long-horizon task agents** — track what the agent has already tried, what worked, what's been ruled out
- **Customer support AI** — give the agent accurate context about the customer's current situation, not historical noise
- **Learning / coaching platforms** — decay skill assessments over time; reinforce on demonstrated usage

---

## Design Principles

1. **Non-invasive** — plugs into your existing stack; nothing to migrate
2. **Transparent** — every retrieval decision is explainable and traceable via lineage
3. **Composable** — use the full API or just the primitives you need
4. **Correctness over recall** — better to return fewer, accurate context units than many stale ones
5. **Time is a first-class primitive** — not an afterthought filter
6. **Configurable decay** — different domains have different context lifecycles; no single model fits all

---

## Reliability Design

### Write Latency: Async Writes + Tiered Conflict Detection

The core problem: semantic conflict detection requires embedding incoming content, searching existing units, and running the classifier — all potentially on the write path. Three mitigations used together:

**1. Non-blocking writes by default**

The agent doesn't wait for conflict resolution. Fire-and-forget with optional confirmation:

```python
# Default: non-blocking, conflict resolution runs in background
await idx.astore(content, scope, confidence)

# When you need confirmation before proceeding
await idx.astore(content, scope, confidence, wait_for_conflict=True)
```

**2. Tiered conflict detection**

```python
# High-throughput: rule-based only (O(1), <1ms)
idx = ContextIdx(conflict_detection="rule_based")

# Quality-critical: semantic (O(n), ~20-50ms)
idx = ContextIdx(conflict_detection="semantic")

# Default: rule-based inline, semantic async in background
idx = ContextIdx(conflict_detection="tiered")
```

Tiered mode resolves obvious conflicts instantly and queues ambiguous ones for background semantic resolution. The State Path picks them up on the next tick.

**3. Write batching**

Buffers writes into micro-batches — amortizes the embedding + search cost across multiple writes. One OpenAI batch call instead of N sequential calls.

---

### Read-After-Write Consistency: Pending Buffer

When writes are non-blocking, an agent that writes a fact and immediately retrieves it might not see it — conflict resolution hasn't finished. The pending buffer solves this.

Every write is immediately placed in a scoped in-memory buffer. The read path checks the buffer first, before hitting the vector backend, and merges pending units into results:

```python
async def astore(self, content, scope, confidence):
    unit = ContextUnit(content=content, scope=scope,
                       confidence=confidence, status="pending")

    # immediately visible to reads in this session
    scope_key = _hash_scope(scope)
    self._pending.setdefault(scope_key, []).append(unit)

    # conflict resolution + persistence runs async
    asyncio.create_task(self._write_and_resolve(unit))

async def aretrieve(self, query, scope, top_k):
    scope_key = _hash_scope(scope)
    pending = self._pending.get(scope_key, [])

    # fetch from vector backend
    vector_results = await self._vector_retrieve(query, scope, top_k)

    # pending units win on conflict, merged into final results
    return _merge_with_pending(vector_results, pending, top_k)
```

**Buffer constraints:**
- Scoped per `session_id` — no cross-user pending bleed in multi-tenant deployments
- Max 50 units per scope, 30s TTL — units exceeding TTL written as-is with `conflict_unresolved` flag for State Path cleanup

---

### Metadata Sync: WAL + Checkpoint Reconciliation

Two stores (vector DB + SQLite/Postgres) can diverge on write failures, crashes, or backup/restore mismatches. Two mechanisms address this at different granularities.

**Write-Ahead Log (WAL)**

Every write appends to a WAL before touching either store. Store operations confirm back with `applied_at`. Crash recovery replays only `status = 'pending'` WAL entries — O(crash window), not O(dataset):

```sql
CREATE TABLE wal (
    seq          BIGINT PRIMARY KEY AUTOINCREMENT,
    unit_id      TEXT,
    operation    TEXT,        -- store | update | delete | supersede
    store_target TEXT,        -- internal | vector | both
    payload      JSON,
    written_at   TIMESTAMP,
    applied_at   TIMESTAMP,   -- null until confirmed applied
    status       TEXT         -- pending | applied | failed
);
```

```python
async def startup_reconcile():
    # replay only unconfirmed entries — O(crash window), not O(dataset)
    pending = await wal.find(status="pending")
    for entry in pending:
        await _replay_wal_entry(entry)
```

**Last Sync Checkpoint**

Tracks the last successfully reconciled timestamp. Periodic health checks scan only units modified since the checkpoint:

```sql
CREATE TABLE sync_checkpoints (
    store_name     TEXT PRIMARY KEY,
    last_synced_at TIMESTAMP,
    units_synced   INTEGER
);
```

```python
async def periodic_reconcile():
    checkpoint = await internal_store.get_checkpoint("vector_backend")
    since = checkpoint.last_synced_at if checkpoint else epoch

    orphaned = await internal_store.find_active_without_vector(since=since)
    for unit in orphaned:
        await vector_backend.upsert(unit)

    await internal_store.update_checkpoint("vector_backend", now())
```

WAL handles crash recovery (exact, O(crash window)). Checkpoint handles broader sync health (periodic, O(recent changes)). Together they make reconciliation near-instant at any dataset size.

**Backup order:** always back up internal store first, then vector DB. On restore, run WAL replay then checkpoint reconciliation.

---

### Single-Store Portability: Capability Flag

Different backends have different metadata capabilities. The backend adapter declares what it supports — `contextidx` auto-routes, the developer never thinks about it:

```python
class VectorBackend(ABC):
    def store(id, embedding, metadata) -> str
    def search(query_embedding, top_k, filters) -> List[SearchResult]
    def delete(id) -> None
    def update_metadata(id, metadata) -> None

    @property
    def supports_metadata_store(self) -> bool:
        return False  # default: needs external internal store
```

```python
class WeaviateBackend(VectorBackend):
    @property
    def supports_metadata_store(self) -> bool:
        return True   # metadata on Weaviate object properties natively

class PGVectorBackend(VectorBackend):
    @property
    def supports_metadata_store(self) -> bool:
        return True   # metadata as Postgres columns, same row

class PineconeBackend(VectorBackend):
    @property
    def supports_metadata_store(self) -> bool:
        return False  # vector-only, needs external SQLite/Postgres
```

```python
class ContextIdx:
    def __init__(self, backend, internal_store=None):
        if backend.supports_metadata_store and internal_store is None:
            self._meta = BackendMetadataStore(backend)
        elif internal_store is not None:
            self._meta = internal_store  # explicit override
        else:
            # auto-provision SQLite — zero config for developer
            self._meta = SQLiteStore(path=".contextidx/meta.db")
            logger.info("Auto-provisioned SQLite store at .contextidx/meta.db")
```

**Note:** even when the backend acts as its own metadata store, the temporal graph edges (`context_graph`) still need a home. Weaviate handles this via native cross-references. For Pinecone and other vector-only stores, SQLite is always auto-provisioned for the graph — transparently.

---

### Full Write + Read Path

```
Incoming write
      │
      ├──→ Pending buffer (immediate, sync)     ← read-after-write
      │
      └──→ WAL append (immediate, sync)         ← crash recovery
              │
              └──→ Conflict resolution (async, tiered)
                      │
                      └──→ Flush to stores (async)
                              ├──→ Vector backend
                              └──→ Metadata store (auto-routed)

Read path
      │
      ├──→ Check pending buffer first           ← read-after-write
      └──→ Vector backend + metadata store
              └──→ Merge + score + return

Startup
      ├──→ WAL replay (pending entries only)    ← O(crash window)
      └──→ Checkpoint scan (since last sync)    ← O(recent changes)
```

---

## Limitations

- **Pending buffer is in-process** — in horizontally scaled deployments with multiple `contextidx` instances, the pending buffer is not shared. Cross-instance read-after-write consistency requires a shared cache (e.g. Redis) or session-affinity routing.
- **WAL compaction** — the WAL is append-only and needs a periodic compaction job to archive `status = 'applied'` entries older than your retention window.
- **Lineage storage overhead** — long-lived entities with frequently updated context accumulate large lineage chains. The State Path consolidation mechanism mitigates but doesn't eliminate this.
- **No extraction pipeline** — `contextidx` does not determine what is worth storing from raw input. Context extraction is treated as an upstream concern.

---

## Tech Stack

### Philosophy: Python core, Rust-accelerated hot paths

`contextidx` is a Python library — because your users are Python AI/ML engineers and `pip install contextidx` needs to just work. Rust is used selectively for the three hot paths that bottleneck at scale, following the same pattern as `pydantic` v2, `tokenizers`, and `polars`: Python API, Rust internals.

---

### Core Dependencies

```
Python 3.11+
├── pydantic v2        # ContextUnit schema + validation (Rust-backed, fast)
├── sqlalchemy 2.0     # Internal store ORM (async-native)
├── aiosqlite          # SQLite async driver for v0
├── asyncpg            # Postgres async driver for v1+
├── numpy              # Decay math and vector ops
└── httpx              # Async HTTP for cloud vector backends
```

**Why pydantic v2 from day one:** Rust-backed, free serialization to SQLite JSON columns, schema validation on every write, and users expect pydantic models from modern Python libraries. Retrofitting later is painful.

---

### Vector Backend Clients

```
pgvector    → psycopg3 + pgvector-python
Weaviate    → weaviate-client v4 (async-native)
Qdrant      → qdrant-client (async)
Chroma      → chromadb-client
Pinecone    → pinecone-client v3
```

---

### Internal Store

```
v0:   SQLite via aiosqlite    # zero-config, embedded, ships with Python
v1+:  Postgres via asyncpg    # same schema, production scale
```

Same schema across both — migration is a connection string swap, nothing else.

---

### Rust-Accelerated Hot Paths (PyO3 + maturin)

Three operations that bottleneck at 100k+ context units:

```
contextidx/
└── _core/                     # Rust extension (optional, falls back to Python)
    ├── decay_engine.rs        # Batch decay score computation
    ├── scoring_engine.rs      # Composite score fusion (tight inner loop)
    └── conflict_detector.rs   # Rule-based conflict detection (O(1) per write)
```

Exposed as `from contextidx._core import _decay` — private accelerated module with pure Python fallback for environments without the compiled extension. Users never interact with it directly.

---

### Recommended Stack: contextidx + Weaviate

For production deployments, Weaviate as the vector backend gives you capabilities that pgvector alone cannot provide:

```
Your AI Agent
      │
      ▼
contextidx                    ← temporal intelligence
  decay · supersession
  conflict resolution
  scoping · time travel
      │
      ▼
Weaviate                      ← search infrastructure
  BM25 + vector hybrid search
  server-side vectorization (text2vec-openai)
  storage-level tenant isolation
  managed HNSW + pre-filter ANN
```

| Capability | pgvector | Weaviate |
|---|---|---|
| Hybrid search (BM25 + vector) | Manual (`tsvector` + RRF) | Built-in |
| Multi-tenancy | `WHERE org_id = …` discipline | Storage-level isolation |
| Embedding plumbing | App-side OpenAI calls | Server-side `text2vec-openai` |
| ANN recall with filters | Post-filter (recall drops) | Pre-filter (recall preserved) |
| Ops burden | Self-managed index tuning | Managed HNSW |

`contextidx` solves temporal correctness. Weaviate solves search infrastructure quality. Neither overlaps with the other.

---

### API Design: Sync + Async

Both sync and async APIs are exposed — many agent frameworks are still sync:

```python
# Sync
context = idx.retrieve(query, scope, top_k)
idx.store(content, scope, confidence)

# Async (preferred for production)
context = await idx.aretrieve(query, scope, top_k)
await idx.astore(content, scope, confidence)
```

---

### Performance Architecture

Read path is on the hot path of every agent inference — must be fast. Write and state paths can be async/background:

```
Read path   → synchronous or async, target <20ms overhead over raw vector search
Write path  → fire-and-forget, non-blocking, queue if needed
State path  → background asyncio task or external cron, never inline with reads
```

The State Path (decay tick, consolidation, expiry) runs as a background `asyncio` task — it should never block agent inference.

---

### Benchmark Stack

```
pytest-benchmark    # latency benchmarks
pytest-asyncio      # async test harness
duckdb              # fast local analytics over benchmark result sets
```

---

### What to Avoid

| Option | Why Not |
|---|---|
| Full Rust library | Packaging friction kills OSS adoption; PyO3 gives 90% of the perf with Python DX |
| FastAPI service | contextidx is a library — adding a network hop to every retrieval defeats the purpose |
| Redis for internal store | Overkill for v0; adds ops dependency; SQLite is underrated for embedded metadata |
| LangChain integration first | LangChain abstractions constrain API design; build clean first, add integrations later |
| Async-only API | Many agent frameworks are still sync; offer both |

---

### Phased Rollout

**v0 — correctness**
```
Pure Python · SQLite internal store · pgvector backend
Sync + async API · WAL + pending buffer from day one
Target: correctness benchmarks pass, no data loss on crash
```

**v0.2 — reliability + Weaviate**
```
Weaviate backend adapter · hybrid search wired through ScoringEngine
Scope → Weaviate tenant mapping · checkpoint reconciliation
Capability flag + auto-routing for single-store mode
Target: hybrid search benchmarks · zero-config DX for all backends
```

**v0.5 — performance**
```
PyO3 Rust for decay + scoring hot paths
Tiered conflict detection · write batching
Background State Path · WAL compaction job
Target: <20ms read overhead at 100k context units
```

**v1.0 — production ready**
```
Postgres internal store · full backend suite (Qdrant, Chroma, Pinecone)
LangChain + LlamaIndex integrations · benchmark suite published
Horizontal scaling guide (Redis pending buffer for multi-instance)
```

---

## Project Status

`contextidx` is in active development. Current focus: pgvector adapter and core temporal graph.

- GitHub: `github.com/shashi/contextidx`
- PyPI: `pip install contextidx` *(coming soon)*
- Docs: `contextidx.dev` *(coming soon)*

Contributions welcome. If you're building AI agents on top of pgvector or Qdrant and want to test the alpha, open an issue.

---

## License

MIT

---

*Technical details based on the TempVec research paper (draft v0.1). Built by [Shashi](https://github.com/shashi).*
