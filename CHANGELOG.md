# Changelog

All notable changes to contextidx are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [1.0.0] — 2026-03-21

### Added
- **Distributed leader election** for multi-instance deployments via Redis (`SET NX EX` lock on state-path loop)
- **Query-type detection** (`contextidx/core/query_type.py`) — classifies queries as factual, temporal, or relational and adjusts composite scoring weights dynamically
- **`ContextIdxConfig`** (`contextidx/config.py`) — all previously hardcoded values (consolidation threshold, max lineage depth, rerank model, RRF constant, WAL retention, etc.) are now configurable
- **`store_batch()`** on `VectorBackend` ABC — all adapters (pgvector, Qdrant, Weaviate, ChromaDB, Pinecone) implement native batch upsert, eliminating N individual round-trips per `astore_batch()` call
- **Dedicated re-ranking module** (`contextidx/core/reranker.py`) — client instantiated once, configurable timeout with graceful fallback to pre-reranking order
- **WAL size/age circuit-breaker** — `max_wal_entries` blocks new writes when the WAL is full; `wal_max_age_hours` drops unrecoverable pending entries with logging
- **WAL: embeddings no longer stored** — only metadata and text are written to the WAL; embeddings are re-computed on replay, reducing WAL storage by ~12 KB per entry
- **`find_expired_units()`** SQL query in store — replaces full table scan in `_expiry_archive()` with an indexed `WHERE expires_at <= :now` query
- **`get_all_graph_edges()`** bulk load — replaces N individual queries on startup with a single SELECT
- **ANN-based pre-filtering** in `find_redundant_pairs()` — eliminates the O(N²) pairwise comparison; candidates are pre-filtered via approximate nearest-neighbor search before exact cosine comparison
- **`age_days` property** on `ContextUnit` (was documented in api_reference.md but missing from the class)
- **Scope validation at public API boundary** — `validate_scope_keys()` called in `astore()`, `astore_batch()`, and `aretrieve()` regardless of backend
- **Shared `cosine_similarity`** utility in `contextidx/utils/math_utils.py` — deduplicated from `conflict_resolver.py` and `consolidation.py`
- **PostgreSQL auto-routing** for metadata store — when the pgvector backend is used with a `postgres_url`, the metadata store defaults to `PostgresStore` instead of SQLite
- **Sync wrappers** detect existing event loops (Jupyter, FastAPI, LangChain) and raise a clear `RuntimeError` with instructions instead of silently deadlocking
- **LangChain integration** (`contextidx/integrations/langchain.py`) — `ContextIdxMemory` implementing `BaseMemory`
- **LlamaIndex integration** (`contextidx/integrations/llamaindex.py`) — `ContextIdxRetriever` implementing `BaseRetriever`
- **Benchmark suite** — hybrid search, conflict detection, consolidation at scale (50K/100K), concurrent writes, WAL replay
- **Stored benchmark baseline** (`.benchmarks/`) and CI regression check (`--benchmark-compare-fail=mean:10%`)
- **Property-based tests** (hypothesis) for decay functions and composite scoring
- **Temporal accuracy tests** (`tests/test_accuracy/test_temporal_scenarios.py`) — stale retrieval, supersession chains, contradiction resolution
- **Re-ranking tests** with mocked OpenAI responses (`tests/test_core/test_reranking.py`)
- **Server tests** for `/ingest`, `/search`, and chunking logic (`tests/test_integration/test_server.py`)
- **Raw pgvector baseline provider** (`scripts/pgvector_baseline.py`) for LoCoMo delta measurement
- **Ablation study script** (`scripts/ablation_study.py`) — grid-search over scoring weight configurations against LoCoMo
- **GitHub Actions CI** (`.github/workflows/ci.yml`) — Python tests, Rust tests, ruff lint, mypy, benchmark regression check
- **`CONTEXTIDX_WINDOW_SIZE` / `CONTEXTIDX_STRIDE`** env vars to configure server chunking without code changes

### Fixed
- `RedisPendingBuffer.add()` was async while `PendingBuffer.add()` was sync — callers in `contextidx.py` did not `await` it, silently dropping all Redis writes
- `aclear()` did not clear the in-memory `TemporalGraph` — stale edges caused `KeyError` on subsequent `aretrieve()` calls
- Re-ranking `AsyncOpenAI` client was constructed on every `aretrieve()` call, wasting connection-pool setup time
- `_decay_tick()` issued N+1 database queries per interval (one `get_decay_state` + one `upsert_decay_state` per active unit) — replaced with batch load/upsert
- `PendingBuffer.remove()` scanned all scope buckets linearly — now uses a reverse-lookup index
- SQL injection risk in pgvector filter-clause key concatenation — keys now validated against an allowlist

### Changed
- Scoring default weights updated (semantic 0.35→0.30, recency 0.20→0.25, confidence 0.15→0.20, decay 0.15→0.10, reinforcement 0.10→0.05, bm25 0.05→0.10)
- `docs/configuration.md` scoring weights now match `scoring_engine.py` defaults
- `docs/backends.md` PGVector hybrid search corrected to "Yes (BM25 + vector RRF)"
- `Cargo.toml` version aligned to `1.0.0` (was `0.5.0`)

---

## [0.5.0] — 2026-02-15

### Added
- Initial public release
- pgvector, Qdrant, Weaviate, ChromaDB, Pinecone backends
- 6-signal composite scoring (semantic, recency, confidence, decay, reinforcement, BM25)
- Rule-based, semantic, and tiered conflict detection
- Write-Ahead Log with async replay
- Redis-backed pending buffer
- FastAPI memorybench server
- LoCoMo accuracy benchmarks (74% overall, 85.7% temporal, 83.3% multi-hop, 57.9% single-hop)
- Rust core via PyO3/maturin for decay computation
