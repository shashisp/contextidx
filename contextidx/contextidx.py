"""Main ContextIdx class — public API for the contextidx library."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Literal

from tenacity import (
    retry,
    retry_base,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from contextidx.backends.base import VectorBackend
from contextidx.config import ContextIdxConfig
from contextidx.core.conflict_resolver import ConflictJudgeFn, ConflictResolver
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.decay_engine import DecayEngine
from contextidx.core.embedding import EmbeddingFunction
from contextidx.core.query_type import detect_query_type, weights_for_query
from contextidx.core.reranker import OpenAIReranker, RerankerFn
from contextidx.core.scoring_engine import ScoringEngine
from contextidx.core.temporal_graph import Relationship, TemporalGraph
from contextidx.exceptions import (
    BackendError,
    ConfigurationError,
    EmbeddingError,
    StoreError,
)
from contextidx.store.base import Store, validate_scope_keys
from contextidx.store.sqlite_store import SQLiteStore
from contextidx.utils.batch_writer import BatchWriter
from contextidx.utils.math_utils import cosine_similarity
from contextidx.utils.conflict_queue import ConflictQueue
from contextidx.utils.pending_buffer import PendingBuffer
from contextidx.utils.wal import WAL

logger = logging.getLogger("contextidx")


def _run_sync(coro):
    """Run *coro* synchronously, raising a clear error inside async contexts.

    ``asyncio.run()`` raises ``RuntimeError`` when called from a running event
    loop (Jupyter, FastAPI, LangChain, …).  We detect that case early and give
    the caller a helpful message instead of an obscure traceback.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        raise RuntimeError(
            "Cannot call a synchronous ContextIdx wrapper from an async context "
            "(e.g. inside an async function, Jupyter notebook, or FastAPI handler). "
            "Use the async variant instead — e.g. 'await idx.astore(...)' rather "
            "than 'idx.store(...)'."
        )
    return asyncio.run(coro)


class OpenAIEmbeddingProvider:
    """Default embedding provider using the OpenAI API.

    Implements :class:`~contextidx.core.embedding.EmbeddingFunction`.
    """

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self._model = model
        self._client: object | None = None
        self._api_key = api_key

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise EmbeddingError(
                    "openai package required for default embedding provider. "
                    "Install with: pip install openai  — or supply a custom "
                    "embedding_fn to ContextIdx."
                )

    async def embed(self, text: str) -> list[float]:
        self._ensure_client()
        try:
            response = await self._client.embeddings.create(  # type: ignore[union-attr]
                input=text,
                model=self._model,
            )
            return response.data[0].embedding
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding API call failed: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self._ensure_client()
        try:
            response = await self._client.embeddings.create(  # type: ignore[union-attr]
                input=texts,
                model=self._model,
            )
            return [d.embedding for d in response.data]
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding API call failed: {exc}") from exc


class ContextIdx:
    """Temporal context layer for vector databases.

    Plugs into an existing vector backend and adds temporal awareness,
    decay, scoping, conflict resolution, and hybrid retrieval.
    """

    def __init__(
        self,
        backend: VectorBackend,
        *,
        decay_model: Literal["exponential", "linear", "step"] = "exponential",
        half_life_days: float = 30.0,
        conflict_detection: Literal["rule_based", "semantic", "tiered", "llm"] = "rule_based",
        conflict_strategy: Literal[
            "LAST_WRITE_WINS", "HIGHEST_CONFIDENCE", "MERGE", "MANUAL"
        ] = "LAST_WRITE_WINS",
        conflict_judge_fn: ConflictJudgeFn | None = None,
        scoring_weights: dict[str, float] | None = None,
        internal_store: Store | None = None,
        internal_store_path: str | None = None,
        internal_store_type: Literal["sqlite", "postgres", "auto"] = "auto",
        internal_store_dsn: str | None = None,
        embedding_fn: EmbeddingFunction | None = None,
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
        recency_bias: float | None = None,
        reranker: RerankerFn | None = None,
        max_graph_edge_nodes: int | None = None,
        config: ContextIdxConfig | None = None,
    ):
        # ── Validate configuration ──
        if half_life_days <= 0:
            raise ConfigurationError(
                f"half_life_days must be > 0, got {half_life_days}"
            )
        if not 0 <= decay_threshold <= 1:
            raise ConfigurationError(
                f"decay_threshold must be in [0, 1], got {decay_threshold}"
            )
        if state_path_interval <= 0:
            raise ConfigurationError(
                f"state_path_interval must be > 0, got {state_path_interval}"
            )
        if batch_size < 1:
            raise ConfigurationError(f"batch_size must be >= 1, got {batch_size}")
        if batch_flush_interval <= 0:
            raise ConfigurationError(
                f"batch_flush_interval must be > 0, got {batch_flush_interval}"
            )
        if reconcile_every_n_ticks < 1:
            raise ConfigurationError(
                f"reconcile_every_n_ticks must be >= 1, got {reconcile_every_n_ticks}"
            )
        if consolidation_every_n_ticks < 1:
            raise ConfigurationError(
                f"consolidation_every_n_ticks must be >= 1, got {consolidation_every_n_ticks}"
            )
        if wal_compact_every_n_ticks < 1:
            raise ConfigurationError(
                f"wal_compact_every_n_ticks must be >= 1, got {wal_compact_every_n_ticks}"
            )
        if (
            internal_store_type == "postgres"
            and not internal_store_dsn
            and internal_store is None
        ):
            raise ConfigurationError(
                "internal_store_dsn is required when internal_store_type='postgres'"
            )
        if pending_buffer_type == "redis" and not redis_url:
            raise ConfigurationError(
                "redis_url is required when pending_buffer_type='redis'"
            )
        if recency_bias is not None and not 0 <= recency_bias <= 1:
            raise ConfigurationError(
                f"recency_bias must be in [0, 1], got {recency_bias}"
            )

        self._cfg = config or ContextIdxConfig()
        # scoring_weights kwarg takes precedence over config.scoring_weights
        _weights = dict(self._cfg.scoring_weights)
        if scoring_weights:
            _weights.update(scoring_weights)

        self._backend = backend
        self._decay_model = decay_model
        self._decay_rate = ContextUnit.decay_rate_from_half_life(half_life_days)
        self._half_life_days = half_life_days
        self._conflict_detection = conflict_detection

        self._decay_engine = DecayEngine()
        self._scoring_engine = ScoringEngine(
            weights=_weights or None,
            recency_half_life_days=self._cfg.recency_half_life_days,
            reinforcement_saturation=self._cfg.reinforcement_saturation,
        )
        self._conflict_resolver = ConflictResolver(
            strategy=conflict_strategy,
            conflict_judge_fn=conflict_judge_fn,
        )
        self._graph = TemporalGraph(max_edge_nodes=max_graph_edge_nodes)

        # Pending buffer: in-memory or Redis-backed
        if pending_buffer_type == "redis":
            from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
            self._pending: PendingBuffer | RedisPendingBuffer = RedisPendingBuffer(
                redis_url=redis_url,  # type: ignore[arg-type]  # validated above
            )
        else:
            self._pending = PendingBuffer()

        # Store auto-routing (see plan architecture diagram)
        self._store: Store
        if internal_store is not None:
            self._store = internal_store
        elif (
            internal_store_type == "postgres"
            or (internal_store_dsn and internal_store_dsn.startswith("postgresql://"))
        ):
            from contextidx.store.postgres_store import PostgresStore
            self._store = PostgresStore(dsn=internal_store_dsn or "")
            logger.info("Using PostgresStore for metadata")
        elif backend.supports_metadata_store and internal_store_type == "auto":
            from contextidx.store.backend_metadata_store import BackendMetadataStore

            path = internal_store_path or ".contextidx/graph.db"
            # When a Postgres DSN is provided alongside a metadata-capable backend,
            # use PostgresStore for the graph/WAL tables so data survives pod restarts.
            graph_store: Store | None = None
            if internal_store_dsn:
                from contextidx.store.postgres_store import PostgresStore
                graph_store = PostgresStore(dsn=internal_store_dsn)
                logger.info(
                    "Using backend as metadata store; PostgresStore for graph/WAL"
                )
            else:
                logger.info(
                    "Using backend as metadata store; SQLite for graph/WAL only"
                )
            self._store = BackendMetadataStore(
                backend, graph_store=graph_store, graph_store_path=path
            )
        else:
            path = internal_store_path or ".contextidx/meta.db"
            self._store = SQLiteStore(path=path)
            logger.info("Auto-provisioned SQLite store at %s", path)

        self._conflict_queue: ConflictQueue | None = None
        if conflict_detection == "tiered":
            self._conflict_queue = ConflictQueue(self._conflict_resolver)

        self._wal: WAL | None = None
        self._embedder: EmbeddingFunction = (
            embedding_fn
            if embedding_fn is not None
            else OpenAIEmbeddingProvider(api_key=openai_api_key, model=embedding_model)
        )
        self._state_path_interval = state_path_interval
        self._decay_threshold = decay_threshold
        self._recency_bias = recency_bias or 0.0
        self._reconcile_every_n_ticks = reconcile_every_n_ticks
        self._consolidation_every_n_ticks = consolidation_every_n_ticks
        self._wal_compact_every_n_ticks = wal_compact_every_n_ticks
        self._tick_count = 0
        self._state_task: asyncio.Task | None = None
        self._running = False
        self._initialized = False

        # Batch writer (optional)
        self._enable_batching = enable_batching
        self._batch_writer: BatchWriter | None = None
        if enable_batching:
            self._batch_writer = BatchWriter(
                store_fn=self._astore_direct,
                embed_batch_fn=self._embed_batch,
                batch_size=batch_size,
                flush_interval=batch_flush_interval,
            )

        # Re-ranking — use the injected reranker or lazy-init OpenAIReranker on first use
        self._reranker: RerankerFn | None = reranker
        self._rerank_client: OpenAIReranker | None = None

    # ── Lifecycle ──

    async def __aenter__(self) -> ContextIdx:
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        await self.aclose()

    async def ainitialize(self) -> None:
        """Initialize stores, replay WAL, start background tasks."""
        if self._initialized:
            return
        await self._store.initialize()
        await self._backend.initialize()
        self._wal = WAL(self._store)
        await self._replay_wal()
        await self._load_graph()
        self._running = True
        self._state_task = asyncio.create_task(self._state_path_loop())
        if self._batch_writer is not None:
            await self._batch_writer.start()
        self._initialized = True
        logger.info("contextidx initialized")

    async def aclose(self) -> None:
        """Shutdown: stop background tasks, close stores."""
        self._running = False
        if self._batch_writer is not None:
            await self._batch_writer.stop()
        if self._state_task and not self._state_task.done():
            self._state_task.cancel()
            try:
                await self._state_task
            except asyncio.CancelledError:
                pass
        if hasattr(self._pending, "close"):
            await self._pending.close()
        await self._store.close()
        await self._backend.close()
        self._initialized = False

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("ContextIdx not initialized. Call ainitialize() first.")

    # ── Resilience helpers ──

    class _retry_rate_limit(retry_base):
        """Retry on transient errors and OpenAI rate-limit / quota 429s."""

        def __call__(self, retry_state: "RetryCallState") -> bool:  # type: ignore[override]
            exc = retry_state.outcome.exception() if retry_state.outcome else None
            if exc is None:
                return False
            if isinstance(exc, (OSError, ConnectionError, TimeoutError)):
                return True
            if isinstance(exc, EmbeddingError) and "429" in str(exc):
                return True
            return False

    _retry_transient = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.2, min=0.1, max=2),
        retry=retry_if_exception_type((OSError, ConnectionError, TimeoutError)),
        reraise=True,
    )

    _retry_embedding = retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=120),
        retry=_retry_rate_limit(),
        reraise=True,
    )

    @_retry_embedding
    async def _embed(self, text: str) -> list[float]:
        try:
            return await self._embedder.embed(text)
        except (OSError, ConnectionError, TimeoutError):
            raise
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Embedding failed: {exc}") from exc

    @_retry_embedding
    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._embedder.embed_batch(texts)
        except (OSError, ConnectionError, TimeoutError):
            raise
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Batch embedding failed: {exc}") from exc

    @_retry_transient
    async def _backend_store(
        self, *, id: str, embedding: list[float], metadata: dict
    ) -> None:
        try:
            await self._backend.store(id=id, embedding=embedding, metadata=metadata)
        except (OSError, ConnectionError, TimeoutError):
            raise
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Backend store failed: {exc}") from exc

    @_retry_transient
    async def _backend_search(
        self, *, query_embedding: list[float], top_k: int, filters: dict | None
    ) -> list:
        try:
            return await self._backend.search(
                query_embedding=query_embedding, top_k=top_k, filters=filters,
            )
        except (OSError, ConnectionError, TimeoutError):
            raise
        except BackendError:
            raise
        except Exception as exc:
            raise BackendError(f"Backend search failed: {exc}") from exc

    # ── Write Path ──

    async def astore(
        self,
        content: str,
        scope: dict[str, str],
        confidence: float = 0.8,
        source: str = "unknown",
        embedding: list[float] | None = None,
        decay_model: str | None = None,
        decay_rate: float | None = None,
        expires_at: datetime | None = None,
        wait_for_conflict: bool = False,
        timestamp: datetime | None = None,
    ) -> str:
        """Store a context unit (async).

        When batching is enabled and no pre-computed embedding is provided,
        the write is routed through the ``BatchWriter`` for amortised
        embedding costs.  Otherwise the direct path is used.

        Returns the stored unit ID.
        """
        self._ensure_initialized()
        validate_scope_keys(scope)

        # WAL circuit-breaker: refuse new writes when pending entries are at the limit
        if (
            self._wal is not None
            and self._cfg.max_wal_entries > 0
            and await self._wal.pending_count() >= self._cfg.max_wal_entries
        ):
            raise BackendError(
                f"WAL circuit-breaker: {self._cfg.max_wal_entries} pending entries "
                "reached. The vector backend may be unavailable. New writes are "
                "blocked until the WAL drains. Check backend connectivity."
            )

        if self._batch_writer is not None and embedding is None:
            future = await self._batch_writer.add(
                content=content,
                scope=scope,
                confidence=confidence,
                source=source,
                decay_model=decay_model,
                decay_rate=decay_rate,
                expires_at=expires_at,
            )
            return await future

        return await self._astore_direct(
            content=content,
            scope=scope,
            confidence=confidence,
            source=source,
            embedding=embedding,
            decay_model=decay_model,
            decay_rate=decay_rate,
            expires_at=expires_at,
            wait_for_conflict=wait_for_conflict,
            timestamp=timestamp,
        )

    async def _astore_direct(
        self,
        content: str,
        scope: dict[str, str],
        confidence: float = 0.8,
        source: str = "unknown",
        embedding: list[float] | None = None,
        decay_model: str | None = None,
        decay_rate: float | None = None,
        expires_at: datetime | None = None,
        wait_for_conflict: bool = False,
        timestamp: datetime | None = None,
    ) -> str:
        """Direct (non-batched) store path."""
        self._ensure_initialized()
        assert self._wal is not None

        unit = ContextUnit(
            id=generate_unit_id(),
            content=content,
            scope=scope,
            confidence=confidence,
            source=source,
            decay_model=decay_model or self._decay_model,  # type: ignore[arg-type]
            decay_rate=decay_rate or self._decay_rate,
            expires_at=expires_at,
        )
        if timestamp is not None:
            unit.timestamp = timestamp

        if embedding is not None:
            unit.embedding = embedding
        else:
            unit.embedding = await self._embed(content)

        wal_payload = unit.model_dump(mode="json")
        wal_payload.pop("embedding", None)  # embeddings are re-computed on WAL replay
        wal_seq = await self._wal.append(
            unit_id=unit.id,
            operation="store",
            store_target="both",
            payload=wal_payload,
        )

        await self._pending.add(unit)

        existing = await self._store.find_units_in_scope(scope)
        superseded_units: list = []

        if self._conflict_detection == "tiered":
            inline, candidates = self._conflict_resolver.detect_tiered(unit, existing)
            if inline:
                result = self._conflict_resolver.resolve(unit, inline)
                unit = result.winner
                superseded_units = result.superseded
            if candidates and self._conflict_queue is not None:
                await self._conflict_queue.enqueue(unit, candidates)
        elif self._conflict_detection == "llm":
            conflicts = await self._conflict_resolver.detect_llm_conflicts(unit, existing)
            if conflicts:
                result = self._conflict_resolver.resolve(unit, conflicts)
                unit = result.winner
                superseded_units = result.superseded
        elif self._conflict_detection == "semantic":
            conflicts = self._conflict_resolver.detect_semantic_conflicts(unit, existing)
            if conflicts:
                result = self._conflict_resolver.resolve(unit, conflicts)
                unit = result.winner
                superseded_units = result.superseded
        else:
            conflicts = self._conflict_resolver.detect_conflicts(unit, existing)
            if conflicts:
                result = self._conflict_resolver.resolve(unit, conflicts)
                unit = result.winner
                superseded_units = result.superseded

        await self._backend_store(
            id=unit.id,
            embedding=unit.embedding,
            metadata={"scope": unit.scope, "source": unit.source, "_content": unit.content},
        )
        await self._store.create_unit(unit)

        for superseded in superseded_units:
            await self._store.update_unit(
                superseded.id, {"superseded_by": unit.id}
            )
            now = datetime.now(timezone.utc)
            self._graph.add_edge(
                unit.id,
                superseded.id,
                Relationship.SUPERSEDES,
                now,
            )
            await self._store.add_graph_edge(
                unit.id, superseded.id, "supersedes", now
            )

        now = datetime.now(timezone.utc)
        decay_score = self._decay_engine.compute_decay(unit, now)
        await self._store.upsert_decay_state(unit.id, decay_score, now, 0)

        await self._wal.mark_applied(wal_seq)
        logger.debug("Stored unit %s", unit.id)
        return unit.id

    async def astore_batch(self, items: list[dict]) -> list[str]:
        """Store multiple context units, embedding them in a single batch call.

        Each dict in *items* must contain ``content`` and ``scope``; other
        keys are forwarded to ``astore()``.  Returns a list of unit IDs in
        the same order as *items*.
        """
        self._ensure_initialized()
        for item in items:
            validate_scope_keys(item.get("scope", {}))
        texts = [item["content"] for item in items]
        embeddings = await self._embed_batch(texts)
        ids: list[str] = []
        for item, emb in zip(items, embeddings):
            uid = await self._astore_direct(embedding=emb, **item)
            ids.append(uid)
        return ids

    def store(
        self,
        content: str,
        scope: dict[str, str],
        confidence: float = 0.8,
        source: str = "unknown",
        embedding: list[float] | None = None,
        **kwargs,
    ) -> str:
        """Store a context unit (sync wrapper)."""
        return _run_sync(
            self.astore(
                content=content,
                scope=scope,
                confidence=confidence,
                source=source,
                embedding=embedding,
                **kwargs,
            )
        )

    # ── Read Path ──

    async def aretrieve(
        self,
        query: str,
        scope: dict[str, str],
        top_k: int = 5,
        at: datetime | None = None,
        query_embedding: list[float] | None = None,
        min_score: float = 0.0,
        recency_bias: float | None = None,
        rerank: bool = False,
    ) -> list[ContextUnit]:
        """Retrieve temporally-scored context units (async).

        Args:
            query: Natural language query.
            scope: Scope filter (e.g. {"user_id": "u123"}).
            top_k: Number of results to return.
            at: Point-in-time for time-travel queries. ``None`` means now.
            query_embedding: Pre-computed query embedding (skips API call).
            min_score: Minimum composite score threshold.
            recency_bias: When set (0.0-1.0), items whose decay score is
                below ``recency_bias * max_decay`` among candidates are
                dropped.  Overrides the instance-level ``_recency_bias``
                if provided.  ``None`` falls back to the constructor value.
            rerank: When True, apply an LLM-based re-ranking step to
                improve precision. Uses gpt-4o-mini for low latency.
        """
        self._ensure_initialized()
        validate_scope_keys(scope)

        query_time = at or datetime.now(timezone.utc)

        # 1. Embed query
        if query_embedding is not None:
            q_emb = query_embedding
        else:
            q_emb = await self._embed(query)

        # 2. Over-fetch from vector backend — use hybrid when supported
        fetch_k = top_k * self._cfg.overfetch_factor
        use_hybrid = getattr(self._backend, "supports_hybrid_search", False)

        if use_hybrid:
            raw_results = await self._backend.hybrid_search(
                query=query,
                query_embedding=q_emb,
                top_k=fetch_k,
                filters=scope,
            )
        else:
            raw_results = await self._backend_search(
                query_embedding=q_emb,
                top_k=fetch_k,
                filters=scope,
            )

        # 3. Collect all IDs we need, then batch-load from the store
        pending = await self._pending.get(scope)
        pending_ids = list({pu.id for pu in pending})
        result_ids = [sr.id for sr in raw_results if sr.id not in set(pending_ids)]

        all_ids = pending_ids + result_ids
        units_map = await self._store.get_units_batch(all_ids) if all_ids else {}

        # Build candidate list: pending units first, then vector results
        candidates: list[tuple[ContextUnit, float, float | None]] = []
        seen_ids: set[str] = set()

        for pu in pending:
            if pu.id not in seen_ids:
                unit = units_map.get(pu.id, pu)
                candidates.append((unit, 1.0, None))
                seen_ids.add(pu.id)

        for sr in raw_results:
            if sr.id in seen_ids:
                continue
            seen_ids.add(sr.id)
            unit = units_map.get(sr.id)
            if unit is None:
                continue
            bm25 = sr.metadata.get("bm25_score") if use_hybrid else None
            candidates.append((unit, sr.score, bm25))

        # 4. Filter (collect superseder IDs for batch load in time-travel mode)
        superseder_ids: list[str] = []
        if at is not None:
            for unit, _, _ in candidates:
                sup_id = self._graph.find_superseded_by(unit.id)
                if sup_id and sup_id not in units_map:
                    superseder_ids.append(sup_id)
            if superseder_ids:
                sup_units = await self._store.get_units_batch(superseder_ids)
                units_map.update(sup_units)

        filtered: list[tuple[ContextUnit, float, float | None]] = []
        for unit, sem_score, bm25 in candidates:
            if not unit.matches_scope(scope):
                continue
            if at is None:
                if unit.is_superseded:
                    continue
                if unit.is_expired:
                    continue
            else:
                if unit.timestamp > at:
                    continue
                if unit.is_expired_at(at):
                    continue
                superseder = self._graph.find_superseded_by(unit.id)
                if superseder:
                    sup_unit = units_map.get(superseder)
                    if sup_unit and sup_unit.timestamp <= at:
                        continue

            decay_score = self._decay_engine.compute_decay(unit, query_time)
            if decay_score < self._decay_threshold:
                continue

            filtered.append((unit, sem_score, bm25))

        # 4b. Graph expansion: follow RELATES_TO edges from top candidates
        # to pull in neighboring context that vector search may have missed.
        # When max_graph_edge_nodes is set, a node's edges may have been evicted
        # from the in-memory graph; fall back to the store for those nodes.
        expansion_ids: set[str] = set()
        filtered_id_set = {u.id for u, _, _ in filtered}
        for unit, _, _ in filtered[:top_k]:
            related = self._graph.get_related(unit.id)
            if not related and self._graph.was_evicted(unit.id):
                # In-memory edges evicted — reload from store for this node
                db_edges = await self._store.get_graph_edges(unit.id)
                for from_id, to_id, rel, created_at in db_edges:
                    if rel == "relates_to":
                        related.append(to_id if from_id == unit.id else from_id)
            for rid in related:
                if rid not in filtered_id_set and rid not in seen_ids:
                    expansion_ids.add(rid)

        if expansion_ids:
            expanded_units = await self._store.get_units_batch(list(expansion_ids))
            for uid, exp_unit in expanded_units.items():
                if exp_unit is None:
                    continue
                if exp_unit.is_superseded or exp_unit.is_expired:
                    continue
                if not exp_unit.matches_scope(scope):
                    continue
                decay_sc = self._decay_engine.compute_decay(exp_unit, query_time)
                if decay_sc < self._decay_threshold:
                    continue
                if q_emb and exp_unit.embedding:
                    sim = cosine_similarity(q_emb, exp_unit.embedding)
                    if sim < self._cfg.graph_expansion_min_score:
                        continue
                    exp_score = sim
                else:
                    exp_score = self._cfg.graph_expansion_default_score
                filtered.append((exp_unit, exp_score, None))
                filtered_id_set.add(uid)

        # 5. Batch-load decay states, then score and rank
        filtered_ids = [unit.id for unit, _, _ in filtered]
        decay_states = (
            await self._store.get_decay_states_batch(filtered_ids)
            if filtered_ids
            else {}
        )

        # Detect query type and build a scoring engine with adapted weights
        q_type = detect_query_type(query)
        dynamic_weights = weights_for_query(q_type)
        # Respect any user-supplied weight overrides on top of dynamic weights
        if self._cfg.scoring_weights:
            dynamic_weights.update(self._cfg.scoring_weights)
        scoring_engine = ScoringEngine(
            weights=dynamic_weights,
            recency_half_life_days=self._cfg.recency_half_life_days,
            reinforcement_saturation=self._cfg.reinforcement_saturation,
        )

        scored: list[tuple[ContextUnit, float, float]] = []
        for unit, sem_score, bm25 in filtered:
            state = decay_states.get(unit.id)
            reinforcement_count = state[2] if state else 0
            decay_score = self._decay_engine.compute_decay(
                unit, query_time, reinforcement_count
            )
            composite = scoring_engine.compute_score(
                unit=unit,
                semantic_score=sem_score,
                query_time=query_time,
                decay_score=decay_score,
                reinforcement_count=reinforcement_count,
                bm25_score=bm25,
            )
            if composite >= min_score:
                scored.append((unit, composite, decay_score))

        effective_bias = recency_bias if recency_bias is not None else self._recency_bias
        if effective_bias and effective_bias > 0 and scored:
            max_decay = max(ds for _, _, ds in scored)
            cutoff = effective_bias * max_decay
            scored = [(u, c, ds) for u, c, ds in scored if ds >= cutoff]

        scored.sort(key=lambda x: x[1], reverse=True)

        # 7. Optional LLM re-ranking for higher precision
        if rerank and scored:
            scored = await self._rerank_with_llm(query, scored, top_k)

        results: list[ContextUnit] = []
        for unit, composite, _ in scored[:top_k]:
            unit.confidence = composite
            results.append(unit)
        return results

    async def _rerank_with_llm(
        self,
        query: str,
        scored: list[tuple[ContextUnit, float, float]],
        top_k: int,
    ) -> list[tuple[ContextUnit, float, float]]:
        """Re-rank candidates using the configured reranker.

        Uses the injected ``reranker`` if provided; otherwise lazy-inits an
        :class:`~contextidx.core.reranker.OpenAIReranker`.  Falls back to
        the original ordering on any error.
        """
        if len(scored) <= 1:
            return scored

        reranker = self._reranker
        if reranker is None:
            if self._rerank_client is None:
                try:
                    self._rerank_client = OpenAIReranker()
                except ImportError:
                    logger.warning("openai not installed; skipping LLM re-rank")
                    return scored
            reranker = self._rerank_client

        try:
            return await reranker(query, scored, top_k)
        except Exception:
            logger.warning("LLM re-rank failed; using original scores", exc_info=True)
            return scored

    def retrieve(
        self,
        query: str,
        scope: dict[str, str],
        top_k: int = 5,
        at: datetime | None = None,
        query_embedding: list[float] | None = None,
        recency_bias: float | None = None,
        **kwargs,
    ) -> list[ContextUnit]:
        """Retrieve temporally-scored context units (sync wrapper)."""
        return _run_sync(
            self.aretrieve(
                query=query,
                scope=scope,
                top_k=top_k,
                at=at,
                query_embedding=query_embedding,
                recency_bias=recency_bias,
                **kwargs,
            )
        )

    # ── Supersede ──

    async def asupersede(self, new_id: str, old_id: str) -> None:
        """Explicitly mark *old_id* as superseded by *new_id*.

        Use this when automatic semantic detection is insufficient and
        the application knows that a newer unit replaces an older one.
        """
        self._ensure_initialized()
        old_unit = await self._store.get_unit(old_id)
        if old_unit is None:
            raise ValueError(f"Unit {old_id!r} not found")
        new_unit = await self._store.get_unit(new_id)
        if new_unit is None:
            raise ValueError(f"Unit {new_id!r} not found")
        await self._store.update_unit(old_id, {"superseded_by": new_id})
        now = datetime.now(timezone.utc)
        self._graph.add_edge(new_id, old_id, Relationship.SUPERSEDES, now)
        await self._store.add_graph_edge(new_id, old_id, "supersedes", now)
        logger.debug("Explicit supersede: %s supersedes %s", new_id, old_id)

    def supersede(self, new_id: str, old_id: str) -> None:
        """Sync wrapper for :meth:`asupersede`."""
        _run_sync(self.asupersede(new_id, old_id))

    # ── Graph linking ──

    async def alink_related(self, id_a: str, id_b: str) -> None:
        """Create a bidirectional RELATES_TO edge between two units.

        Used to connect chunks from the same session so that graph-expanded
        retrieval can pull in neighboring context.
        """
        self._ensure_initialized()
        now = datetime.now(timezone.utc)
        self._graph.add_edge(id_a, id_b, Relationship.RELATES_TO, now)
        await self._store.add_graph_edge(id_a, id_b, "relates_to", now)

    # ── Clear ──

    async def aclear(self, scope: dict[str, str]) -> int:
        """Delete all units matching *scope* from both the store and backend.

        Returns the number of units removed.
        """
        self._ensure_initialized()
        units = await self._store.find_units_in_scope(
            scope, include_superseded=True, include_archived=True,
        )
        deleted_ids: set[str] = set()
        for unit in units:
            try:
                await self._backend.delete(unit.id)
            except Exception:
                logger.debug("Backend delete failed for %s (may not exist)", unit.id)
            await self._store.delete_unit(unit.id)
            deleted_ids.add(unit.id)
        self._graph.remove_units(deleted_ids)
        if hasattr(self._pending, "clear_scope"):
            result = self._pending.clear_scope(scope)
            if asyncio.iscoroutine(result):
                await result
        logger.info("Cleared %d units from scope %s", len(units), scope)
        return len(units)

    def clear(self, scope: dict[str, str]) -> int:
        """Sync wrapper for :meth:`aclear`."""
        return _run_sync(self.aclear(scope))

    # ── Reinforce ──

    async def areinforce(self, unit_id: str) -> None:
        """Mark a context unit as used, partially resetting its decay clock."""
        self._ensure_initialized()
        count = await self._store.increment_reinforcement(unit_id)
        unit = await self._store.get_unit(unit_id)
        if unit:
            now = datetime.now(timezone.utc)
            score = self._decay_engine.compute_decay(unit, now, count)
            await self._store.upsert_decay_state(unit_id, score, now, count)
        logger.debug("Reinforced unit %s (count=%d)", unit_id, count)

    def reinforce(self, unit_id: str) -> None:
        """Sync wrapper for areinforce."""
        _run_sync(self.areinforce(unit_id))

    # ── Lineage ──

    async def alineage(self, unit_id: str) -> list[ContextUnit]:
        """Get full version history for a context unit."""
        self._ensure_initialized()
        chain = self._graph.get_lineage(unit_id)
        units: list[ContextUnit] = []
        for uid in chain:
            u = await self._store.get_unit(uid)
            if u:
                units.append(u)
        return units

    def lineage(self, unit_id: str) -> list[ContextUnit]:
        return asyncio.run(self.alineage(unit_id))

    # ── Diff ──

    async def adiff(
        self, scope: dict[str, str], since: str | datetime
    ) -> list[ContextUnit]:
        """Return context units created or modified since a given time.

        ``since`` accepts a datetime or a duration string like ``"30d"``, ``"7d"``, ``"24h"``.
        """
        self._ensure_initialized()
        if isinstance(since, str):
            since_dt = self._parse_duration(since)
        else:
            since_dt = since
        units = await self._store.find_units_in_scope(
            scope, include_superseded=True
        )
        return [u for u in units if u.timestamp >= since_dt]

    def diff(self, scope: dict[str, str], since: str | datetime) -> list[ContextUnit]:
        return asyncio.run(self.adiff(scope, since))

    # ── Reconciliation ──

    async def areconcile(self) -> dict:
        """Detect and repair store drift between internal store and vector backend.

        Returns a dict with reconciliation stats.
        """
        self._ensure_initialized()

        checkpoint = await self._store.get_checkpoint("vector_backend")
        if checkpoint:
            since, _ = checkpoint
        else:
            since = datetime.min.replace(tzinfo=timezone.utc)

        unit_ids = await self._store.find_active_without_vector(since)

        reinserted = 0
        errors = 0
        for uid in unit_ids:
            unit = await self._store.get_unit(uid)
            if unit is None:
                continue
            try:
                results = await self._backend_search(
                    query_embedding=unit.embedding or [],
                    top_k=1,
                    filters=unit.scope,
                )
                found = any(r.id == uid for r in results)
                if not found and unit.embedding:
                    await self._backend_store(
                        id=unit.id,
                        embedding=unit.embedding,
                        metadata={"scope": unit.scope, "source": unit.source},
                    )
                    reinserted += 1
            except BackendError:
                logger.exception("Reconciliation failed for unit %s", uid)
                errors += 1

        now = datetime.now(timezone.utc)
        await self._store.update_checkpoint(
            "vector_backend", now, len(unit_ids)
        )

        stats = {
            "checked": len(unit_ids),
            "reinserted": reinserted,
            "errors": errors,
            "checkpoint": now.isoformat(),
        }
        logger.info("Reconciliation complete: %s", stats)
        return stats

    # ── Conflict Resolution Callback ──

    async def _apply_conflict_resolution(
        self, new_unit: ContextUnit, conflicts: list[ContextUnit]
    ) -> None:
        """Apply the result of semantic conflict detection (called by ConflictQueue)."""
        result = self._conflict_resolver.resolve(new_unit, conflicts)
        for superseded in result.superseded:
            await self._store.update_unit(
                superseded.id, {"superseded_by": result.winner.id}
            )
            now = datetime.now(timezone.utc)
            self._graph.add_edge(
                result.winner.id,
                superseded.id,
                Relationship.SUPERSEDES,
                now,
            )
            await self._store.add_graph_edge(
                result.winner.id, superseded.id, "supersedes", now
            )
        logger.debug(
            "Semantic conflict resolved: winner=%s, superseded=%d",
            result.winner.id, len(result.superseded),
        )

    # ── State Path ──

    async def _acquire_leader_lock(self) -> bool:
        """Try to acquire a distributed state-path leader lock via Redis.

        Returns ``True`` when this instance should run the tick:
        - Always ``True`` for non-Redis deployments (single-instance assumed).
        - For Redis deployments, only ``True`` when the ``SET NX EX`` succeeds,
          meaning no other instance already holds the lock.

        The lock TTL is ``2 × state_path_interval`` so a slow tick never blocks
        the next one from running on another instance after a restart.
        """
        try:
            from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
        except ImportError:
            return True  # redis not installed — single-instance mode

        if not isinstance(self._pending, RedisPendingBuffer):
            return True  # No Redis — single-instance mode, always leader

        ttl = max(1, int(self._state_path_interval * 2))
        acquired = await self._pending._redis.set(
            "ctxidx:state_path_leader", "1", nx=True, ex=ttl
        )
        if acquired is None:
            logger.debug("State path: another instance holds the leader lock; skipping tick")
        return acquired is not None

    async def _state_path_loop(self) -> None:
        while self._running:
            try:
                if await self._acquire_leader_lock():
                    await self._state_path_tick()
            except (StoreError, BackendError) as exc:
                logger.exception("State path tick error: %s", exc)
            except Exception:
                logger.exception("Unexpected state path tick error")
            await asyncio.sleep(self._state_path_interval)

    async def _state_path_tick(self) -> None:
        await self._decay_tick()
        await self._expiry_archive()

        if self._conflict_queue is not None:
            try:
                await self._conflict_queue.drain(self._apply_conflict_resolution)
            except (StoreError, BackendError) as exc:
                logger.exception("Conflict queue drain failed: %s", exc)
            except Exception:
                logger.exception("Unexpected conflict queue error")

        self._tick_count += 1
        if self._tick_count % self._reconcile_every_n_ticks == 0:
            try:
                await self.areconcile()
            except (StoreError, BackendError) as exc:
                logger.exception("Periodic reconciliation failed: %s", exc)
            except Exception:
                logger.exception("Unexpected reconciliation error")
        if self._tick_count % self._consolidation_every_n_ticks == 0:
            try:
                await self._consolidation_tick()
            except (StoreError, BackendError) as exc:
                logger.exception("Consolidation tick failed: %s", exc)
            except Exception:
                logger.exception("Unexpected consolidation error")
        if self._tick_count % self._wal_compact_every_n_ticks == 0:
            try:
                await self._wal_compact_tick()
            except (StoreError, BackendError) as exc:
                logger.exception("WAL compaction tick failed: %s", exc)
            except Exception:
                logger.exception("Unexpected WAL compaction error")

    async def _consolidation_tick(self) -> None:
        """Merge semantically redundant units within each scope."""
        from contextidx.core.consolidation import find_redundant_pairs, merge_units

        units = await self._store.find_active_units()
        pairs = await find_redundant_pairs(
            units,
            threshold=self._cfg.consolidation_threshold,
            ann_search_fn=self._backend.search,
        )
        merged = 0
        for keeper_id, absorbed_id in pairs:
            keeper = await self._store.get_unit(keeper_id)
            absorbed = await self._store.get_unit(absorbed_id)
            if keeper is None or absorbed is None:
                continue
            if absorbed.is_superseded:
                continue
            updated = merge_units(keeper, absorbed)
            await self._store.update_unit(
                keeper_id, {"confidence": updated.confidence, "version": updated.version}
            )
            await self._store.update_unit(absorbed_id, {"superseded_by": keeper_id})
            now = datetime.now(timezone.utc)
            self._graph.add_edge(keeper_id, absorbed_id, Relationship.SUPERSEDES, now)
            await self._store.add_graph_edge(keeper_id, absorbed_id, "supersedes", now)
            merged += 1
        if merged:
            logger.info("Consolidation merged %d redundant pairs", merged)

    async def _wal_compact_tick(self) -> None:
        """Remove applied WAL entries and optionally drop unrecoverable pending ones."""
        if self._wal is None:
            return
        removed = await self._wal.compact(retention_hours=self._cfg.wal_retention_hours)
        if removed:
            logger.info("WAL compaction removed %d applied entries", removed)
        if self._cfg.wal_max_age_hours > 0:
            dropped = await self._wal.drop_stale(
                max_age_hours=self._cfg.wal_max_age_hours
            )
            if dropped:
                logger.warning(
                    "WAL stale-drop removed %d unrecoverable pending entries", dropped
                )

    async def _decay_tick(self) -> None:
        """Recalculate decay scores for all active units."""
        units = await self._store.find_active_units()
        if not units:
            return
        now = datetime.now(timezone.utc)
        unit_ids = [u.id for u in units]
        decay_states = await self._store.get_decay_states_batch(unit_ids)
        batch: list[tuple[str, float, datetime, int]] = []
        for unit in units:
            state = decay_states.get(unit.id)
            rc = state[2] if state else 0
            score = self._decay_engine.compute_decay(unit, now, rc)
            batch.append((unit.id, score, now, rc))
        await self._store.upsert_decay_states_batch(batch)

    async def _expiry_archive(self) -> None:
        """Archive expired units using an indexed SQL query."""
        now = datetime.now(timezone.utc)
        expired_ids = await self._store.find_expired_units(now)
        for unit_id in expired_ids:
            await self._store.update_unit(unit_id, {"archived_at": now.isoformat()})

    # ── WAL Replay ──

    async def _replay_wal(self) -> None:
        assert self._wal is not None
        entries = await self._wal.replay_pending()
        if not entries:
            return
        logger.info("Replaying %d WAL entries", len(entries))
        for entry in entries:
            try:
                if entry.operation == "store":
                    unit = ContextUnit.model_validate(entry.payload)
                    existing = await self._store.get_unit(unit.id)
                    if existing is None:
                        if unit.embedding:
                            await self._backend_store(
                                id=unit.id,
                                embedding=unit.embedding,
                                metadata={"scope": unit.scope, "source": unit.source},
                            )
                        await self._store.create_unit(unit)
                await self._wal.mark_applied(entry.seq)
            except (StoreError, BackendError) as exc:
                logger.exception("WAL replay failed for seq=%d: %s", entry.seq, exc)
                await self._wal.mark_failed(entry.seq)
            except Exception:
                logger.exception("Unexpected WAL replay error for seq=%d", entry.seq)
                await self._wal.mark_failed(entry.seq)

    # ── Graph Loading ──

    async def _load_graph(self) -> None:
        """Bulk-load all graph edges from the store into memory (single query)."""
        all_edges = await self._store.get_all_graph_edges()
        for from_id, to_id, rel, created_at in all_edges:
            self._graph.add_edge(from_id, to_id, rel, created_at)

    # ── Helpers ──

    @staticmethod
    def _parse_duration(s: str) -> datetime:
        """Parse ``"30d"``, ``"7d"``, ``"24h"`` to a past datetime."""
        s = s.strip()
        now = datetime.now(timezone.utc)
        if s.endswith("d"):
            days = int(s[:-1])
            return now - timedelta(days=days)
        elif s.endswith("h"):
            hours = int(s[:-1])
            return now - timedelta(hours=hours)
        elif s.endswith("m"):
            minutes = int(s[:-1])
            return now - timedelta(minutes=minutes)
        else:
            raise ValueError(f"Unknown duration format: {s!r}. Use '30d', '24h', or '60m'.")
