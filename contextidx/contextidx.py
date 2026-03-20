"""Main ContextIdx class — public API for the contextidx library."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Literal

from contextidx.backends.base import VectorBackend
from contextidx.core.conflict_resolver import ConflictResolver
from contextidx.core.context_unit import ContextUnit, generate_unit_id
from contextidx.core.decay_engine import DecayEngine
from contextidx.core.scoring_engine import ScoringEngine
from contextidx.core.temporal_graph import Relationship, TemporalGraph
from contextidx.store.base import Store
from contextidx.store.sqlite_store import SQLiteStore
from contextidx.utils.batch_writer import BatchWriter
from contextidx.utils.conflict_queue import ConflictQueue
from contextidx.utils.pending_buffer import PendingBuffer
from contextidx.utils.wal import WAL

logger = logging.getLogger("contextidx")


class EmbeddingProvider:
    """Generates text embeddings via OpenAI-compatible API."""

    def __init__(self, api_key: str | None = None, model: str = "text-embedding-3-small"):
        self._model = model
        self._client: object | None = None
        self._api_key = api_key

    async def embed(self, text: str) -> list[float]:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package required for embedding generation. "
                    "Install with: pip install openai"
                )
        response = await self._client.embeddings.create(  # type: ignore[union-attr]
            input=text,
            model=self._model,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError("openai package required.")
        response = await self._client.embeddings.create(  # type: ignore[union-attr]
            input=texts,
            model=self._model,
        )
        return [d.embedding for d in response.data]


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
        conflict_detection: Literal["rule_based", "semantic", "tiered"] = "rule_based",
        conflict_strategy: Literal[
            "LAST_WRITE_WINS", "HIGHEST_CONFIDENCE", "MERGE", "MANUAL"
        ] = "LAST_WRITE_WINS",
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
    ):
        self._backend = backend
        self._decay_model = decay_model
        self._decay_rate = ContextUnit.decay_rate_from_half_life(half_life_days)
        self._half_life_days = half_life_days
        self._conflict_detection = conflict_detection

        self._decay_engine = DecayEngine()
        self._scoring_engine = ScoringEngine(weights=scoring_weights)
        self._conflict_resolver = ConflictResolver(strategy=conflict_strategy)
        self._graph = TemporalGraph()

        # Pending buffer: in-memory or Redis-backed
        if pending_buffer_type == "redis":
            from contextidx.utils.redis_pending_buffer import RedisPendingBuffer
            self._pending: PendingBuffer | RedisPendingBuffer = RedisPendingBuffer(
                redis_url=redis_url or "redis://localhost:6379/0",
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
            self._store = BackendMetadataStore(backend, graph_store_path=path)
            logger.info("Using backend as metadata store; SQLite for graph/WAL only")
        else:
            path = internal_store_path or ".contextidx/meta.db"
            self._store = SQLiteStore(path=path)
            logger.info("Auto-provisioned SQLite store at %s", path)

        self._conflict_queue: ConflictQueue | None = None
        if conflict_detection == "tiered":
            self._conflict_queue = ConflictQueue(self._conflict_resolver)

        self._wal: WAL | None = None
        self._embedder = EmbeddingProvider(api_key=openai_api_key, model=embedding_model)
        self._state_path_interval = state_path_interval
        self._decay_threshold = decay_threshold
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
                embed_batch_fn=self._embedder.embed_batch,
                batch_size=batch_size,
                flush_interval=batch_flush_interval,
            )

    # ── Lifecycle ──

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
    ) -> str:
        """Store a context unit (async).

        When batching is enabled and no pre-computed embedding is provided,
        the write is routed through the ``BatchWriter`` for amortised
        embedding costs.  Otherwise the direct path is used.

        Returns the stored unit ID.
        """
        self._ensure_initialized()

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

        if embedding is not None:
            unit.embedding = embedding
        else:
            unit.embedding = await self._embedder.embed(content)

        wal_seq = await self._wal.append(
            unit_id=unit.id,
            operation="store",
            store_target="both",
            payload=unit.model_dump(mode="json"),
        )

        self._pending.add(unit)

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

        await self._backend.store(
            id=unit.id,
            embedding=unit.embedding,
            metadata={"scope": unit.scope, "source": unit.source},
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
        texts = [item["content"] for item in items]
        embeddings = await self._embedder.embed_batch(texts)
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
        return asyncio.run(
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
    ) -> list[ContextUnit]:
        """Retrieve temporally-scored context units (async).

        Args:
            query: Natural language query.
            scope: Scope filter (e.g. {"user_id": "u123"}).
            top_k: Number of results to return.
            at: Point-in-time for time-travel queries. ``None`` means now.
            query_embedding: Pre-computed query embedding (skips API call).
            min_score: Minimum composite score threshold.
        """
        self._ensure_initialized()

        query_time = at or datetime.now(timezone.utc)

        # 1. Embed query
        if query_embedding is not None:
            q_emb = query_embedding
        else:
            q_emb = await self._embedder.embed(query)

        # 2. Over-fetch from vector backend — use hybrid when supported
        fetch_k = top_k * 3
        use_hybrid = getattr(self._backend, "supports_hybrid_search", False)

        if use_hybrid:
            raw_results = await self._backend.hybrid_search(
                query=query,
                query_embedding=q_emb,
                top_k=fetch_k,
                filters=scope,
            )
        else:
            raw_results = await self._backend.search(
                query_embedding=q_emb,
                top_k=fetch_k,
                filters=scope,
            )

        # 3. Load full units and merge pending buffer
        candidates: list[tuple[ContextUnit, float, float | None]] = []
        seen_ids: set[str] = set()

        pending = self._pending.get(scope)
        for pu in pending:
            if pu.id not in seen_ids:
                stored = await self._store.get_unit(pu.id)
                unit = stored if stored is not None else pu
                candidates.append((unit, 1.0, None))
                seen_ids.add(pu.id)

        for sr in raw_results:
            if sr.id in seen_ids:
                continue
            seen_ids.add(sr.id)
            unit = await self._store.get_unit(sr.id)
            if unit is None:
                continue
            bm25 = sr.metadata.get("bm25_score") if use_hybrid else None
            candidates.append((unit, sr.score, bm25))

        # 4. Filter
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
                    sup_unit = await self._store.get_unit(superseder)
                    if sup_unit and sup_unit.timestamp <= at:
                        continue

            decay_score = self._decay_engine.compute_decay(unit, query_time)
            if decay_score < self._decay_threshold:
                continue

            filtered.append((unit, sem_score, bm25))

        # 5. Score and rank
        scored: list[tuple[ContextUnit, float]] = []
        for unit, sem_score, bm25 in filtered:
            state = await self._store.get_decay_state(unit.id)
            reinforcement_count = state[2] if state else 0
            decay_score = self._decay_engine.compute_decay(
                unit, query_time, reinforcement_count
            )
            composite = self._scoring_engine.compute_score(
                unit=unit,
                semantic_score=sem_score,
                query_time=query_time,
                decay_score=decay_score,
                reinforcement_count=reinforcement_count,
                bm25_score=bm25,
            )
            if composite >= min_score:
                scored.append((unit, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [unit for unit, _ in scored[:top_k]]

    def retrieve(
        self,
        query: str,
        scope: dict[str, str],
        top_k: int = 5,
        at: datetime | None = None,
        query_embedding: list[float] | None = None,
        **kwargs,
    ) -> list[ContextUnit]:
        """Retrieve temporally-scored context units (sync wrapper)."""
        return asyncio.run(
            self.aretrieve(
                query=query,
                scope=scope,
                top_k=top_k,
                at=at,
                query_embedding=query_embedding,
                **kwargs,
            )
        )

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
        asyncio.run(self.areinforce(unit_id))

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
                results = await self._backend.search(
                    query_embedding=unit.embedding or [],
                    top_k=1,
                    filters=unit.scope,
                )
                found = any(r.id == uid for r in results)
                if not found and unit.embedding:
                    await self._backend.store(
                        id=unit.id,
                        embedding=unit.embedding,
                        metadata={"scope": unit.scope, "source": unit.source},
                    )
                    reinserted += 1
            except Exception:
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

    async def _state_path_loop(self) -> None:
        while self._running:
            try:
                await self._state_path_tick()
            except Exception:
                logger.exception("State path tick error")
            await asyncio.sleep(self._state_path_interval)

    async def _state_path_tick(self) -> None:
        await self._decay_tick()
        await self._expiry_archive()

        if self._conflict_queue is not None:
            try:
                await self._conflict_queue.drain(self._apply_conflict_resolution)
            except Exception:
                logger.exception("Conflict queue drain failed")

        self._tick_count += 1
        if self._tick_count % self._reconcile_every_n_ticks == 0:
            try:
                await self.areconcile()
            except Exception:
                logger.exception("Periodic reconciliation failed")
        if self._tick_count % self._consolidation_every_n_ticks == 0:
            try:
                await self._consolidation_tick()
            except Exception:
                logger.exception("Consolidation tick failed")
        if self._tick_count % self._wal_compact_every_n_ticks == 0:
            try:
                await self._wal_compact_tick()
            except Exception:
                logger.exception("WAL compaction tick failed")

    async def _consolidation_tick(self) -> None:
        """Merge semantically redundant units within each scope."""
        from contextidx.core.consolidation import find_redundant_pairs, merge_units

        units = await self._store.find_active_units()
        pairs = find_redundant_pairs(units)
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
        """Archive applied WAL entries older than 24 hours."""
        if self._wal is None:
            return
        removed = await self._wal.compact(retention_hours=24)
        if removed:
            logger.info("WAL compaction removed %d entries", removed)

    async def _decay_tick(self) -> None:
        """Recalculate decay scores for all active units."""
        units = await self._store.find_active_units()
        now = datetime.now(timezone.utc)
        for unit in units:
            state = await self._store.get_decay_state(unit.id)
            rc = state[2] if state else 0
            score = self._decay_engine.compute_decay(unit, now, rc)
            await self._store.upsert_decay_state(unit.id, score, now, rc)

    async def _expiry_archive(self) -> None:
        """Archive expired step-decay units."""
        units = await self._store.find_active_units()
        now = datetime.now(timezone.utc)
        for unit in units:
            if unit.is_expired_at(now):
                await self._store.update_unit(
                    unit.id, {"archived_at": now.isoformat()}
                )

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
                            await self._backend.store(
                                id=unit.id,
                                embedding=unit.embedding,
                                metadata={"scope": unit.scope, "source": unit.source},
                            )
                        await self._store.create_unit(unit)
                await self._wal.mark_applied(entry.seq)
            except Exception:
                logger.exception("WAL replay failed for seq=%d", entry.seq)
                await self._wal.mark_failed(entry.seq)

    # ── Graph Loading ──

    async def _load_graph(self) -> None:
        """Load all graph edges from the store into memory."""
        from contextidx.core.temporal_graph import Edge, Relationship

        units = await self._store.find_active_units()
        for unit in units:
            edges = await self._store.get_graph_edges(unit.id)
            for from_id, to_id, rel, created_at in edges:
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
