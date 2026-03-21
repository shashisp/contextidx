"""FastAPI HTTP wrapper for contextidx.

Exposes the ContextIdx API over REST for integration with external
benchmarking frameworks like memorybench.

Usage::

    uvicorn contextidx.server:app --host 0.0.0.0 --port 8741

Configuration via environment variables:

    OPENAI_API_KEY            Required for embeddings
    CONTEXTIDX_HALF_LIFE      Half-life in days (default: 30)
    CONTEXTIDX_DETECTION      Conflict detection mode (default: semantic)
    CONTEXTIDX_STRATEGY       Conflict strategy (default: LAST_WRITE_WINS)
    CONTEXTIDX_STORE_PATH     SQLite store path (default: .contextidx/memorybench.db)
    CONTEXTIDX_BACKEND        "memory" or a pgvector DSN (default: memory)
    CONTEXTIDX_RECENCY_BIAS   Recency bias 0-1 (default: none)
    CONTEXTIDX_WINDOW_SIZE    Chunk window size in lines (default: 8)
    CONTEXTIDX_STRIDE         Chunk stride in lines (default: 3)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dateutil_parser
from pydantic import BaseModel, Field

logger = logging.getLogger("contextidx.server")


# ── Request / Response models ────────────────────────────────────────────────

class MessageIn(BaseModel):
    role: str
    content: str
    timestamp: str | None = None
    speaker: str | None = None


class SessionIn(BaseModel):
    sessionId: str
    messages: list[MessageIn]
    metadata: dict[str, Any] | None = None


class IngestRequest(BaseModel):
    sessions: list[SessionIn]
    containerTag: str
    metadata: dict[str, Any] | None = None


class IngestResponse(BaseModel):
    documentIds: list[str]


class SearchRequest(BaseModel):
    query: str
    containerTag: str
    limit: int = 10
    threshold: float | None = None
    rerank: bool = True


class SearchResultOut(BaseModel):
    content: str
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    results: list[SearchResultOut]


class ClearResponse(BaseModel):
    deleted: int


class HealthResponse(BaseModel):
    status: str
    version: str


# ── Application factory ──────────────────────────────────────────────────────

def _build_app():
    from fastapi import FastAPI, HTTPException

    _idx = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal _idx
        _idx = await _create_idx()
        logger.info("contextidx server ready")
        yield
        if _idx is not None:
            await _idx.aclose()
            _idx = None
            logger.info("contextidx server shut down")

    app = FastAPI(
        title="contextidx",
        description="Temporal context layer -- memorybench provider API",
        version="1.0.0",
        lifespan=lifespan,
    )

    def _get_idx():
        if _idx is None:
            raise HTTPException(status_code=503, detail="contextidx not initialized")
        return _idx

    # ── Endpoints ─────────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="ok", version="1.0.0")

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(req: IngestRequest):
        idx = _get_idx()
        document_ids: list[str] = []
        scope = {"container": req.containerTag}

        WINDOW_SIZE = int(os.environ.get("CONTEXTIDX_WINDOW_SIZE", "8"))
        STRIDE = int(os.environ.get("CONTEXTIDX_STRIDE", "3"))

        items: list[dict] = []
        # Track how many chunks each session produces so we can link them
        session_chunk_counts: list[int] = []

        for session in req.sessions:
            if not session.messages:
                continue

            meta = session.metadata or {}
            formatted_date = meta.get("formattedDate") or meta.get("date") or ""
            source = f"session:{session.sessionId}"

            ts: datetime | None = None
            raw_date = meta.get("date")
            if raw_date:
                try:
                    ts = dateutil_parser.parse(raw_date)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    pass

            all_lines: list[str] = []
            for m in session.messages:
                speaker = m.speaker or m.role
                all_lines.append(f"{speaker}: {m.content}")

            date_header = f"[Session date: {formatted_date}]" if formatted_date else ""
            session_items_start = len(items)

            if len(all_lines) <= WINDOW_SIZE:
                content = (date_header + "\n" + "\n".join(all_lines)).strip()
                item: dict[str, Any] = {
                    "content": content,
                    "scope": scope,
                    "source": source,
                }
                if ts is not None:
                    item["timestamp"] = ts
                items.append(item)
            else:
                chunk_idx = 0
                for start in range(0, len(all_lines), STRIDE):
                    window = all_lines[start : start + WINDOW_SIZE]
                    if not window:
                        break
                    content = (date_header + "\n" + "\n".join(window)).strip()
                    item = {
                        "content": content,
                        "scope": scope,
                        "source": f"{source}:chunk-{chunk_idx}",
                    }
                    if ts is not None:
                        item["timestamp"] = ts
                    items.append(item)
                    chunk_idx += 1
                    if start + WINDOW_SIZE >= len(all_lines):
                        break

            session_chunk_counts.append(len(items) - session_items_start)

        if items:
            ids = await idx.astore_batch(items)
            document_ids.extend(ids)

            # Link consecutive chunks within each session via RELATES_TO edges
            offset = 0
            for count in session_chunk_counts:
                session_ids = ids[offset : offset + count]
                for i in range(len(session_ids) - 1):
                    await idx.alink_related(session_ids[i], session_ids[i + 1])
                offset += count

        return IngestResponse(documentIds=document_ids)

    @app.post("/search", response_model=SearchResponse)
    async def search(req: SearchRequest):
        idx = _get_idx()
        scope = {"container": req.containerTag}

        units = await idx.aretrieve(
            query=req.query,
            scope=scope,
            top_k=req.limit,
            min_score=0.0,
            rerank=req.rerank,
        )

        results = [
            SearchResultOut(
                content=u.content,
                score=u.confidence,
                metadata={
                    "source": u.source,
                    "timestamp": u.timestamp.isoformat(),
                    "scope": u.scope,
                },
            )
            for u in units
        ]
        return SearchResponse(results=results)

    @app.delete("/clear/{container_tag}", response_model=ClearResponse)
    async def clear(container_tag: str):
        idx = _get_idx()
        scope = {"container": container_tag}
        deleted = await idx.aclear(scope)
        return ClearResponse(deleted=deleted)

    return app


async def _create_idx():
    from contextidx.contextidx import ContextIdx
    from contextidx.store.sqlite_store import SQLiteStore

    backend_cfg = os.environ.get("CONTEXTIDX_BACKEND", "memory")
    half_life = float(os.environ.get("CONTEXTIDX_HALF_LIFE", "30"))
    detection = os.environ.get("CONTEXTIDX_DETECTION", "rule_based")
    strategy = os.environ.get("CONTEXTIDX_STRATEGY", "LAST_WRITE_WINS")
    store_path = os.environ.get("CONTEXTIDX_STORE_PATH", ".contextidx/memorybench.db")
    recency_raw = os.environ.get("CONTEXTIDX_RECENCY_BIAS")
    recency_bias = float(recency_raw) if recency_raw else None

    store: SQLiteStore | None = None

    if backend_cfg == "memory":
        from tests.conftest import InMemoryVectorBackend
        backend = InMemoryVectorBackend()
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        store = SQLiteStore(path=store_path)
    elif backend_cfg.startswith("postgresql://"):
        from contextidx.backends.pgvector import PGVectorBackend
        from contextidx.store.postgres_store import PostgresStore
        backend = PGVectorBackend(conn_string=backend_cfg)
        store = PostgresStore(dsn=backend_cfg)
    else:
        from tests.conftest import InMemoryVectorBackend
        backend = InMemoryVectorBackend()
        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        store = SQLiteStore(path=store_path)
        logger.warning("Unknown CONTEXTIDX_BACKEND=%r, falling back to memory", backend_cfg)

    idx = ContextIdx(
        backend=backend,
        internal_store=store,
        half_life_days=half_life,
        conflict_detection=detection,  # type: ignore[arg-type]
        conflict_strategy=strategy,  # type: ignore[arg-type]
        recency_bias=recency_bias,
        decay_threshold=0.0,
    )
    await idx.ainitialize()
    return idx


app = _build_app()
