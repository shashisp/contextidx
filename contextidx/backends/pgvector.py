"""PGVector backend adapter for contextidx.

Requires optional dependencies:
    pip install contextidx[pgvector]
"""

from __future__ import annotations

import json
from typing import Any

from contextidx.backends.base import SearchResult, VectorBackend

try:
    import psycopg
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
except ImportError as exc:
    raise ImportError(
        "pgvector backend requires psycopg3. Install with: pip install contextidx[pgvector]"
    ) from exc

_TABLE = "contextidx_vectors"


class PGVectorBackend(VectorBackend):
    """PostgreSQL + pgvector backend with optional BM25 hybrid search.

    Stores embeddings in a ``vector`` column, metadata in ``jsonb``,
    and chunk text in ``content`` with a tsvector index for full-text search.
    """

    def __init__(
        self,
        conn_string: str,
        table_name: str = _TABLE,
        dimensions: int = 1536,
    ):
        self._conn_string = conn_string
        self._table = table_name
        self._dimensions = dimensions
        self._pool: AsyncConnectionPool | None = None

    @property
    def supports_metadata_store(self) -> bool:
        return True

    @property
    def supports_hybrid_search(self) -> bool:
        return True

    async def initialize(self) -> None:
        self._pool = AsyncConnectionPool(
            conninfo=self._conn_string,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
        await self._pool.open()
        async with self._pool.connection() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._table} (
                    id        TEXT PRIMARY KEY,
                    embedding vector({self._dimensions}),
                    metadata  JSONB DEFAULT '{{}}'::jsonb,
                    content   TEXT DEFAULT ''
                )
            """)
            # Add content column if upgrading from older schema
            await conn.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE {self._table} ADD COLUMN IF NOT EXISTS content TEXT DEFAULT '';
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)
            # tsvector generated column for full-text search
            await conn.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE {self._table}
                        ADD COLUMN IF NOT EXISTS search_vector tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED;
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$;
            """)
            await conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self._table}_search_vector
                ON {self._table} USING GIN (search_vector)
            """)
            await conn.commit()

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    def _get_pool(self) -> AsyncConnectionPool:
        if self._pool is None:
            raise RuntimeError("Backend not initialized. Call initialize() first.")
        return self._pool

    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        meta_json = json.dumps(metadata or {})
        vec_literal = _to_pg_vector(embedding)
        async with self._get_pool().connection() as conn:
            await conn.execute(
                f"""INSERT INTO {self._table} (id, embedding, metadata, content)
                    VALUES (%s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content""",
                (id, vec_literal, meta_json, _extract_content(metadata)),
            )
            await conn.commit()
        return id

    async def store_batch(
        self,
        items: list[tuple[str, list[float], dict | None]],
    ) -> list[str]:
        if not items:
            return []
        rows = [
            (id_, _to_pg_vector(emb), json.dumps(meta or {}), _extract_content(meta))
            for id_, emb, meta in items
        ]
        async with self._get_pool().connection() as conn:
            await conn.executemany(
                f"""INSERT INTO {self._table} (id, embedding, metadata, content)
                    VALUES (%s, %s::vector, %s::jsonb, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content""",
                rows,
            )
            await conn.commit()
        return [id_ for id_, _, _, _ in rows]


    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        vec_literal = _to_pg_vector(query_embedding)
        where_clause, params = _build_filter_clause(filters)
        query = f"""
            SELECT id, metadata,
                   1 - (embedding <=> %s::vector) AS score
            FROM {self._table}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [vec_literal, *params, vec_literal, top_k]

        async with self._get_pool().connection() as conn:
            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()

        return [
            SearchResult(
                id=r["id"],
                score=float(r["score"]),
                metadata=r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"] or "{}"),
            )
            for r in rows
        ]

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion of vector search + PostgreSQL full-text search."""
        vec_literal = _to_pg_vector(query_embedding)
        where_clause, filter_params = _build_filter_clause(filters)
        rrf_k = 60  # standard RRF constant

        # CTE-based RRF: rank each result in both vector and BM25 orderings,
        # then fuse scores with 1/(k+rank).
        sql = f"""
            WITH vector_ranked AS (
                SELECT id, metadata,
                       1 - (embedding <=> %s::vector) AS vec_score,
                       ROW_NUMBER() OVER (ORDER BY embedding <=> %s::vector) AS vec_rank
                FROM {self._table}
                {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            ),
            bm25_ranked AS (
                SELECT id, metadata,
                       ts_rank_cd(search_vector, plainto_tsquery('english', %s)) AS bm25_score,
                       ROW_NUMBER() OVER (
                           ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                       ) AS bm25_rank
                FROM {self._table}
                {where_clause}
                ORDER BY ts_rank_cd(search_vector, plainto_tsquery('english', %s)) DESC
                LIMIT %s
            ),
            fused AS (
                SELECT
                    COALESCE(v.id, b.id) AS id,
                    COALESCE(v.metadata, b.metadata) AS metadata,
                    COALESCE(v.vec_score, 0) AS vec_score,
                    COALESCE(b.bm25_score, 0) AS bm25_score,
                    COALESCE(1.0 / ({rrf_k} + v.vec_rank), 0) +
                    COALESCE(1.0 / ({rrf_k} + b.bm25_rank), 0) AS rrf_score
                FROM vector_ranked v
                FULL OUTER JOIN bm25_ranked b ON v.id = b.id
            )
            SELECT id, metadata, vec_score, bm25_score, rrf_score
            FROM fused
            ORDER BY rrf_score DESC
            LIMIT %s
        """

        fetch_k = top_k * 3
        # Build params: vector CTE needs (vec, vec, *filter, vec, fetch_k),
        # BM25 CTE needs (query_text, query_text, *filter, query_text, fetch_k),
        # final LIMIT is top_k
        params: list[Any] = [
            vec_literal, vec_literal, *filter_params, vec_literal, fetch_k,
            query, query, *filter_params, query, fetch_k,
            top_k,
        ]

        async with self._get_pool().connection() as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()

        return [
            SearchResult(
                id=r["id"],
                score=float(r["vec_score"]),
                metadata={
                    **(r["metadata"] if isinstance(r["metadata"], dict) else json.loads(r["metadata"] or "{}")),
                    "bm25_score": float(r["bm25_score"]),
                },
            )
            for r in rows
        ]

    async def delete(self, id: str) -> None:
        async with self._get_pool().connection() as conn:
            await conn.execute(f"DELETE FROM {self._table} WHERE id = %s", (id,))
            await conn.commit()

    async def update_metadata(self, id: str, metadata: dict) -> None:
        meta_json = json.dumps(metadata)
        async with self._get_pool().connection() as conn:
            await conn.execute(
                f"UPDATE {self._table} SET metadata = metadata || %s::jsonb WHERE id = %s",
                (meta_json, id),
            )
            await conn.commit()


def _to_pg_vector(embedding: list[float]) -> str:
    return "[" + ",".join(str(x) for x in embedding) + "]"


def _extract_content(metadata: dict | None) -> str:
    """Pull the source text from metadata for full-text indexing."""
    if not metadata:
        return ""
    # contextidx stores content in the ContextUnit, not in vector metadata.
    # The content is passed through when available.
    return metadata.get("_content", "")


def _build_filter_clause(
    filters: dict[str, Any] | None,
) -> tuple[str, list]:
    """Build a WHERE clause from scope filters applied to metadata JSONB.

    Scope keys are stored nested under ``metadata->'scope'`` by contextidx,
    so we query into the nested object.
    """
    if not filters:
        return "", []
    conditions: list[str] = []
    params: list = []
    for key, val in filters.items():
        conditions.append(f"metadata->'scope'->>%s = %s")
        params.extend([key, str(val)])
    return "WHERE " + " AND ".join(conditions), params
