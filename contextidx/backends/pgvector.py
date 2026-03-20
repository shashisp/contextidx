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
    """PostgreSQL + pgvector backend.

    Stores embeddings in a ``vector`` column and metadata in a ``jsonb`` column.
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
                    metadata  JSONB DEFAULT '{{}}'::jsonb
                )
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
                f"""INSERT INTO {self._table} (id, embedding, metadata)
                    VALUES (%s, %s::vector, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata""",
                (id, vec_literal, meta_json),
            )
            await conn.commit()
        return id

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


def _build_filter_clause(
    filters: dict[str, Any] | None,
) -> tuple[str, list]:
    """Build a WHERE clause from scope filters applied to metadata JSONB."""
    if not filters:
        return "", []
    conditions: list[str] = []
    params: list = []
    for key, val in filters.items():
        conditions.append(f"metadata->>%s = %s")
        params.extend([key, str(val)])
    return "WHERE " + " AND ".join(conditions), params
