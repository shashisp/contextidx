"""Integration tests for write batching: BatchWriter and astore_batch."""

from __future__ import annotations

import asyncio
import math

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.utils.batch_writer import BatchWriter


def _normalized(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 0 else vec


class TestBatchWriter:
    async def test_batch_writer_flush(self):
        """Items are flushed and futures resolve to unit IDs."""
        stored = []

        async def mock_store(content, scope, embedding=None, **kwargs):
            uid = f"unit_{len(stored)}"
            stored.append((content, scope, embedding))
            return uid

        async def mock_embed_batch(texts):
            return [_normalized([float(i), 1.0, 0.0]) for i in range(len(texts))]

        writer = BatchWriter(
            store_fn=mock_store,
            embed_batch_fn=mock_embed_batch,
            batch_size=3,
            flush_interval=0,
        )

        f1 = await writer.add("alpha", {"user_id": "u1"})
        f2 = await writer.add("beta", {"user_id": "u1"})
        await writer.flush()

        assert f1.done()
        assert f2.done()
        assert f1.result() == "unit_0"
        assert f2.result() == "unit_1"
        assert len(stored) == 2

    async def test_auto_flush_on_batch_size(self):
        """Buffer auto-flushes when reaching batch_size."""
        stored = []

        async def mock_store(content, scope, embedding=None, **kwargs):
            uid = f"unit_{len(stored)}"
            stored.append(content)
            return uid

        async def mock_embed_batch(texts):
            return [[1.0, 0.0] for _ in texts]

        writer = BatchWriter(
            store_fn=mock_store,
            embed_batch_fn=mock_embed_batch,
            batch_size=2,
            flush_interval=0,
        )

        f1 = await writer.add("a", {"user_id": "u1"})
        f2 = await writer.add("b", {"user_id": "u1"})
        assert f1.done()
        assert f2.done()
        assert len(stored) == 2

    async def test_embed_failure_propagates(self):
        """If embed_batch fails, futures receive the exception."""
        async def mock_store(**kwargs):
            return "id"

        async def mock_embed_batch(texts):
            raise RuntimeError("API down")

        writer = BatchWriter(
            store_fn=mock_store,
            embed_batch_fn=mock_embed_batch,
            batch_size=10,
            flush_interval=0,
        )

        f1 = await writer.add("test", {"user_id": "u1"})
        await writer.flush()

        assert f1.done()
        with pytest.raises(RuntimeError, match="API down"):
            f1.result()

    async def test_pending_count(self):
        async def mock_store(**kwargs):
            return "id"

        async def mock_embed_batch(texts):
            return [[0.0] for _ in texts]

        writer = BatchWriter(
            store_fn=mock_store,
            embed_batch_fn=mock_embed_batch,
            batch_size=100,
            flush_interval=0,
        )

        await writer.add("a", {"user_id": "u1"})
        await writer.add("b", {"user_id": "u1"})
        assert writer.pending_count == 2
        await writer.flush()
        assert writer.pending_count == 0


class TestAstoreBatch:
    @pytest.fixture
    async def ctx(self, mock_backend, tmp_path):
        ctx = ContextIdx(
            backend=mock_backend,
            internal_store_path=str(tmp_path / "meta.db"),
            state_path_interval=9999,
        )
        await ctx.ainitialize()
        yield ctx
        await ctx.aclose()

    async def test_astore_batch_returns_ids(self, ctx):
        """astore_batch should store items and return their IDs."""
        emb1 = _normalized([1.0, 0.0, 0.0, 0.0])
        emb2 = _normalized([0.0, 1.0, 0.0, 0.0])

        ctx._embedder.embed_batch = self._mock_embed_batch([emb1, emb2])

        ids = await ctx.astore_batch([
            {"content": "first item", "scope": {"user_id": "u1"}},
            {"content": "second item", "scope": {"user_id": "u1"}},
        ])
        assert len(ids) == 2
        assert all(isinstance(uid, str) for uid in ids)

        for uid in ids:
            unit = await ctx._store.get_unit(uid)
            assert unit is not None

    @staticmethod
    def _mock_embed_batch(embeddings):
        async def embed_batch(texts):
            return embeddings[:len(texts)]
        return embed_batch
