"""Micro-batch writer that amortizes embedding and backend write costs.

Accumulates ``(content, scope, kwargs)`` tuples and flushes them in
batches, calling ``EmbeddingProvider.embed_batch()`` once per flush
instead of embedding one item at a time.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

logger = logging.getLogger("contextidx.batch_writer")


@dataclass
class _PendingItem:
    content: str
    scope: dict[str, str]
    kwargs: dict[str, Any]
    future: asyncio.Future


class BatchWriter:
    """Buffers writes into micro-batches.

    Parameters:
        store_fn: Async callable ``(content, scope, **kwargs) -> str`` that
            persists a single item.  Typically ``ContextIdx.astore``.
        batch_size: Maximum number of items to buffer before auto-flushing.
        flush_interval: Seconds between automatic flushes (0 to disable).
    """

    def __init__(
        self,
        store_fn: Callable[..., Coroutine[Any, Any, str]],
        embed_batch_fn: Callable[..., Coroutine[Any, Any, list[list[float]]]],
        *,
        batch_size: int = 10,
        flush_interval: float = 0.5,
    ):
        self._store_fn = store_fn
        self._embed_batch_fn = embed_batch_fn
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._buffer: list[_PendingItem] = []
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start the auto-flush timer."""
        if self._running:
            return
        self._running = True
        if self._flush_interval > 0:
            self._timer_task = asyncio.create_task(self._timer_loop())

    async def stop(self) -> None:
        """Flush remaining items and stop the timer."""
        self._running = False
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            try:
                await self._timer_task
            except asyncio.CancelledError:
                pass
        await self.flush()

    async def add(
        self,
        content: str,
        scope: dict[str, str],
        **kwargs,
    ) -> asyncio.Future:
        """Add an item to the buffer.  Returns a future that resolves to the unit ID."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        async with self._lock:
            self._buffer.append(_PendingItem(
                content=content, scope=scope, kwargs=kwargs, future=future,
            ))
            if len(self._buffer) >= self._batch_size:
                await self._flush_locked()
        return future

    async def flush(self) -> None:
        """Flush all buffered items."""
        async with self._lock:
            await self._flush_locked()

    async def _flush_locked(self) -> None:
        """Flush while already holding the lock."""
        if not self._buffer:
            return

        batch = list(self._buffer)
        self._buffer.clear()

        texts = [item.content for item in batch]
        try:
            embeddings = await self._embed_batch_fn(texts)
        except Exception as exc:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(exc)
            return

        for item, embedding in zip(batch, embeddings):
            try:
                unit_id = await self._store_fn(
                    content=item.content,
                    scope=item.scope,
                    embedding=embedding,
                    **item.kwargs,
                )
                if not item.future.done():
                    item.future.set_result(unit_id)
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)

        logger.debug("Flushed batch of %d items", len(batch))

    async def _timer_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self._flush_interval)
            try:
                await self.flush()
            except Exception:
                logger.exception("Auto-flush failed")

    @property
    def pending_count(self) -> int:
        return len(self._buffer)
