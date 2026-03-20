"""Redis-backed pending buffer for multi-instance contextidx deployments.

Drop-in replacement for the in-memory ``PendingBuffer``. Uses Redis sorted
sets keyed by scope hash, with timestamps as scores for TTL-based expiry.

Requires optional dependency: pip install contextidx[redis]
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from contextidx.core.context_unit import ContextUnit

try:
    import redis.asyncio as aioredis
except ImportError as exc:
    raise ImportError(
        "RedisPendingBuffer requires redis>=5.0.0. "
        "Install with: pip install contextidx[redis]"
    ) from exc


def _hash_scope(scope: dict[str, str]) -> str:
    canonical = json.dumps(scope, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _serialize_unit(unit: ContextUnit) -> str:
    return json.dumps({
        "id": unit.id,
        "content": unit.content,
        "embedding": unit.embedding,
        "scope": unit.scope,
        "confidence": unit.confidence,
        "decay_rate": unit.decay_rate,
        "decay_model": unit.decay_model,
        "version": unit.version,
        "source": unit.source,
        "superseded_by": unit.superseded_by,
        "timestamp": unit.timestamp.isoformat(),
        "expires_at": unit.expires_at.isoformat() if unit.expires_at else None,
    })


def _deserialize_unit(raw: str | bytes) -> ContextUnit:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return ContextUnit(
        id=data["id"],
        content=data["content"],
        embedding=data.get("embedding"),
        scope=data["scope"],
        confidence=data["confidence"],
        decay_rate=data["decay_rate"],
        decay_model=data["decay_model"],
        version=data["version"],
        source=data["source"],
        superseded_by=data.get("superseded_by"),
        timestamp=datetime.fromisoformat(data["timestamp"]),
        expires_at=datetime.fromisoformat(data["expires_at"]) if data.get("expires_at") else None,
    )


_KEY_PREFIX = "ctxidx:pending:"


class RedisPendingBuffer:
    """Redis-backed scoped buffer for read-after-write consistency.

    Same public interface as ``PendingBuffer`` but backed by Redis, enabling
    multiple contextidx instances to share pending state.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        ttl_seconds: int = 30,
        max_units_per_scope: int = 50,
    ):
        self._ttl = ttl_seconds
        self._max = max_units_per_scope
        self._redis: aioredis.Redis = aioredis.from_url(redis_url, decode_responses=False)

    def _key(self, scope: dict[str, str]) -> str:
        return f"{_KEY_PREFIX}{_hash_scope(scope)}"

    async def add(self, unit: ContextUnit) -> None:
        key = self._key(unit.scope)
        now = datetime.now(timezone.utc).timestamp()
        payload = _serialize_unit(unit)
        pipe = self._redis.pipeline()
        pipe.zadd(key, {payload: now})
        pipe.expire(key, self._ttl * 2)
        await pipe.execute()

        count = await self._redis.zcard(key)
        if count > self._max:
            await self._redis.zpopmin(key, count - self._max)

    async def get(self, scope: dict[str, str]) -> list[ContextUnit]:
        key = self._key(scope)
        cutoff = datetime.now(timezone.utc).timestamp() - self._ttl
        await self._redis.zremrangebyscore(key, "-inf", cutoff)
        members = await self._redis.zrangebyscore(key, cutoff, "+inf")
        return [_deserialize_unit(m) for m in members]

    async def remove(self, unit_id: str) -> None:
        async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            members = await self._redis.zrange(key, 0, -1)
            for m in members:
                data = json.loads(m if isinstance(m, str) else m.decode("utf-8"))
                if data.get("id") == unit_id:
                    await self._redis.zrem(key, m)
                    return

    async def flush_expired(self) -> list[ContextUnit]:
        expired: list[ContextUnit] = []
        cutoff = datetime.now(timezone.utc).timestamp() - self._ttl
        async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            members = await self._redis.zrangebyscore(key, "-inf", cutoff)
            for m in members:
                expired.append(_deserialize_unit(m))
            if members:
                await self._redis.zremrangebyscore(key, "-inf", cutoff)
        return expired

    async def clear(self) -> None:
        async for key in self._redis.scan_iter(match=f"{_KEY_PREFIX}*"):
            await self._redis.delete(key)

    async def close(self) -> None:
        await self._redis.aclose()
