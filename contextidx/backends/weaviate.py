"""Weaviate vector backend with hybrid search and multi-tenancy."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from contextidx.backends.base import SearchResult, VectorBackend

logger = logging.getLogger("contextidx.backends.weaviate")


def _hash_scope(scope: dict[str, str]) -> str:
    raw = json.dumps(scope, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class WeaviateBackend(VectorBackend):
    """Weaviate backend using the v4 async Python client.

    Requires ``weaviate-client>=4.7.0``.  Stores embeddings and metadata as
    native Weaviate object properties, supports hybrid (BM25 + vector) search,
    and uses multi-tenancy for scope isolation.
    """

    def __init__(
        self,
        *,
        collection_name: str = "ContextUnit",
        url: str | None = None,
        api_key: str | None = None,
        additional_headers: dict[str, str] | None = None,
        client: Any | None = None,
    ):
        self._collection_name = collection_name
        self._url = url
        self._api_key = api_key
        self._additional_headers = additional_headers or {}
        self._external_client = client
        self._client: Any | None = None
        self._collection: Any | None = None

    @property
    def supports_metadata_store(self) -> bool:
        return True

    @property
    def supports_hybrid_search(self) -> bool:
        return True

    def _scope_to_tenant(self, scope: dict[str, str] | None) -> str:
        """Map scope to Weaviate tenant name.

        Uses ``user_id`` as primary tenant key.  Falls back to a hash of the
        full scope dict.  A ``None`` / empty scope maps to a default tenant.
        """
        if not scope:
            return "default"
        if "user_id" in scope:
            return f"user_{scope['user_id']}"
        return f"scope_{_hash_scope(scope)}"

    async def initialize(self) -> None:
        try:
            import weaviate
            from weaviate.classes.config import (
                Configure,
                DataType,
                Property,
            )
            from weaviate.classes.tenants import Tenant
        except ImportError:
            raise ImportError(
                "weaviate-client>=4.7.0 is required for WeaviateBackend. "
                "Install with: pip install 'contextidx[weaviate]'"
            )

        if self._external_client is not None:
            self._client = self._external_client
        elif self._url:
            from weaviate.classes.init import Auth

            self._client = weaviate.use_async_with_custom(
                http_host=self._url.split("://")[-1].split(":")[0],
                http_port=int(self._url.split(":")[-1]) if ":" in self._url.rsplit("://", 1)[-1] else 8080,
                http_secure=self._url.startswith("https"),
                grpc_host=self._url.split("://")[-1].split(":")[0],
                grpc_port=50051,
                grpc_secure=self._url.startswith("https"),
                headers=self._additional_headers,
                auth_credentials=Auth.api_key(self._api_key) if self._api_key else None,
            )
        else:
            self._client = weaviate.use_async_with_local(
                headers=self._additional_headers,
            )

        if not self._client.is_connected():
            await self._client.connect()

        exists = await self._client.collections.exists(self._collection_name)
        if not exists:
            await self._client.collections.create(
                name=self._collection_name,
                multi_tenancy_config=Configure.multi_tenancy(
                    enabled=True,
                    auto_tenant_creation=True,
                ),
                properties=[
                    Property(name="unit_id", data_type=DataType.TEXT),
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="scope", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="confidence", data_type=DataType.NUMBER),
                    Property(name="decay_rate", data_type=DataType.NUMBER),
                    Property(name="decay_model", data_type=DataType.TEXT),
                    Property(name="version", data_type=DataType.INT),
                    Property(name="superseded_by", data_type=DataType.TEXT),
                    Property(name="created_at", data_type=DataType.TEXT),
                    Property(name="expires_at", data_type=DataType.TEXT),
                    Property(name="extra_metadata", data_type=DataType.TEXT),
                ],
            )
            logger.info("Created Weaviate collection %s", self._collection_name)

        self._collection = self._client.collections.use(self._collection_name)

    async def close(self) -> None:
        if self._client and self._external_client is None:
            await self._client.close()
            self._client = None

    def _get_tenant_collection(self, scope: dict[str, str] | None = None):
        tenant = self._scope_to_tenant(scope)
        return self._collection.with_tenant(tenant)

    async def store(
        self,
        id: str,
        embedding: list[float],
        metadata: dict | None = None,
    ) -> str:
        meta = metadata or {}
        scope = meta.get("scope", {})
        col = self._get_tenant_collection(scope)

        properties = {
            "unit_id": id,
            "content": meta.get("content", ""),
            "scope": json.dumps(scope) if isinstance(scope, dict) else str(scope),
            "source": meta.get("source", "unknown"),
            "confidence": meta.get("confidence", 0.8),
            "decay_rate": meta.get("decay_rate", 0.023),
            "decay_model": meta.get("decay_model", "exponential"),
            "version": meta.get("version", 1),
            "superseded_by": meta.get("superseded_by", ""),
            "created_at": meta.get("created_at", ""),
            "expires_at": meta.get("expires_at", ""),
            "extra_metadata": json.dumps(
                {k: v for k, v in meta.items() if k not in {
                    "scope", "source", "confidence", "decay_rate", "decay_model",
                    "version", "superseded_by", "created_at", "expires_at", "content",
                }}
            ),
        }

        uuid = await col.data.insert(
            properties=properties,
            vector=embedding,
        )
        return id

    async def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        col = self._get_tenant_collection(filters)

        from weaviate.classes.query import MetadataQuery

        response = await col.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True),
        )

        results = []
        for obj in response.objects:
            score = 1.0 - (obj.metadata.distance or 0.0)
            results.append(
                SearchResult(
                    id=obj.properties.get("unit_id", str(obj.uuid)),
                    score=max(0.0, min(1.0, score)),
                    metadata=_obj_to_meta(obj),
                )
            )
        return results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        filters: dict | None = None,
        alpha: float = 0.5,
    ) -> list[SearchResult]:
        col = self._get_tenant_collection(filters)

        from weaviate.classes.query import MetadataQuery

        response = await col.query.hybrid(
            query=query,
            vector=query_embedding,
            alpha=alpha,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True, score=True, explain_score=True),
        )

        results = []
        for obj in response.objects:
            combined_score = obj.metadata.score if obj.metadata.score is not None else 0.5
            meta = _obj_to_meta(obj)
            meta["bm25_score"] = min(1.0, max(0.0, combined_score))
            results.append(
                SearchResult(
                    id=obj.properties.get("unit_id", str(obj.uuid)),
                    score=max(0.0, min(1.0, combined_score)),
                    metadata=meta,
                )
            )
        return results

    async def delete(self, id: str) -> None:
        collections = [self._collection]
        tenants = await self._collection.tenants.get()
        for tenant_name in tenants:
            col = self._collection.with_tenant(tenant_name)
            from weaviate.classes.query import Filter

            response = await col.query.fetch_objects(
                filters=Filter.by_property("unit_id").equal(id),
                limit=1,
            )
            for obj in response.objects:
                await col.data.delete_by_id(obj.uuid)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        tenants = await self._collection.tenants.get()
        for tenant_name in tenants:
            col = self._collection.with_tenant(tenant_name)
            from weaviate.classes.query import Filter

            response = await col.query.fetch_objects(
                filters=Filter.by_property("unit_id").equal(id),
                limit=1,
            )
            for obj in response.objects:
                props: dict[str, Any] = {}
                for k, v in metadata.items():
                    if k in {"content", "scope", "source", "confidence", "decay_rate",
                             "decay_model", "version", "superseded_by", "created_at", "expires_at"}:
                        if k == "scope" and isinstance(v, dict):
                            props[k] = json.dumps(v)
                        else:
                            props[k] = v
                if props:
                    await col.data.update(uuid=obj.uuid, properties=props)


def _obj_to_meta(obj: Any) -> dict:
    """Extract a flat metadata dict from a Weaviate object."""
    props = obj.properties or {}
    meta: dict[str, Any] = {}
    for k, v in props.items():
        if k == "scope":
            try:
                meta[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                meta[k] = v
        elif k == "extra_metadata":
            try:
                meta.update(json.loads(v))
            except (json.JSONDecodeError, TypeError):
                pass
        else:
            meta[k] = v
    return meta
