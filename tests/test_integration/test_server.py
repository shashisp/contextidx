"""Integration tests for the contextidx FastAPI server.

Tests the full HTTP lifecycle: health -> ingest -> search -> clear,
using a custom embedding function to avoid real OpenAI calls.
"""

from __future__ import annotations

import asyncio
import math
import os
import time
from multiprocessing import Process

import httpx
import pytest


_DIM = 1536


def _deterministic_embedding(text: str) -> list[float]:
    """Produce a deterministic embedding from text content."""
    h = hash(text) & 0xFFFFFFFF
    raw = [math.sin((h + i) * 0.01) for i in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw]


def _start_server(port: int, store_path: str):
    """Start the server in a subprocess with a patched embedder."""
    import uvicorn

    os.environ["OPENAI_API_KEY"] = "test-not-used"
    os.environ["CONTEXTIDX_STORE_PATH"] = store_path
    os.environ["CONTEXTIDX_BACKEND"] = "memory"

    from contextidx.server import _build_app, _create_idx
    import contextidx.server as server_mod

    async def _patched_create_idx():
        from contextidx.contextidx import ContextIdx
        from contextidx.store.sqlite_store import SQLiteStore
        from tests.conftest import InMemoryVectorBackend

        class FakeEmbedder:
            async def embed(self, text: str) -> list[float]:
                return _deterministic_embedding(text)

            async def embed_batch(self, texts: list[str]) -> list[list[float]]:
                return [_deterministic_embedding(t) for t in texts]

        os.makedirs(os.path.dirname(store_path) or ".", exist_ok=True)
        store = SQLiteStore(path=store_path)
        backend = InMemoryVectorBackend()
        idx = ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            embedding_fn=FakeEmbedder(),
        )
        await idx.ainitialize()
        return idx

    server_mod._create_idx = _patched_create_idx  # type: ignore[attr-defined]
    app = _build_app()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


class TestServerEndpoints:
    """Full HTTP lifecycle test of the contextidx server."""

    @pytest.fixture(autouse=True)
    def server(self, tmp_path):
        port = 18741
        store_path = str(tmp_path / "test_server.db")
        proc = Process(target=_start_server, args=(port, store_path), daemon=True)
        proc.start()

        base = f"http://127.0.0.1:{port}"
        for _ in range(40):
            try:
                resp = httpx.get(f"{base}/health", timeout=1.0)
                if resp.status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(0.25)
        else:
            proc.kill()
            pytest.fail("Server did not start in time")

        self.base = base
        yield
        proc.kill()
        proc.join(timeout=3)

    def test_health(self):
        resp = httpx.get(f"{self.base}/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_ingest_search_clear_lifecycle(self):
        resp = httpx.post(
            f"{self.base}/ingest",
            json={
                "sessions": [
                    {
                        "sessionId": "s1",
                        "messages": [
                            {"role": "user", "content": "I love hiking in the mountains"},
                            {"role": "assistant", "content": "Great!"},
                            {"role": "user", "content": "Especially the Alps"},
                        ],
                        "metadata": {"date": "2024-06-15"},
                    },
                    {
                        "sessionId": "s2",
                        "messages": [
                            {"role": "user", "content": "I started learning piano last week"},
                        ],
                    },
                ],
                "containerTag": "question_1",
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        ingest_data = resp.json()
        assert len(ingest_data["documentIds"]) == 2

        resp = httpx.post(
            f"{self.base}/search",
            json={
                "query": "What outdoor activities does the user enjoy?",
                "containerTag": "question_1",
                "limit": 5,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        search_data = resp.json()
        assert len(search_data["results"]) >= 1
        contents = [r["content"] for r in search_data["results"]]
        assert any("hiking" in c.lower() or "alps" in c.lower() for c in contents)

        resp = httpx.delete(f"{self.base}/clear/question_1", timeout=10.0)
        assert resp.status_code == 200
        clear_data = resp.json()
        assert clear_data["deleted"] >= 2

        resp = httpx.post(
            f"{self.base}/search",
            json={
                "query": "What outdoor activities does the user enjoy?",
                "containerTag": "question_1",
                "limit": 5,
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        assert len(resp.json()["results"]) == 0

    def test_ingest_skips_assistant_only_sessions(self):
        resp = httpx.post(
            f"{self.base}/ingest",
            json={
                "sessions": [
                    {
                        "sessionId": "s_asst",
                        "messages": [
                            {"role": "assistant", "content": "I can help with that!"},
                        ],
                    },
                ],
                "containerTag": "question_2",
            },
            timeout=10.0,
        )
        assert resp.status_code == 200
        assert len(resp.json()["documentIds"]) == 0

    def test_clear_nonexistent_container(self):
        resp = httpx.delete(f"{self.base}/clear/nonexistent_tag", timeout=10.0)
        assert resp.status_code == 200
        assert resp.json()["deleted"] == 0
