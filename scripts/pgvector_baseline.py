#!/usr/bin/env python3
"""Raw pgvector baseline for LoCoMo accuracy comparison.

Ingests LoCoMo conversations into pgvector *without* any temporal processing
(no decay, no supersession, no conflict detection) and runs the same QA
evaluation as the main contextidx benchmark.  The delta between this score
and the contextidx score shows how much value the temporal layer adds.

Usage
-----
    # In-memory (default) — for quick local runs without a real Postgres:
    python scripts/pgvector_baseline.py --data path/to/locomo.json

    # Against a real PostgreSQL + pgvector instance:
    python scripts/pgvector_baseline.py \\
        --dsn "postgresql://user:pass@localhost/bench" \\
        --data path/to/locomo.json

    # Limit number of conversations for a fast smoke test:
    python scripts/pgvector_baseline.py --data locomo.json --max-convs 5

Output
------
Prints accuracy metrics (overall, single-hop, multi-hop, temporal) to stdout
in the same format as run_memorybench.sh so results can be compared side-by-side.

Requirements
------------
    pip install contextidx[pgvector] openai tqdm
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import openai
except ImportError:
    sys.exit("openai package required: pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs):  # type: ignore[misc]
        return it

# ── helpers ───────────────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("BASELINE_EMBEDDING_MODEL", "text-embedding-3-small")
_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def _embed(text: str) -> list[float]:
    resp = await _get_client().embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p_toks = set(_normalize_answer(pred).split())
    g_toks = set(_normalize_answer(gold).split())
    if not p_toks or not g_toks:
        return float(p_toks == g_toks)
    common = p_toks & g_toks
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


# ── in-memory vector store (no Postgres dependency) ───────────────────────────

@dataclass
class _InMemoryStore:
    _items: list[tuple[str, list[float]]] = field(default_factory=list)

    def insert(self, text: str, embedding: list[float]) -> None:
        self._items.append((text, embedding))

    def search(self, query_emb: list[float], top_k: int) -> list[str]:
        scored = [
            (_cosine_similarity(query_emb, emb), text)
            for text, emb in self._items
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [text for _, text in scored[:top_k]]


# ── pgvector store (optional) ─────────────────────────────────────────────────

class _PGVectorStore:
    """Thin async wrapper around pgvector for the baseline."""

    def __init__(self, dsn: str, dim: int = 1536) -> None:
        self._dsn = dsn
        self._dim = dim
        self._pool = None

    async def initialize(self) -> None:
        try:
            import asyncpg
        except ImportError:
            sys.exit("asyncpg required for pgvector baseline: pip install asyncpg")
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        async with self._pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS baseline_vectors (
                    id SERIAL PRIMARY KEY,
                    content TEXT,
                    embedding vector({self._dim})
                )
            """)
            await conn.execute(
                "TRUNCATE baseline_vectors"
            )

    async def insert(self, text: str, embedding: list[float]) -> None:
        vec = "[" + ",".join(str(x) for x in embedding) + "]"
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO baseline_vectors (content, embedding) VALUES ($1, $2::vector)",
                text, vec,
            )

    async def search(self, query_emb: list[float], top_k: int) -> list[str]:
        vec = "[" + ",".join(str(x) for x in query_emb) + "]"
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content FROM baseline_vectors "
                "ORDER BY embedding <=> $1::vector LIMIT $2",
                vec, top_k,
            )
        return [r["content"] for r in rows]

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()


# ── chunking (mirrors server.py logic) ────────────────────────────────────────

def _chunk_session(messages: list[dict], window: int, stride: int) -> list[str]:
    lines = [
        f"{m.get('role', 'unknown')}: {m.get('content', '')}"
        for m in messages
    ]
    if len(lines) <= window:
        return ["\n".join(lines)]
    chunks = []
    for start in range(0, len(lines), stride):
        chunk = lines[start : start + window]
        chunks.append("\n".join(chunk))
        if start + window >= len(lines):
            break
    return chunks


# ── evaluation ────────────────────────────────────────────────────────────────

async def _answer_question(
    store: _InMemoryStore | _PGVectorStore,
    question: str,
    top_k: int,
) -> str:
    q_emb = await _embed(question)
    chunks = await store.search(q_emb, top_k) if isinstance(store, _PGVectorStore) else store.search(q_emb, top_k)  # type: ignore[arg-type]
    context = "\n\n---\n\n".join(chunks)
    prompt = (
        f"Answer the question based ONLY on the following context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer (be concise):"
    )
    resp = await _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


@dataclass
class _Metrics:
    total: int = 0
    correct: int = 0
    f1_sum: float = 0.0
    by_type: dict[str, list[float]] = field(default_factory=dict)

    def record(self, pred: str, gold: str, q_type: str) -> None:
        self.total += 1
        f1 = _f1(pred, gold)
        self.f1_sum += f1
        if f1 >= 0.5:
            self.correct += 1
        self.by_type.setdefault(q_type, []).append(f1)

    def report(self) -> dict[str, Any]:
        overall_acc = self.correct / self.total if self.total else 0.0
        overall_f1 = self.f1_sum / self.total if self.total else 0.0
        by_type_acc = {
            qt: sum(1 for s in scores if s >= 0.5) / len(scores)
            for qt, scores in self.by_type.items()
        }
        return {
            "total_questions": self.total,
            "overall_accuracy": round(overall_acc, 4),
            "overall_f1": round(overall_f1, 4),
            "by_type": {k: round(v, 4) for k, v in by_type_acc.items()},
        }


async def run(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.data).read_text())
    convs: list[dict] = data if isinstance(data, list) else data.get("conversations", [])
    if args.max_convs:
        convs = convs[: args.max_convs]

    # Build store
    if args.dsn:
        store: _InMemoryStore | _PGVectorStore = _PGVectorStore(args.dsn)
        await store.initialize()  # type: ignore[union-attr]
    else:
        store = _InMemoryStore()

    # Ingest
    print(f"Ingesting {len(convs)} conversations …", file=sys.stderr)
    for conv in tqdm(convs, desc="ingest", file=sys.stderr):
        sessions = conv.get("sessions", [conv])
        for sess in sessions:
            msgs = sess.get("messages", [])
            chunks = _chunk_session(msgs, args.window_size, args.stride)
            for chunk in chunks:
                emb = await _embed(chunk)
                if isinstance(store, _PGVectorStore):
                    await store.insert(chunk, emb)
                else:
                    store.insert(chunk, emb)

    # Evaluate
    metrics = _Metrics()
    qa_pairs: list[tuple[str, str, str]] = []
    for conv in convs:
        for qa in conv.get("qa_pairs", conv.get("questions", [])):
            question = qa.get("question", qa.get("q", ""))
            answer = qa.get("answer", qa.get("a", ""))
            q_type = qa.get("type", qa.get("category", "factual"))
            if question and answer:
                qa_pairs.append((question, answer, q_type))

    print(f"Evaluating {len(qa_pairs)} questions …", file=sys.stderr)
    for question, answer, q_type in tqdm(qa_pairs, desc="eval", file=sys.stderr):
        pred = await _answer_question(store, question, args.top_k)
        metrics.record(pred, answer, q_type)

    if isinstance(store, _PGVectorStore):
        await store.close()  # type: ignore[union-attr]

    report = metrics.report()
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw pgvector baseline for LoCoMo")
    parser.add_argument("--data", required=True, help="Path to LoCoMo JSON file")
    parser.add_argument("--dsn", default="", help="PostgreSQL DSN (omit for in-memory)")
    parser.add_argument("--max-convs", type=int, default=0, help="Limit conversations (0=all)")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks retrieved per question")
    parser.add_argument("--window-size", type=int, default=8, help="Chunk window size")
    parser.add_argument("--stride", type=int, default=3, help="Chunk stride")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY environment variable required")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
