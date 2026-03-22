#!/usr/bin/env python3
"""Scoring weight ablation study for contextidx.

Runs a grid search over scoring weight configurations and measures accuracy
on a local LoCoMo-style JSON dataset to find the optimal weight vector.

Usage
-----
    python scripts/ablation_study.py --data path/to/locomo.json

    # Restrict the search space for a quicker run:
    python scripts/ablation_study.py --data locomo.json --max-convs 10

    # Save results to a JSON file:
    python scripts/ablation_study.py --data locomo.json --out ablation.json

Output
------
Prints a ranked table of weight configurations + accuracy to stdout.
Writes full results (including per-type accuracy) to --out if specified.

How it works
------------
1. Ingest conversations into an in-memory vector backend (no Postgres needed).
2. For each weight configuration, run retrieval + answer generation with the
   given weights applied to ScoringEngine.
3. Compute F1 against gold answers; report overall and per-type accuracy.

The grid covers six signals (semantic, bm25, recency, confidence, decay,
reinforcement).  Because exhaustive 6D search is expensive, the grid uses a
coarse step (0.2) over a curated set of "driver" weights that are varied one
at a time while keeping the others proportionally balanced.

Requirements
------------
    pip install contextidx openai tqdm
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

from contextidx.backends.base import SearchResult, VectorBackend
from contextidx.config import ContextIdxConfig
from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.environ.get("ABLATION_EMBEDDING_MODEL", "text-embedding-3-small")
_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _client


async def _embed_text(text: str) -> list[float]:
    resp = await _get_client().embeddings.create(input=text, model=EMBEDDING_MODEL)
    return resp.data[0].embedding


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def _f1(pred: str, gold: str) -> float:
    p = set(_normalize(pred).split())
    g = set(_normalize(gold).split())
    if not p or not g:
        return float(p == g)
    common = p & g
    if not common:
        return 0.0
    prec = len(common) / len(p)
    rec = len(common) / len(g)
    return 2 * prec * rec / (prec + rec)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── Lightweight in-memory backend (no external DB) ────────────────────────────

class _InMemoryBackend(VectorBackend):
    def __init__(self) -> None:
        self._store: dict[str, tuple[list[float], dict]] = {}

    @property
    def supports_metadata_store(self) -> bool:
        return False

    async def store(self, id: str, embedding: list[float], metadata: dict | None = None) -> str:
        self._store[id] = (embedding, metadata or {})
        return id

    async def search(self, query_embedding: list[float], top_k: int, filters: dict | None = None) -> list[SearchResult]:
        results = []
        for vid, (emb, meta) in self._store.items():
            if filters:
                scope = meta.get("scope", {})
                if not all(scope.get(k) == v for k, v in filters.items()):
                    continue
            results.append(SearchResult(id=vid, score=_cosine(query_embedding, emb), metadata=meta))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    async def delete(self, id: str) -> None:
        self._store.pop(id, None)

    async def update_metadata(self, id: str, metadata: dict) -> None:
        if id in self._store:
            emb, existing = self._store[id]
            existing.update(metadata)
            self._store[id] = (emb, existing)

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def clear(self) -> None:
        self._store.clear()


# ── Weight grid generation ────────────────────────────────────────────────────

def _generate_weight_configs() -> list[dict[str, float]]:
    """Generate a curated ablation grid.

    Varies semantic and recency weights over [0.1, 0.3, 0.5] while keeping
    the remainder split equally among other signals.  This produces 9 configs
    — enough to surface directional trends without exhaustive search.
    """
    configs = []
    signals = ["semantic", "bm25", "recency", "confidence", "decay", "reinforcement"]
    semantic_values = [0.15, 0.30, 0.45]
    recency_values = [0.10, 0.25, 0.40]

    for sem in semantic_values:
        for rec in recency_values:
            remainder = max(0.0, 1.0 - sem - rec)
            per_other = remainder / 4.0
            cfg = {
                "semantic": sem,
                "bm25": per_other,
                "recency": rec,
                "confidence": per_other,
                "decay": per_other,
                "reinforcement": per_other,
            }
            configs.append(cfg)

    return configs


# ── Ingestion ─────────────────────────────────────────────────────────────────

def _chunk_session(messages: list[dict], window: int, stride: int) -> list[str]:
    lines = [f"{m.get('role','')}: {m.get('content','')}" for m in messages]
    if len(lines) <= window:
        return ["\n".join(lines)]
    chunks = []
    for start in range(0, len(lines), stride):
        chunks.append("\n".join(lines[start : start + window]))
        if start + window >= len(lines):
            break
    return chunks


async def _ingest(
    idx: ContextIdx,
    convs: list[dict],
    scope_prefix: str,
    window: int,
    stride: int,
) -> None:
    for conv_i, conv in enumerate(convs):
        sessions = conv.get("sessions", [conv])
        scope = {"container": f"{scope_prefix}_conv{conv_i}"}
        for sess in sessions:
            msgs = sess.get("messages", [])
            for chunk in _chunk_session(msgs, window, stride):
                emb = await _embed_text(chunk)
                await idx.astore(content=chunk, scope=scope, embedding=emb)


# ── Answer generation ─────────────────────────────────────────────────────────

async def _answer(
    idx: ContextIdx,
    question: str,
    scope: dict,
    top_k: int,
) -> str:
    q_emb = await _embed_text(question)
    results = await idx.aretrieve(
        query=question,
        scope=scope,
        top_k=top_k,
        query_embedding=q_emb,
    )
    context = "\n\n---\n\n".join(r.content for r in results)
    resp = await _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Answer based ONLY on context.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
        }],
        max_tokens=100,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ── Main evaluation loop ──────────────────────────────────────────────────────

@dataclass
class _Run:
    weights: dict[str, float]
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

    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    def avg_f1(self) -> float:
        return self.f1_sum / self.total if self.total else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "accuracy": round(self.accuracy(), 4),
            "avg_f1": round(self.avg_f1(), 4),
            "by_type": {
                qt: round(sum(1 for s in sc if s >= 0.5) / len(sc), 4)
                for qt, sc in self.by_type.items()
            },
        }


async def run(args: argparse.Namespace) -> None:
    data = json.loads(Path(args.data).read_text())
    convs: list[dict] = data if isinstance(data, list) else data.get("conversations", [])
    if args.max_convs:
        convs = convs[: args.max_convs]

    qa_pairs: list[tuple[str, str, str, int]] = []
    for conv_i, conv in enumerate(convs):
        for qa in conv.get("qa_pairs", conv.get("questions", [])):
            q = qa.get("question", qa.get("q", ""))
            a = qa.get("answer", qa.get("a", ""))
            qt = qa.get("type", qa.get("category", "factual"))
            if q and a:
                qa_pairs.append((q, a, qt, conv_i))

    weight_configs = _generate_weight_configs()
    print(
        f"Running {len(weight_configs)} configurations × {len(qa_pairs)} questions …",
        file=sys.stderr,
    )

    results: list[_Run] = []

    for config_i, weights in enumerate(weight_configs):
        print(f"\n[{config_i + 1}/{len(weight_configs)}] weights={weights}", file=sys.stderr)

        run_result = _Run(weights=weights)
        backend = _InMemoryBackend()
        cfg = ContextIdxConfig(scoring_weights=weights)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = SQLiteStore(path=f"{tmp}/ablation.db")
            async with ContextIdx(
                backend=backend,
                internal_store=store,
                config=cfg,
                conflict_detection="rule_based",
                half_life_days=90,
                embedding_fn=None,
                openai_api_key=OPENAI_API_KEY,
            ) as idx:
                await _ingest(idx, convs, scope_prefix=f"cfg{config_i}", window=args.window_size, stride=args.stride)

                for question, gold, q_type, conv_i in tqdm(qa_pairs, desc=f"cfg{config_i}", file=sys.stderr):
                    scope = {"container": f"cfg{config_i}_conv{conv_i}"}
                    pred = await _answer(idx, question, scope, args.top_k)
                    run_result.record(pred, gold, q_type)

        results.append(run_result)

    # Sort by accuracy descending
    results.sort(key=lambda r: r.accuracy(), reverse=True)

    print("\n" + "=" * 70)
    print(f"{'Rank':<5} {'Semantic':<10} {'Recency':<10} {'Acc':<8} {'F1':<8}")
    print("-" * 70)
    for rank, r in enumerate(results, 1):
        w = r.weights
        print(
            f"{rank:<5} {w['semantic']:<10.2f} {w['recency']:<10.2f} "
            f"{r.accuracy():<8.3f} {r.avg_f1():<8.3f}"
        )
    print("=" * 70)
    print(f"\nBest config: {results[0].weights}")
    print(f"Best accuracy: {results[0].accuracy():.1%}")

    if args.out:
        output = [r.summary() for r in results]
        Path(args.out).write_text(json.dumps(output, indent=2))
        print(f"\nFull results written to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Scoring weight ablation study")
    parser.add_argument("--data", required=True, help="Path to LoCoMo-style JSON")
    parser.add_argument("--max-convs", type=int, default=0, help="Limit conversations (0=all)")
    parser.add_argument("--top-k", type=int, default=5, help="Results per query")
    parser.add_argument("--window-size", type=int, default=8, help="Chunk window size")
    parser.add_argument("--stride", type=int, default=3, help="Chunk stride")
    parser.add_argument("--out", default="", help="Output JSON path for full results")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        sys.exit("OPENAI_API_KEY environment variable required")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
