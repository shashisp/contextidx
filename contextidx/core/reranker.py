"""Re-ranking protocol and built-in OpenAI implementation.

Define a custom reranker by implementing :class:`RerankerFn` and passing it
to ``ContextIdx(reranker=...)``.  Example using a local cross-encoder::

    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def my_reranker(query, candidates, top_k):
        texts = [(query, u.content) for u, _, _ in candidates[: top_k * 2]]
        scores = model.predict(texts)
        reranked = [
            (u, float(s), d)
            for (u, _, d), s in zip(candidates[: top_k * 2], scores)
        ]
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked + candidates[top_k * 2 :]

    idx = ContextIdx(backend=..., reranker=my_reranker)
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from contextidx.core.context_unit import ContextUnit

logger = logging.getLogger("contextidx.reranker")


@runtime_checkable
class RerankerFn(Protocol):
    """Protocol for pluggable re-ranking functions.

    A re-ranker receives the *query*, the full *candidates* list (already
    sorted by composite score), and the desired *top_k*.  It returns the
    candidates in a new order (same tuples, potentially new scores).

    Implementations must be async; wrap sync models with ``asyncio.to_thread``.
    """

    async def __call__(
        self,
        query: str,
        candidates: list[tuple[ContextUnit, float, float]],
        top_k: int,
    ) -> list[tuple[ContextUnit, float, float]]: ...


class OpenAIReranker:
    """Default LLM re-ranker using the OpenAI chat completions API.

    Scores the top ``2 * top_k`` candidates with *model* on a 0-10 relevance
    scale, then blends LLM score with the original composite score using
    ``blend`` (default 0.6 → 60 % LLM, 40 % composite).

    Falls back to the original ordering on any error.

    Parameters
    ----------
    model:
        OpenAI chat model to use (default ``gpt-4o-mini``).
    blend:
        Weight given to the LLM score when combining with the composite score.
        Must be in [0, 1].  0 → pure composite, 1 → pure LLM.
    timeout:
        Seconds to wait for the LLM response before falling back.
        ``None`` means no timeout.
    api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        blend: float = 0.6,
        timeout: float | None = None,
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._blend = blend
        self._timeout = timeout
        self._api_key = api_key
        self._client: object | None = None

    def _ensure_client(self) -> None:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError as exc:
                raise ImportError(
                    "openai package is required for OpenAIReranker. "
                    "Install with: pip install openai"
                ) from exc

    async def __call__(
        self,
        query: str,
        candidates: list[tuple[ContextUnit, float, float]],
        top_k: int,
    ) -> list[tuple[ContextUnit, float, float]]:
        rerank_pool = candidates[: top_k * 2]
        if len(rerank_pool) <= 1:
            return candidates

        self._ensure_client()

        numbered = "\n".join(
            f"[{i}] {u.content[:300]}" for i, (u, _, _) in enumerate(rerank_pool)
        )
        coro = self._client.chat.completions.create(  # type: ignore[union-attr]
            model=self._model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance judge. Given a query and numbered text "
                        "passages, rate each passage's relevance to the query on a "
                        "scale of 0-10. Output ONLY a JSON array of objects with "
                        "'index' and 'score' keys. "
                        'Example: [{"index":0,"score":8},{"index":1,"score":3}]'
                    ),
                },
                {
                    "role": "user",
                    "content": f"Query: {query}\n\nPassages:\n{numbered}",
                },
            ],
        )

        if self._timeout is not None:
            resp = await asyncio.wait_for(coro, timeout=self._timeout)
        else:
            resp = await coro

        raw = resp.choices[0].message.content or "[]"
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        rankings = json.loads(raw)
        score_map: dict[int, float] = {
            r["index"]: float(r["score"]) for r in rankings
        }

        reranked: list[tuple[ContextUnit, float, float]] = []
        for i, (unit, composite, decay) in enumerate(rerank_pool):
            llm_score = score_map.get(i, 5.0) / 10.0
            blended = (1.0 - self._blend) * composite + self._blend * llm_score
            reranked.append((unit, blended, decay))
        reranked.sort(key=lambda x: x[1], reverse=True)

        return reranked + candidates[top_k * 2 :]
