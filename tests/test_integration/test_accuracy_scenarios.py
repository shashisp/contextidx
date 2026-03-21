"""Synthetic accuracy tests for the three core failure modes.

These test the scenarios from the accuracy benchmark plan:
- Scenario A: Stale retrieval (decay should filter old goals)
- Scenario B: Supersession (temporal graph should exclude old titles)
- Scenario C: Contradiction accumulation (conflict resolution should pick latest)
- Recency bias: adaptive decay threshold filters stale items relative to fresh ones

Each test uses embeddings designed so that items about the same topic have
high cosine similarity (>0.90), simulating what real embedding models produce.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.contextidx import ContextIdx
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 64


def _topic_embedding(topic_seed: int, variant: float = 0.0) -> list[float]:
    """Generate an embedding for a topic with a small variant offset.

    Items with the same topic_seed but different variants will have
    high cosine similarity (>0.95), simulating real embeddings for
    statements about the same attribute/topic.
    """
    base = [math.sin((topic_seed + 1) * (i + 1) * 0.1) for i in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in base))
    base = [x / norm for x in base]
    if variant != 0.0:
        perturb = [math.cos((variant + 1) * (i + 1) * 0.3) * 0.05 for i in range(_DIM)]
        mixed = [b + p for b, p in zip(base, perturb)]
        norm = math.sqrt(sum(x * x for x in mixed))
        mixed = [x / norm for x in mixed]
        return mixed
    return base


TOPIC_FITNESS = 1
TOPIC_JOB_TITLE = 2
TOPIC_COMM_PREF = 3
TOPIC_QUERY_FITNESS = 1
TOPIC_QUERY_JOB = 2
TOPIC_QUERY_COMM = 3


class TestScenarioA_StaleRetrieval:
    """Scenario A: Stale retrieval via decay.

    Session 1 (t-90d): "training for weight loss"
    Session 2 (t-30d): "training for muscle building"
    Query (t-0):       "current fitness goal?"
    Expected:          only "muscle building" returned
    """

    async def test_only_recent_goal_returned(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "scenario_a.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            half_life_days=30,
            decay_threshold=0.01,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_old = _topic_embedding(TOPIC_FITNESS, variant=1.0)
            id_old = await idx.astore(
                content="I'm training for a marathon, focusing on weight loss",
                scope={"user_id": "u1"},
                source="session_1",
                embedding=emb_old,
            )
            await idx._store.update_unit(
                id_old, {"created_at": (now - timedelta(days=90)).isoformat()}
            )

            emb_new = _topic_embedding(TOPIC_FITNESS, variant=2.0)
            id_new = await idx.astore(
                content="Switched goals — now training for muscle building",
                scope={"user_id": "u1"},
                source="session_2",
                embedding=emb_new,
            )
            await idx._store.update_unit(
                id_new, {"created_at": (now - timedelta(days=30)).isoformat()}
            )

            q_emb = _topic_embedding(TOPIC_QUERY_FITNESS, variant=0.0)
            results = await idx.aretrieve(
                query="What is the user's current fitness goal?",
                scope={"user_id": "u1"},
                top_k=5,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("muscle building" in c for c in contents), (
                f"Expected 'muscle building' in results, got: {contents}"
            )
            assert not any("weight loss" in c for c in contents), (
                f"Expected 'weight loss' to be filtered out, got: {contents}"
            )


class TestScenarioB_Supersession:
    """Scenario B: Supersession via TemporalGraph.

    Session 1 (t-180d): "got promoted to junior engineer"
    Session 3 (t-60d):  "got promoted — senior engineer now"
    Query (t-0):        "current job title?"
    Expected:           only "senior engineer" returned
    """

    async def test_only_current_title_returned(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "scenario_b.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            half_life_days=90,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_junior = _topic_embedding(TOPIC_JOB_TITLE, variant=1.0)
            id_junior = await idx.astore(
                content="Just got promoted to junior engineer",
                scope={"user_id": "u1"},
                source="session_1",
                embedding=emb_junior,
            )
            await idx._store.update_unit(
                id_junior, {"created_at": (now - timedelta(days=180)).isoformat()}
            )

            emb_senior = _topic_embedding(TOPIC_JOB_TITLE, variant=2.0)
            id_senior = await idx.astore(
                content="Got promoted again — senior engineer now",
                scope={"user_id": "u1"},
                source="session_3",
                embedding=emb_senior,
            )
            await idx._store.update_unit(
                id_senior, {"created_at": (now - timedelta(days=60)).isoformat()}
            )

            q_emb = _topic_embedding(TOPIC_QUERY_JOB, variant=0.0)
            results = await idx.aretrieve(
                query="What is the user's current job title?",
                scope={"user_id": "u1"},
                top_k=5,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("senior engineer" in c.lower() for c in contents), (
                f"Expected 'senior engineer' in results, got: {contents}"
            )
            assert not any(
                "junior engineer" in c.lower()
                for c in contents
                if "senior" not in c.lower()
            ), (
                f"Expected 'junior engineer' to be superseded, got: {contents}"
            )


class TestScenarioC_Contradiction:
    """Scenario C: Contradiction accumulation via ConflictResolver.

    Session 2 (t-120d): "I prefer formal communication"
    Session 5 (t-14d):  "I prefer casual communication, drop the formality"
    Query (t-0):        "How should the agent address this user?"
    Expected:           only "casual" returned
    """

    async def test_only_latest_preference_returned(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "scenario_c.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            half_life_days=60,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_formal = _topic_embedding(TOPIC_COMM_PREF, variant=1.0)
            id_formal = await idx.astore(
                content="I prefer formal communication",
                scope={"user_id": "u1"},
                source="session_2",
                embedding=emb_formal,
            )
            await idx._store.update_unit(
                id_formal, {"created_at": (now - timedelta(days=120)).isoformat()}
            )

            emb_casual = _topic_embedding(TOPIC_COMM_PREF, variant=2.0)
            id_casual = await idx.astore(
                content="I prefer casual communication, please drop the formality",
                scope={"user_id": "u1"},
                source="session_5",
                embedding=emb_casual,
            )
            await idx._store.update_unit(
                id_casual, {"created_at": (now - timedelta(days=14)).isoformat()}
            )

            q_emb = _topic_embedding(TOPIC_QUERY_COMM, variant=0.0)
            results = await idx.aretrieve(
                query="How should the agent address this user?",
                scope={"user_id": "u1"},
                top_k=5,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("casual" in c.lower() for c in contents), (
                f"Expected 'casual' in results, got: {contents}"
            )
            assert not any(
                "formal" in c.lower()
                for c in contents
                if "casual" not in c.lower()
            ), (
                f"Expected 'formal' preference to be superseded, got: {contents}"
            )


TOPIC_FOOD = 10
TOPIC_MUSIC = 11
TOPIC_PET = 12
TOPIC_HOBBY = 13


class TestLLMConflictDetection:
    """conflict_detection='llm' uses a judge function for highest accuracy."""

    async def test_llm_mode_uses_judge(self, tmp_path):
        """The judge confirms semantic candidates, so supersession happens."""
        calls: list[tuple[str, str]] = []

        async def judge(a: str, b: str) -> bool:
            calls.append((a, b))
            return True

        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "llm_mode.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="llm",
            conflict_judge_fn=judge,
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            emb_old = _topic_embedding(TOPIC_HOBBY, variant=1.0)
            await idx.astore(
                content="I enjoy painting watercolors",
                scope={"user_id": "u1"},
                embedding=emb_old,
            )

            emb_new = _topic_embedding(TOPIC_HOBBY, variant=2.0)
            await idx.astore(
                content="I switched from painting to sculpture",
                scope={"user_id": "u1"},
                embedding=emb_new,
            )

            assert len(calls) >= 1, "Judge should have been called"

            q_emb = _topic_embedding(TOPIC_HOBBY, variant=0.0)
            results = await idx.aretrieve(
                query="hobby",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )
            contents = [r.content for r in results]
            assert any("sculpture" in c for c in contents)
            assert not any(
                "watercolors" in c for c in contents if "sculpture" not in c
            )

    async def test_llm_mode_judge_rejects(self, tmp_path):
        """If the judge says 'no conflict', both units survive."""
        async def judge(a: str, b: str) -> bool:
            return False

        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "llm_reject.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="llm",
            conflict_judge_fn=judge,
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            emb_a = _topic_embedding(TOPIC_HOBBY, variant=1.0)
            await idx.astore(
                content="I enjoy painting watercolors",
                scope={"user_id": "u1"},
                embedding=emb_a,
            )

            emb_b = _topic_embedding(TOPIC_HOBBY, variant=2.0)
            await idx.astore(
                content="I also do sculpture on weekends",
                scope={"user_id": "u1"},
                embedding=emb_b,
            )

            q_emb = _topic_embedding(TOPIC_HOBBY, variant=0.0)
            results = await idx.aretrieve(
                query="hobbies",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )
            assert len(results) == 2, (
                f"Judge said no conflict, both should survive. Got: {[r.content for r in results]}"
            )

    async def test_llm_mode_falls_back_without_judge(self, tmp_path):
        """Without a judge, 'llm' mode falls back to pure semantic detection."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "llm_fallback.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="llm",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            emb_a = _topic_embedding(TOPIC_HOBBY, variant=1.0)
            await idx.astore(
                content="I enjoy painting watercolors",
                scope={"user_id": "u1"},
                embedding=emb_a,
            )

            emb_b = _topic_embedding(TOPIC_HOBBY, variant=2.0)
            await idx.astore(
                content="I switched from painting to sculpture",
                scope={"user_id": "u1"},
                embedding=emb_b,
            )

            q_emb = _topic_embedding(TOPIC_HOBBY, variant=0.0)
            results = await idx.aretrieve(
                query="hobby",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )
            assert len(results) == 1, (
                f"Without judge, semantic fallback should supersede. Got: {[r.content for r in results]}"
            )


class TestExplicitSupersede:
    """asupersede() lets the application explicitly mark one unit as superseded."""

    async def test_superseded_unit_excluded_from_retrieval(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "supersede.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            emb_a = _topic_embedding(TOPIC_PET, variant=1.0)
            id_a = await idx.astore(
                content="I have a cat named Whiskers",
                scope={"user_id": "u1"},
                embedding=emb_a,
            )

            emb_b = _topic_embedding(TOPIC_PET, variant=2.0)
            id_b = await idx.astore(
                content="Whiskers passed away, I now have a dog named Buddy",
                scope={"user_id": "u1"},
                embedding=emb_b,
            )

            q_emb = _topic_embedding(TOPIC_PET, variant=0.0)
            before = await idx.aretrieve(
                query="What pet does the user have?",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )
            before_contents = [r.content for r in before]
            assert any("Whiskers" in c for c in before_contents) or any(
                "Buddy" in c for c in before_contents
            )

            await idx.asupersede(id_b, id_a)

            after = await idx.aretrieve(
                query="What pet does the user have?",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )
            after_contents = [r.content for r in after]
            assert any("Buddy" in c for c in after_contents), (
                f"Expected 'Buddy' in results, got: {after_contents}"
            )
            assert not any(
                "Whiskers" in c
                for c in after_contents
                if "Buddy" not in c
            ), (
                f"Expected old cat unit to be superseded, got: {after_contents}"
            )

    async def test_supersede_missing_unit_raises(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "supersede_err.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            with pytest.raises(ValueError, match="not found"):
                await idx.asupersede("nonexistent_new", "nonexistent_old")


class TestRecencyBias:
    """Recency bias filters items whose decay score is much lower than the best candidate."""

    async def test_recency_bias_filters_stale_items(self, tmp_path):
        """Store two items on *different* topics (no semantic overlap) with
        very different ages.  Without recency_bias both survive; with it the
        ancient one is dropped because its decay score is tiny relative to
        the fresh one."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "recency.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            half_life_days=30,
            decay_threshold=0.001,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_old = _topic_embedding(TOPIC_FOOD, variant=1.0)
            id_old = await idx.astore(
                content="Loves sushi and ramen",
                scope={"user_id": "u1"},
                embedding=emb_old,
            )
            await idx._store.update_unit(
                id_old, {"created_at": (now - timedelta(days=300)).isoformat()}
            )

            emb_new = _topic_embedding(TOPIC_MUSIC, variant=1.0)
            id_new = await idx.astore(
                content="Currently listening to jazz",
                scope={"user_id": "u1"},
                embedding=emb_new,
            )

            both_emb = _topic_embedding(TOPIC_FOOD, variant=0.0)

            all_results = await idx.aretrieve(
                query="anything about user",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=both_emb,
                recency_bias=0.0,
            )
            assert len(all_results) >= 1

            biased_results = await idx.aretrieve(
                query="anything about user",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=both_emb,
                recency_bias=0.25,
            )
            biased_contents = [r.content for r in biased_results]
            assert not any("sushi" in c.lower() for c in biased_contents), (
                f"Expected 300-day-old item to be filtered by recency_bias, got: {biased_contents}"
            )

    async def test_constructor_level_recency_bias(self, tmp_path):
        """Verify that the constructor-level recency_bias applies by default."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "recency2.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="semantic",
            half_life_days=30,
            decay_threshold=0.001,
            recency_bias=0.25,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_old = _topic_embedding(TOPIC_FOOD, variant=1.0)
            id_old = await idx.astore(
                content="Loves sushi and ramen",
                scope={"user_id": "u1"},
                embedding=emb_old,
            )
            await idx._store.update_unit(
                id_old, {"created_at": (now - timedelta(days=300)).isoformat()}
            )

            emb_new = _topic_embedding(TOPIC_MUSIC, variant=1.0)
            await idx.astore(
                content="Currently listening to jazz",
                scope={"user_id": "u1"},
                embedding=emb_new,
            )

            both_emb = _topic_embedding(TOPIC_FOOD, variant=0.0)
            results = await idx.aretrieve(
                query="anything about user",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=both_emb,
            )
            contents = [r.content for r in results]
            assert not any("sushi" in c.lower() for c in contents), (
                f"Constructor recency_bias should filter stale item, got: {contents}"
            )
