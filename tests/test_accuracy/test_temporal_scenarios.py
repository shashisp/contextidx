"""Temporal scenario tests for contextidx accuracy.

Covers scenarios not in test_accuracy_scenarios.py:
- Multi-hop supersession chains (A → B → C)
- Expiry filtering (TTL-expired units excluded)
- Mixed temporal states (active, superseded, expired in same scope)
- Graph expansion with similarity threshold
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from contextidx.config import ContextIdxConfig
from contextidx.contextidx import ContextIdx
from contextidx.core.temporal_graph import Edge, Relationship
from contextidx.store.sqlite_store import SQLiteStore
from tests.conftest import InMemoryVectorBackend

_DIM = 64


def _neg_embedding(seed: int, variant: float = 0.0) -> list[float]:
    """Return the negation of _unit_embedding — guaranteed near-last in cosine ranking."""
    return [-x for x in _unit_embedding(seed, variant)]


def _unit_embedding(seed: int, variant: float = 0.0) -> list[float]:
    """Normalised embedding for a topic seed with an optional small variant."""
    base = [math.sin((seed + 1) * (i + 1) * 0.1) for i in range(_DIM)]
    norm = math.sqrt(sum(x * x for x in base))
    base = [x / norm for x in base]
    if variant != 0.0:
        perturb = [math.cos((variant + 1) * (i + 1) * 0.3) * 0.05 for i in range(_DIM)]
        mixed = [b + p for b, p in zip(base, perturb)]
        norm = math.sqrt(sum(x * x for x in mixed))
        mixed = [x / norm for x in mixed]
        return mixed
    return base


# Stable topic seeds
T_ROLE = 10
T_LOCATION = 20
T_SKILL = 30
T_HEALTH = 40


class TestSupersessionChain:
    """Multi-hop supersession: A → B → C.

    Only the final (most recent) unit should be returned; all predecessors
    must be excluded because they carry superseded_by values.
    """

    async def test_three_hop_chain_returns_only_latest(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "chain.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_a = _unit_embedding(T_ROLE, variant=1.0)
            id_a = await idx.astore(
                content="Intern at Acme",
                scope={"user_id": "u1"},
                embedding=emb_a,
            )
            await idx._store.update_unit(
                id_a, {"created_at": (now - timedelta(days=365)).isoformat()}
            )

            emb_b = _unit_embedding(T_ROLE, variant=2.0)
            id_b = await idx.astore(
                content="Junior engineer at Acme",
                scope={"user_id": "u1"},
                embedding=emb_b,
            )
            await idx._store.update_unit(
                id_b, {"created_at": (now - timedelta(days=180)).isoformat()}
            )
            # B supersedes A
            await idx.asupersede(id_b, id_a)

            emb_c = _unit_embedding(T_ROLE, variant=3.0)
            id_c = await idx.astore(
                content="Senior engineer at Acme",
                scope={"user_id": "u1"},
                embedding=emb_c,
            )
            await idx._store.update_unit(
                id_c, {"created_at": (now - timedelta(days=30)).isoformat()}
            )
            # C supersedes B
            await idx.asupersede(id_c, id_b)

            q_emb = _unit_embedding(T_ROLE, variant=0.0)
            results = await idx.aretrieve(
                query="current role",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("Senior" in c for c in contents), (
                f"Expected 'Senior engineer' in results, got: {contents}"
            )
            assert not any("Intern" in c for c in contents), (
                f"Expected 'Intern' to be superseded, got: {contents}"
            )
            assert not any(
                "Junior" in c for c in contents if "Senior" not in c
            ), (
                f"Expected 'Junior engineer' to be superseded, got: {contents}"
            )

    async def test_supersession_chain_unit_ids_reflect_lineage(self, tmp_path):
        """Store verifies the superseded_by field is set on predecessor units."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "chain_meta.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            id_old = await idx.astore(
                content="Python developer",
                scope={"user_id": "u1"},
                embedding=_unit_embedding(T_SKILL, variant=1.0),
            )
            id_new = await idx.astore(
                content="Python + Rust developer",
                scope={"user_id": "u1"},
                embedding=_unit_embedding(T_SKILL, variant=2.0),
            )
            await idx.asupersede(id_new, id_old)

            old_unit = await idx._store.get_unit(id_old)
            assert old_unit is not None
            assert old_unit.superseded_by == id_new, (
                f"Expected superseded_by={id_new}, got {old_unit.superseded_by}"
            )

            new_unit = await idx._store.get_unit(id_new)
            assert new_unit is not None
            assert new_unit.superseded_by is None


class TestExpiryFiltering:
    """Units with expires_at in the past must not appear in retrieval."""

    async def test_expired_unit_excluded(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "expiry.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)

            emb_expired = _unit_embedding(T_HEALTH, variant=1.0)
            await idx.astore(
                content="On a low-carb diet until end of January",
                scope={"user_id": "u1"},
                embedding=emb_expired,
                expires_at=now - timedelta(days=10),
            )

            emb_active = _unit_embedding(T_HEALTH, variant=2.0)
            await idx.astore(
                content="Now following a Mediterranean diet",
                scope={"user_id": "u1"},
                embedding=emb_active,
                expires_at=now + timedelta(days=90),
            )

            q_emb = _unit_embedding(T_HEALTH, variant=0.0)
            results = await idx.aretrieve(
                query="current diet",
                scope={"user_id": "u1"},
                top_k=10,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("Mediterranean" in c for c in contents), (
                f"Expected active diet in results, got: {contents}"
            )
            assert not any("low-carb" in c.lower() for c in contents), (
                f"Expected expired diet to be excluded, got: {contents}"
            )

    async def test_not_yet_expired_unit_returned(self, tmp_path):
        """Unit with expires_at in the future is still active."""
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "not_expired.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)
            emb = _unit_embedding(T_LOCATION, variant=1.0)
            await idx.astore(
                content="Working remotely from Portugal",
                scope={"user_id": "u1"},
                embedding=emb,
                expires_at=now + timedelta(days=30),
            )

            q_emb = _unit_embedding(T_LOCATION, variant=0.0)
            results = await idx.aretrieve(
                query="current location",
                scope={"user_id": "u1"},
                top_k=5,
                query_embedding=q_emb,
            )
            assert any("Portugal" in r.content for r in results), (
                f"Future-expiry unit should still be returned. Got: {[r.content for r in results]}"
            )


class TestMixedTemporalStates:
    """Scope containing active, superseded, and expired units simultaneously.

    Only active (not superseded, not expired) units should be returned.
    """

    async def test_only_active_units_returned(self, tmp_path):
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "mixed.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
        ) as idx:
            now = datetime.now(timezone.utc)
            scope = {"user_id": "u1"}

            # Unit A — will be superseded
            id_superseded = await idx.astore(
                content="Old company: TechCorp",
                scope=scope,
                embedding=_unit_embedding(T_ROLE, variant=1.0),
            )

            # Unit B — active (supersedes A)
            id_active = await idx.astore(
                content="Current company: StartupXYZ",
                scope=scope,
                embedding=_unit_embedding(T_ROLE, variant=2.0),
            )
            await idx.asupersede(id_active, id_superseded)

            # Unit C — expired
            await idx.astore(
                content="Visiting Berlin for a conference",
                scope=scope,
                embedding=_unit_embedding(T_LOCATION, variant=1.0),
                expires_at=now - timedelta(days=1),
            )

            q_emb = _unit_embedding(T_ROLE, variant=0.0)
            results = await idx.aretrieve(
                query="work and travel",
                scope=scope,
                top_k=10,
                query_embedding=q_emb,
            )

            contents = [r.content for r in results]
            assert any("StartupXYZ" in c for c in contents), (
                f"Active unit should appear, got: {contents}"
            )
            assert not any("TechCorp" in c for c in contents), (
                f"Superseded unit must not appear, got: {contents}"
            )
            assert not any("Berlin" in c for c in contents), (
                f"Expired unit must not appear, got: {contents}"
            )


class TestGraphExpansionMinScore:
    """graph_expansion_min_score filters loosely-related graph neighbours."""

    async def test_min_score_zero_includes_low_similarity_neighbours(self, tmp_path):
        """With min_score=0.0 (default), even dissimilar graph neighbours appear."""
        cfg = ContextIdxConfig(graph_expansion_min_score=0.0, overfetch_factor=3)
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "gs_low.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="rule_based",
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
            config=cfg,
        ) as idx:
            scope = {"user_id": "u1"}

            # Anchor — high similarity to query
            id_anchor = await idx.astore(
                content="Alice is a software engineer",
                scope=scope,
                embedding=_unit_embedding(T_ROLE, variant=1.0),
            )
            # Neighbour — very different topic; will be connected via graph edge
            id_neighbour = await idx.astore(
                content="Alice lives in Helsinki",
                scope=scope,
                embedding=_unit_embedding(T_LOCATION, variant=1.0),
            )

            # Manually add a RELATES_TO edge
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            await idx._store.add_graph_edge(
                id_anchor, id_neighbour, "RELATES_TO",
                now,
            )
            idx._graph.load_edges([Edge(id_anchor, id_neighbour, Relationship.RELATES_TO, now)])

            q_emb = _unit_embedding(T_ROLE, variant=0.0)
            results = await idx.aretrieve(
                query="Tell me about Alice",
                scope=scope,
                top_k=10,
                query_embedding=q_emb,
            )
            contents = [r.content for r in results]
            # Both should be present (neighbour via graph expansion, no min_score filter)
            assert any("software engineer" in c for c in contents)
            assert any("Helsinki" in c for c in contents), (
                f"With min_score=0.0 neighbour should be included via graph, got: {contents}"
            )

    async def test_high_min_score_excludes_dissimilar_neighbours(self, tmp_path):
        """With a high min_score, dissimilar graph neighbours are excluded.

        Strategy: use distinct-topic embeddings for each stored unit to avoid
        conflict detection superseding them, then connect the anchor to the
        dissimilar neighbour via a graph edge.  With min_score=0.9 the
        dissimilar neighbour is filtered during graph expansion.
        """
        # fetch_k = top_k(1) * overfetch(2) = 2
        # 3 units stored: anchor (sim~0.96), decoy (sim~0.96), tacos (sim~0.005)
        # tacos ranks #3, so it is NOT in the initial vector search pool.
        # It can only enter via graph expansion — where min_score=0.9 blocks it.
        cfg = ContextIdxConfig(graph_expansion_min_score=0.9, overfetch_factor=2)
        backend = InMemoryVectorBackend()
        store = SQLiteStore(path=tmp_path / "gs_high.db")

        async with ContextIdx(
            backend=backend,
            internal_store=store,
            conflict_detection="none",  # disable to avoid superseding similar units
            half_life_days=365,
            embedding_fn=None,
            openai_api_key="test",
            config=cfg,
        ) as idx:
            scope = {"user_id": "u1"}

            # Anchor — query-topic (T_SKILL)
            id_anchor = await idx.astore(
                content="Bob is a data scientist",
                scope=scope,
                embedding=_unit_embedding(T_SKILL, variant=1.0),
            )
            # Decoy — different topic (T_LOCATION) so no conflict with anchor,
            # but still in-scope so it occupies a slot in the top-2 overfetch.
            await idx.astore(
                content="Bob works remotely from London",
                scope=scope,
                embedding=_unit_embedding(T_LOCATION, variant=1.0),
            )
            # Neighbour — negated T_SKILL embedding: cosine sim ≈ -0.96 with the
            # query, so it is ranked absolutely last and never in the top-fetch_k
            # vector results.  It can ONLY enter via graph expansion.
            id_neighbour = await idx.astore(
                content="Bob's favourite food is tacos",
                scope=scope,
                embedding=_neg_embedding(T_SKILL, variant=1.0),
            )

            from datetime import datetime, timezone
            now = datetime.now(timezone.utc)
            await idx._store.add_graph_edge(
                id_anchor, id_neighbour, "RELATES_TO", now,
            )
            idx._graph.load_edges([Edge(id_anchor, id_neighbour, Relationship.RELATES_TO, now)])

            # Clear the pending buffer so units are sourced from the vector backend
            # rather than the in-memory read-after-write cache (which gives 1.0 score
            # to every just-stored unit, masking the graph expansion filter).
            idx._pending.clear()

            q_emb = _unit_embedding(T_SKILL, variant=0.0)
            results = await idx.aretrieve(
                query="Tell me about Bob's professional skills",
                scope=scope,
                top_k=1,
                query_embedding=q_emb,
            )
            contents = [r.content for r in results]
            assert any("data scientist" in c or "London" in c for c in contents)
            assert not any("tacos" in c for c in contents), (
                f"With min_score=0.9 dissimilar graph neighbour should be excluded, got: {contents}"
            )
