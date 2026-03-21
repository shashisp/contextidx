from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from enum import Enum
from typing import NamedTuple


class Relationship(str, Enum):
    SUPERSEDES = "supersedes"
    RELATES_TO = "relates_to"
    VERSION_OF = "version_of"
    CAUSED_BY = "caused_by"


class Edge(NamedTuple):
    from_id: str
    to_id: str
    relationship: Relationship
    created_at: datetime


class TemporalGraph:
    """In-memory directed graph tracking relationships between ContextUnits.

    Persisted to the ``context_graph`` table via the Store layer.

    Parameters
    ----------
    max_edge_nodes:
        Maximum number of *nodes* whose edges are kept in memory.  When the
        cap is exceeded the coldest 10 % of nodes are evicted — their edges
        are removed from the in-memory graph but remain in the database.
        ``None`` (default) means unlimited, preserving the original behaviour.

        Evicted nodes are tracked in :attr:`evicted_nodes`.  Callers that
        need graph neighbours for an evicted node should fall back to the
        store (``Store.get_graph_edges(unit_id)``).
    """

    def __init__(self, max_edge_nodes: int | None = None) -> None:
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._incoming: dict[str, list[Edge]] = defaultdict(list)
        self._max_edge_nodes = max_edge_nodes
        # LRU tracking: unit_id → last access time (monotonic seconds)
        self._lru: dict[str, float] = {}
        # Nodes evicted from the in-memory graph (still in the DB)
        self.evicted_nodes: set[str] = set()

    # ── LRU helpers ──

    def _touch(self, *unit_ids: str) -> None:
        t = time.monotonic()
        for uid in unit_ids:
            self._lru[uid] = t

    def _maybe_evict(self) -> None:
        if self._max_edge_nodes is None:
            return
        n = len(self._lru)
        if n <= self._max_edge_nodes:
            return
        evict_count = max(1, n // 10)
        oldest = sorted(self._lru.items(), key=lambda x: x[1])[:evict_count]
        evicted = {uid for uid, _ in oldest}
        for uid in evicted:
            self._lru.pop(uid, None)
            self._outgoing.pop(uid, None)
            self._incoming.pop(uid, None)
        self.evicted_nodes.update(evicted)
        # Purge dangling references in remaining adjacency lists
        for uid in list(self._outgoing):
            self._outgoing[uid] = [
                e for e in self._outgoing[uid] if e.to_id not in evicted
            ]
        for uid in list(self._incoming):
            self._incoming[uid] = [
                e for e in self._incoming[uid] if e.from_id not in evicted
            ]

    # ── Public API ──

    def add_edge(
        self,
        from_id: str,
        to_id: str,
        relationship: str | Relationship,
        created_at: datetime,
    ) -> Edge:
        rel = Relationship(relationship)
        edge = Edge(from_id=from_id, to_id=to_id, relationship=rel, created_at=created_at)
        self._outgoing[from_id].append(edge)
        self._incoming[to_id].append(edge)
        self._touch(from_id, to_id)
        # Re-admit previously evicted nodes that are actively being written
        self.evicted_nodes.discard(from_id)
        self.evicted_nodes.discard(to_id)
        self._maybe_evict()
        return edge

    def get_superseded(self, unit_id: str) -> list[str]:
        """Get IDs of all units directly superseded by *unit_id*."""
        self._touch(unit_id)
        return [
            e.to_id
            for e in self._outgoing.get(unit_id, [])
            if e.relationship == Relationship.SUPERSEDES
        ]

    def find_superseded_by(self, unit_id: str) -> str | None:
        """Find the unit that supersedes *unit_id*, if any."""
        self._touch(unit_id)
        for e in self._incoming.get(unit_id, []):
            if e.relationship == Relationship.SUPERSEDES:
                return e.from_id
        return None

    def get_lineage(self, unit_id: str) -> list[str]:
        """Walk the VERSION_OF chain backwards to the root, then forwards.

        Returns ordered list: [oldest_version, ..., newest_version].
        """
        self._touch(unit_id)
        root = unit_id
        visited: set[str] = {root}
        while True:
            parent = self._find_parent_version(root)
            if parent is None or parent in visited:
                break
            visited.add(parent)
            root = parent

        chain = [root]
        visited_chain: set[str] = {root}
        current = root
        while True:
            child = self._find_child_version(current)
            if child is None or child in visited_chain:
                break
            visited_chain.add(child)
            chain.append(child)
            current = child
        return chain

    def get_related(self, unit_id: str) -> list[str]:
        """Get bidirectional RELATES_TO neighbours."""
        self._touch(unit_id)
        ids: set[str] = set()
        for e in self._outgoing.get(unit_id, []):
            if e.relationship == Relationship.RELATES_TO:
                ids.add(e.to_id)
        for e in self._incoming.get(unit_id, []):
            if e.relationship == Relationship.RELATES_TO:
                ids.add(e.from_id)
        return list(ids)

    def get_caused_by(self, unit_id: str) -> list[str]:
        """Get units that caused *unit_id*."""
        self._touch(unit_id)
        return [
            e.from_id
            for e in self._incoming.get(unit_id, [])
            if e.relationship == Relationship.CAUSED_BY
        ]

    def get_edges_for(self, unit_id: str) -> list[Edge]:
        """All edges involving *unit_id* (outgoing and incoming)."""
        self._touch(unit_id)
        out = list(self._outgoing.get(unit_id, []))
        inc = [e for e in self._incoming.get(unit_id, []) if e not in out]
        return out + inc

    def was_evicted(self, unit_id: str) -> bool:
        """Return ``True`` if *unit_id*'s edges were evicted from the cache.

        Callers should fall back to ``Store.get_graph_edges(unit_id)`` when
        this returns ``True`` and the graph returned an empty neighbour list.
        """
        return unit_id in self.evicted_nodes

    def remove_units(self, unit_ids: set[str]) -> None:
        """Remove all edges involving any of *unit_ids* from the graph."""
        for uid in unit_ids:
            self._outgoing.pop(uid, None)
            self._incoming.pop(uid, None)
            self._lru.pop(uid, None)
            self.evicted_nodes.discard(uid)
        # Also purge dangling references in remaining adjacency lists
        for uid in list(self._outgoing):
            self._outgoing[uid] = [e for e in self._outgoing[uid] if e.to_id not in unit_ids]
        for uid in list(self._incoming):
            self._incoming[uid] = [e for e in self._incoming[uid] if e.from_id not in unit_ids]

    def clear(self) -> None:
        """Remove all edges from the graph."""
        self._outgoing.clear()
        self._incoming.clear()
        self._lru.clear()
        self.evicted_nodes.clear()

    def load_edges(self, edges: list[Edge]) -> None:
        """Bulk-load edges (e.g. from database on startup)."""
        for e in edges:
            self._outgoing[e.from_id].append(e)
            self._incoming[e.to_id].append(e)
            self._touch(e.from_id, e.to_id)
        self._maybe_evict()

    def _find_parent_version(self, unit_id: str) -> str | None:
        for e in self._incoming.get(unit_id, []):
            if e.relationship == Relationship.VERSION_OF:
                return e.from_id
        return None

    def _find_child_version(self, unit_id: str) -> str | None:
        for e in self._outgoing.get(unit_id, []):
            if e.relationship == Relationship.VERSION_OF:
                return e.to_id
        return None
