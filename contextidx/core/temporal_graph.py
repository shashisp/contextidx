from __future__ import annotations

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

    Persisted to the `context_graph` table via the Store layer.
    """

    def __init__(self) -> None:
        self._outgoing: dict[str, list[Edge]] = defaultdict(list)
        self._incoming: dict[str, list[Edge]] = defaultdict(list)

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
        return edge

    def get_superseded(self, unit_id: str) -> list[str]:
        """Get IDs of all units directly superseded by *unit_id*."""
        return [
            e.to_id
            for e in self._outgoing.get(unit_id, [])
            if e.relationship == Relationship.SUPERSEDES
        ]

    def find_superseded_by(self, unit_id: str) -> str | None:
        """Find the unit that supersedes *unit_id*, if any."""
        for e in self._incoming.get(unit_id, []):
            if e.relationship == Relationship.SUPERSEDES:
                return e.from_id
        return None

    def get_lineage(self, unit_id: str) -> list[str]:
        """Walk the VERSION_OF chain backwards to the root, then forwards.

        Returns ordered list: [oldest_version, ..., newest_version].
        """
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
        return [
            e.from_id
            for e in self._incoming.get(unit_id, [])
            if e.relationship == Relationship.CAUSED_BY
        ]

    def get_edges_for(self, unit_id: str) -> list[Edge]:
        """All edges involving *unit_id* (outgoing and incoming)."""
        out = list(self._outgoing.get(unit_id, []))
        inc = [e for e in self._incoming.get(unit_id, []) if e not in out]
        return out + inc

    def load_edges(self, edges: list[Edge]) -> None:
        """Bulk-load edges (e.g. from database on startup)."""
        for e in edges:
            self._outgoing[e.from_id].append(e)
            self._incoming[e.to_id].append(e)

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
