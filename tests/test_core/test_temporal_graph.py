from datetime import datetime, timezone

from contextidx.core.temporal_graph import Edge, Relationship, TemporalGraph


def _now():
    return datetime.now(timezone.utc)


class TestAddEdge:
    def test_add_and_retrieve_supersedes(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        assert g.get_superseded("a") == ["b"]

    def test_find_superseded_by(self):
        g = TemporalGraph()
        g.add_edge("new", "old", "supersedes", _now())
        assert g.find_superseded_by("old") == "new"
        assert g.find_superseded_by("new") is None

    def test_add_edge_returns_edge(self):
        g = TemporalGraph()
        edge = g.add_edge("a", "b", "relates_to", _now())
        assert isinstance(edge, Edge)
        assert edge.from_id == "a"
        assert edge.to_id == "b"


class TestLineage:
    def test_single_version(self):
        g = TemporalGraph()
        assert g.get_lineage("a") == ["a"]

    def test_linear_chain(self):
        g = TemporalGraph()
        g.add_edge("v1", "v2", "version_of", _now())
        g.add_edge("v2", "v3", "version_of", _now())
        chain = g.get_lineage("v2")
        assert chain == ["v1", "v2", "v3"]

    def test_lineage_from_leaf(self):
        g = TemporalGraph()
        g.add_edge("v1", "v2", "version_of", _now())
        g.add_edge("v2", "v3", "version_of", _now())
        chain = g.get_lineage("v3")
        assert chain == ["v1", "v2", "v3"]


class TestRelated:
    def test_bidirectional(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "relates_to", _now())
        assert "b" in g.get_related("a")
        assert "a" in g.get_related("b")


class TestCausedBy:
    def test_caused_by(self):
        g = TemporalGraph()
        g.add_edge("event", "unit", "caused_by", _now())
        assert g.get_caused_by("unit") == ["event"]


class TestLoadEdges:
    def test_bulk_load(self):
        g = TemporalGraph()
        now = _now()
        edges = [
            Edge("a", "b", Relationship.SUPERSEDES, now),
            Edge("c", "d", Relationship.RELATES_TO, now),
        ]
        g.load_edges(edges)
        assert g.get_superseded("a") == ["b"]
        assert "d" in g.get_related("c")


class TestGetEdgesFor:
    def test_returns_all(self):
        g = TemporalGraph()
        now = _now()
        g.add_edge("a", "b", "supersedes", now)
        g.add_edge("c", "a", "relates_to", now)
        edges = g.get_edges_for("a")
        assert len(edges) == 2


class TestRemoveUnits:
    def test_removes_outgoing_edges(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        g.remove_units({"a"})
        assert g.get_superseded("a") == []

    def test_removes_incoming_edges(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        g.remove_units({"b"})
        assert g.find_superseded_by("b") is None

    def test_purges_dangling_references(self):
        """After removing 'b', 'a' should have no outgoing edge to 'b'."""
        g = TemporalGraph()
        g.add_edge("a", "b", "relates_to", _now())
        g.remove_units({"b"})
        assert "b" not in g.get_related("a")

    def test_remove_multiple_units(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        g.add_edge("b", "c", "version_of", _now())
        g.remove_units({"a", "b"})
        assert g.get_superseded("a") == []
        assert g.find_superseded_by("b") is None
        # 'c' still exists; its incoming VERSION_OF from 'b' is gone
        assert g.get_lineage("c") == ["c"]

    def test_remove_nonexistent_unit_is_safe(self):
        g = TemporalGraph()
        g.remove_units({"does_not_exist"})  # should not raise


class TestClear:
    def test_clear_removes_all_edges(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        g.add_edge("c", "d", "relates_to", _now())
        g.clear()
        assert g.get_superseded("a") == []
        assert g.get_related("c") == []

    def test_clear_then_add_works(self):
        g = TemporalGraph()
        g.add_edge("a", "b", "supersedes", _now())
        g.clear()
        g.add_edge("x", "y", "relates_to", _now())
        assert "y" in g.get_related("x")
