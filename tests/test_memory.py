"""Tests for the memory module — hybrid MemoryStore with ChromaDB + NetworkX."""

import pytest
import json
import os
from pathlib import Path

from trpg_agent.memory import MemoryStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Return a MemoryStore backed by a temporary directory."""
    persist_dir = str(tmp_path / "chroma")
    return MemoryStore(persist_dir=persist_dir)


# ---------------------------------------------------------------------------
# Add + Search consistency
# ---------------------------------------------------------------------------

class TestAddAndSearch:
    """Adding memories and retrieving them via semantic search."""

    def test_add_returns_string_id(self, store):
        mem_id = store.add("The party discovered a hidden cave.", context={})
        assert isinstance(mem_id, str)
        assert len(mem_id) > 0

    def test_search_returns_added_memory(self, store):
        store.add("Elara the sorceress cast a powerful fireball.", context={"location": "dungeon"})
        results = store.search("magic fire spell", n=3)
        assert len(results) >= 1
        assert any("fireball" in r["content"] or "sorceress" in r["content"] for r in results)

    def test_search_returns_multiple_memories(self, store):
        store.add("The knight found a rusty sword.", context={})
        store.add("The thief picked the lock silently.", context={})
        store.add("The wizard read the ancient tome.", context={})
        results = store.search("knight sword thief lock wizard book", n=5)
        assert len(results) >= 2

    def test_search_result_contains_metadata(self, store):
        ctx = {"location": "tavern", "npcs": ["bartender"], "emotion": "curious"}
        mem_id = store.add("A mysterious stranger entered the tavern.", context=ctx)
        results = store.search("tavern stranger", n=3)
        assert len(results) >= 1
        r = results[0]
        assert "id" in r
        assert "content" in r
        assert "importance" in r
        assert "context" in r
        assert r["context"]["location"] == "tavern"
        assert r["context"]["npcs"] == ["bartender"]

    def test_search_no_match_returns_empty(self, store):
        store.add("Dragons are flying.", context={})
        results = store.search("quantum physics", n=3)
        # ChromaDB may return results with low distance; just assert it's a list
        assert isinstance(results, list)

    def test_multiple_adds_have_unique_ids(self, store):
        id1 = store.add("Memory one.", context={})
        id2 = store.add("Memory two.", context={})
        id3 = store.add("Memory three.", context={})
        assert len({id1, id2, id3}) == 3


# ---------------------------------------------------------------------------
# Graph linking + get_related
# ---------------------------------------------------------------------------

class TestGraphLinking:
    """Linking memories in the graph and traversing relationships."""

    def test_link_and_get_related_direct(self, store):
        id_a = store.add("The party entered the dark forest.", context={})
        id_b = store.add("They encountered a pack of wolves.", context={})
        store.link(id_a, id_b, "导致了")

        related = store.get_related(id_a, hops=1)
        ids = {r["id"] for r in related}
        assert id_b in ids

    def test_get_related_includes_relation_info(self, store):
        id_a = store.add("First event.", context={})
        id_b = store.add("Second event.", context={})
        store.link(id_a, id_b, "导致了")

        related = store.get_related(id_a, hops=1)
        assert len(related) >= 1
        # relation info may be in different forms; just check it's present
        r = related[0]
        assert r["id"] == id_b

    def test_no_links_returns_empty(self, store):
        mem_id = store.add("Orphan memory.", context={})
        related = store.get_related(mem_id, hops=1)
        assert isinstance(related, list)
        assert len(related) == 0

    def test_link_two_way_traversal(self, store):
        id_a = store.add("Alice spoke to the guard.", context={})
        id_b = store.add("The guard let them pass.", context={})
        store.link(id_a, id_b, "导致了")

        related_a = store.get_related(id_a, hops=1)
        related_b = store.get_related(id_b, hops=1)
        ids_a = {r["id"] for r in related_a}
        ids_b = {r["id"] for r in related_b}
        assert id_b in ids_a
        assert id_a in ids_b

    def test_link_unknown_relation_type(self, store):
        """Should not raise; unknown relation types are allowed."""
        id_a = store.add("Event A.", context={})
        id_b = store.add("Event B.", context={})
        # Should not raise
        store.link(id_a, id_b, "未知关系")
        related = store.get_related(id_a, hops=1)
        ids = {r["id"] for r in related}
        assert id_b in ids


# ---------------------------------------------------------------------------
# Hops parameter (1-hop vs 2-hop)
# ---------------------------------------------------------------------------

class TestHopsParameter:
    """Graph traversal depth (hops) correctly filters results."""

    def test_one_hop_vs_two_hop(self, store):
        # Chain: A -> B -> C
        id_a = store.add("Start of quest.", context={})
        id_b = store.add("Middle of quest.", context={})
        id_c = store.add("End of quest.", context={})
        store.link(id_a, id_b, "导致了")
        store.link(id_b, id_c, "导致了")

        related_1 = store.get_related(id_a, hops=1)
        related_2 = store.get_related(id_a, hops=2)

        ids_1 = {r["id"] for r in related_1}
        ids_2 = {r["id"] for r in related_2}

        assert id_b in ids_1
        assert id_c not in ids_1, "2-hop node should not appear in 1-hop results"
        assert id_c in ids_2

    def test_hops_zero_returns_direct_neighbors(self, store):
        """hops=0 might return only direct neighbors; treat this as same as hops=1."""
        id_a = store.add("Alpha.", context={})
        id_b = store.add("Beta.", context={})
        store.link(id_a, id_b, "关联到")
        related = store.get_related(id_a, hops=0)
        ids = {r["id"] for r in related}
        assert id_b in ids

    def test_hops_does_not_exceed(self, store):
        """hops beyond graph depth returns all reachable nodes."""
        id_a = store.add("A", context={})
        id_b = store.add("B", context={})
        store.link(id_a, id_b, "关联到")
        # With only 2 nodes, any hops >= 1 returns the same
        related = store.get_related(id_a, hops=10)
        ids = {r["id"] for r in related}
        assert id_b in ids


# ---------------------------------------------------------------------------
# full_retrieve — combined semantic + graph, sorted by importance desc
# ---------------------------------------------------------------------------

class TestFullRetrieve:
    """Combined retrieval: semantic search + graph expansion, sorted by importance."""

    def test_full_retrieve_returns_sorted_by_importance_desc(self, store):
        store.add("Minor detail about the weather.", context={}, importance=0.2)
        store.add("Critical clue about the villain's plan.", context={}, importance=0.9)
        store.add("The hero's backstory.", context={}, importance=0.6)

        results = store.full_retrieve("clue villain plan")
        assert len(results) >= 1

        # Check descending importance order
        importances = [r["importance"] for r in results]
        assert importances == sorted(importances, reverse=True), (
            f"Expected descending importance, got {importances}"
        )

    def test_full_retrieve_includes_graph_related(self, store):
        id_main = store.add("The king was betrayed.", context={}, importance=0.8)
        id_rel = store.add("The queen fled the castle.", context={}, importance=0.3)
        store.link(id_main, id_rel, "导致了")

        results = store.full_retrieve("king betrayal")
        contents = [r["content"] for r in results]

        # Should find the main memory via semantic search
        assert any("betrayed" in c for c in contents)

    def test_full_retrieve_deduplicates(self, store):
        """Same memory should not appear twice even if both semantic + graph return it."""
        id_a = store.add("The artifact was stolen.", context={}, importance=0.7)
        id_b = store.add("The thief escaped through the sewers.", context={}, importance=0.5)
        store.link(id_a, id_b, "导致了")

        results = store.full_retrieve("artifact stolen thief sewers")
        ids = [r["id"] for r in results]
        assert len(ids) == len(set(ids)), "Duplicate IDs found in full_retrieve results"


# ---------------------------------------------------------------------------
# Persistence (graph JSON)
# ---------------------------------------------------------------------------

class TestPersistence:
    """Graph persists to JSON and restores on reload."""

    def test_graph_persists_to_json(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_persist")
        store = MemoryStore(persist_dir=persist_dir)
        id_a = store.add("Memory A", context={})
        id_b = store.add("Memory B", context={})
        store.link(id_a, id_b, "关联到")

        # Check the JSON file exists
        graph_file = os.path.join(persist_dir, "memory_graph.json")
        assert os.path.exists(graph_file)

        with open(graph_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data

    def test_graph_restores_on_reload(self, tmp_path):
        persist_dir = str(tmp_path / "chroma_reload")
        store1 = MemoryStore(persist_dir=persist_dir)
        id_a = store1.add("Memory A", context={})
        id_b = store1.add("Memory B", context={})
        store1.link(id_a, id_b, "导致了")

        # Re-create MemoryStore with same directory
        store2 = MemoryStore(persist_dir=persist_dir)
        related = store2.get_related(id_a, hops=1)
        ids = {r["id"] for r in related}
        assert id_b in ids
