"""Tests for the RAG knowledge base — KnowledgeBase with ChromaDB + permission filtering."""

import pytest

from trpg_agent.rag import KnowledgeBase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def kb(tmp_path):
    """Return a KnowledgeBase backed by a temporary directory."""
    persist_dir = str(tmp_path / "chroma_knowledge")
    return KnowledgeBase(persist_dir=persist_dir)


# ---------------------------------------------------------------------------
# Permission filtering
# ---------------------------------------------------------------------------

class TestPermissionFiltering:
    """Knowledge visibility is correctly restricted by known_by."""

    def test_add_and_query_with_permission(self, kb):
        """酒馆老板 can see secret passage; 旅人 cannot."""
        kb.add_knowledge(
            "酒馆地下有一条密道通往郊外，入口在酒窖第三个木桶后面。",
            known_by=["酒馆老板"],
        )

        results = kb.query("密道", character="酒馆老板")
        assert len(results) > 0
        assert "密道" in results[0]

        results = kb.query("密道", character="旅人")
        assert len(results) == 0

    def test_public_knowledge(self, kb):
        """Public knowledge (known_by=所有人) is visible to any character."""
        kb.add_knowledge("北方荒原常年被冰雪覆盖，只有苔藓能够生长。", known_by="所有人")

        results = kb.query("北方荒原", character="旅人")
        assert len(results) > 0
        assert "北方荒原" in results[0]

        results = kb.query("北方荒原", character="酒馆老板")
        assert len(results) > 0
        assert "北方荒原" in results[0]

    def test_known_by_as_string(self, kb):
        """known_by as a plain string (not list) is handled correctly."""
        kb.add_knowledge("城堡大门在每晚十点准时关闭。", known_by="所有人")

        results = kb.query("城堡大门", character="任意角色")
        assert len(results) > 0
        assert "城堡大门" in results[0]


# ---------------------------------------------------------------------------
# Threshold filtering
# ---------------------------------------------------------------------------

class TestThreshold:
    """Semantic distance threshold correctly filters unrelated results."""

    def test_threshold_filter(self, kb):
        """A completely unrelated query returns no results."""
        kb.add_knowledge(
            "北境荒原生长着一种名为'冰棘草'的耐寒植物，其汁液可作解毒剂。",
            known_by="所有人",
        )

        # Use a very strict threshold so unrelated content is excluded
        results = kb.query("量子力学相对论黑洞", character="旅人", threshold=0.1)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# load_from_dir
# ---------------------------------------------------------------------------

class TestLoadFromDir:
    """Loading knowledge from .md files in a directory."""

    def test_load_from_dir(self, tmp_path):
        """After loading from .md files, knowledge is queryable."""
        persist_dir = str(tmp_path / "chroma_knowledge")
        kb = KnowledgeBase(persist_dir=persist_dir)

        md_file = tmp_path / "test_lore.md"
        md_file.write_text(
            "龙裔是古代龙族与人类的混血后裔，拥有操控雷电的天赋。",
            encoding="utf-8",
        )

        kb.load_from_dir(str(tmp_path))

        results = kb.query("龙裔雷电", character="旅人")
        assert len(results) > 0
        assert "龙裔" in results[0]
