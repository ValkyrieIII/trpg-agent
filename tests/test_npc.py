"""Tests for the NPC module — NPCCharacter dataclass and NPCStore persistence."""

import sys

import pytest
import yaml

from trpg_agent.npc import NPCCharacter, NPCStore, load_npc


# ===================================================================
#  NPCCharacter Tests
# ===================================================================

# -------------------------------------------------------------------
#  Fixtures — temporary YAML files
# -------------------------------------------------------------------

@pytest.fixture
def complete_npc_yaml(tmp_path):
    """Write a complete NPC YAML and return its path."""
    data = {
        "npc": {
            "name": "老李",
            "core": [
                "酒馆老板，年过五旬",
                "见多识广，消息灵通",
                "表面热情实则精明",
            ],
            "attributes": {
                "strength": 10,
                "agility": 8,
                "intelligence": 14,
                "willpower": 12,
            },
            "skills": [
                {"name": "说服", "value": 70},
                {"name": "聆听", "value": 65},
            ],
            "personality": {
                "tone": "温和热情",
                "verbal_tics": "喜欢用'哎呀'开头",
                "catchphrases": [
                    "哎呀，这年头生意不好做啊",
                    "来来来，喝杯酒暖暖身子",
                ],
                "emotion_map": {
                    "calm": "笑眯眯地擦着酒杯",
                    "wary": "眼神闪烁，笑容收敛",
                    "hostile": "握紧了手中的酒瓶",
                },
            },
            "few_shot": [
                {
                    "input": "老板，最近有什么新闻吗？",
                    "output": "哎呀，您可问对人了。昨儿个听说北边的森林里出了怪事...",
                },
                {
                    "input": "来杯酒。",
                    "output": "好嘞！这是本店招牌的蜜酒，包您满意。",
                },
            ],
        }
    }
    path = tmp_path / "npc_laoli.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)


@pytest.fixture
def minimal_npc_yaml(tmp_path):
    """Minimal valid NPC (no skills, no few_shot)."""
    data = {
        "npc": {
            "name": "路人甲",
            "core": ["一个普通的路人"],
            "attributes": {"strength": 10},
            "personality": {
                "tone": "平淡",
                "verbal_tics": "无",
                "emotion_map": {
                    "calm": "面无表情",
                },
            },
        }
    }
    path = tmp_path / "npc_minimal.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)


# -------------------------------------------------------------------
#  Test: Load NPC from YAML
# -------------------------------------------------------------------

class TestLoadNPC:
    """Tests for load_npc() and NPCCharacter.load()."""

    def test_load_complete_npc(self, complete_npc_yaml):
        """Loading a complete YAML should return a fully populated NPCCharacter."""
        npc = load_npc(complete_npc_yaml)
        assert npc.name == "老李"
        assert len(npc.core) == 3
        assert npc.attributes["agility"] == 8
        assert len(npc.skills) == 2
        assert npc.skills[0]["name"] == "说服"
        assert npc.personality["tone"] == "温和热情"
        assert len(npc.personality["catchphrases"]) == 2
        assert npc.personality["emotion_map"]["calm"] == "笑眯眯地擦着酒杯"
        assert len(npc.few_shot) == 2
        assert npc.few_shot[0]["input"] == "老板，最近有什么新闻吗？"

    def test_load_class_method(self, complete_npc_yaml):
        """NPCCharacter.load() class method should behave identically."""
        npc = NPCCharacter.load(complete_npc_yaml)
        assert isinstance(npc, NPCCharacter)
        assert npc.name == "老李"

    def test_minimal_npc(self, minimal_npc_yaml):
        """A minimal NPC (no skills/few_shot) should load."""
        npc = load_npc(minimal_npc_yaml)
        assert npc.name == "路人甲"
        assert npc.skills == []
        assert npc.few_shot == []

    def test_missing_name(self, tmp_path):
        """Missing 'name' should print a friendly error and exit."""
        data = {
            "npc": {
                "core": ["测试"],
                "attributes": {"strength": 10},
                "personality": {"tone": "平淡", "verbal_tics": "无", "emotion_map": {}},
            }
        }
        path = tmp_path / "no_name.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        with pytest.raises(SystemExit) as exc:
            load_npc(str(path))
        assert exc.value.code == 1

    def test_missing_npc_section(self, tmp_path):
        """Missing top-level 'npc' key should print a friendly error and exit."""
        data = {"player": {"name": "test"}}
        path = tmp_path / "no_npc_section.yaml"
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True)
        with pytest.raises(SystemExit) as exc:
            load_npc(str(path))
        assert exc.value.code == 1

    def test_file_not_found(self):
        """Non-existent file should print a friendly error and exit."""
        with pytest.raises(SystemExit) as exc:
            load_npc("/nonexistent/path.yaml")
        assert exc.value.code == 1


# -------------------------------------------------------------------
#  Test: build_personality_prompt
# -------------------------------------------------------------------

class TestBuildPersonalityPrompt:
    """Tests for NPCCharacter.build_personality_prompt()."""

    def test_contains_core(self, complete_npc_yaml):
        """Prompt should include core background lines."""
        npc = load_npc(complete_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "酒馆老板，年过五旬" in prompt
        assert "见多识广，消息灵通" in prompt

    def test_contains_tone_and_verbal_tics(self, complete_npc_yaml):
        """Prompt should include tone and verbal tics."""
        npc = load_npc(complete_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "温和热情" in prompt
        assert "哎呀" in prompt

    def test_contains_catchphrases(self, complete_npc_yaml):
        """Prompt should include catchphrases."""
        npc = load_npc(complete_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "这年头生意不好做啊" in prompt
        assert "喝杯酒暖暖身子" in prompt

    def test_contains_few_shot_examples(self, complete_npc_yaml):
        """Prompt should include few-shot dialogue examples."""
        npc = load_npc(complete_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "有什么新闻吗" in prompt
        assert "招牌的蜜酒" in prompt

    def test_no_catchphrases_minimal(self, minimal_npc_yaml):
        """Minimal NPC without catchphrases should not include the section."""
        npc = load_npc(minimal_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "【口头禅】" not in prompt

    def test_no_few_shot_minimal(self, minimal_npc_yaml):
        """Minimal NPC without few_shot should not include examples."""
        npc = load_npc(minimal_npc_yaml)
        prompt = npc.build_personality_prompt()
        assert "【对话示例】" not in prompt


# -------------------------------------------------------------------
#  Test: build_state_prompt
# -------------------------------------------------------------------

class TestBuildStatePrompt:
    """Tests for NPCCharacter.build_state_prompt()."""

    def test_contains_emotion_and_trust(self, complete_npc_yaml):
        """State prompt should include emotion, trust, stamina."""
        npc = load_npc(complete_npc_yaml)
        state = {"emotion": "calm", "trust": 0.7, "stamina": "fresh"}
        prompt = npc.build_state_prompt(state)
        assert "calm" in prompt
        assert "0.7" in prompt
        assert "fresh" in prompt

    def test_contains_behaviour_from_emotion_map(self, complete_npc_yaml):
        """State prompt should map emotion to behaviour via emotion_map."""
        npc = load_npc(complete_npc_yaml)
        state = {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}
        prompt = npc.build_state_prompt(state)
        assert "笑眯眯地擦着酒杯" in prompt

    def test_wary_emotion(self, complete_npc_yaml):
        """Wary emotion should map to its behaviour."""
        npc = load_npc(complete_npc_yaml)
        state = {"emotion": "wary", "trust": 0.3, "stamina": "tired"}
        prompt = npc.build_state_prompt(state)
        assert "眼神闪烁" in prompt

    def test_hostile_emotion(self, complete_npc_yaml):
        """Hostile emotion should map to its behaviour."""
        npc = load_npc(complete_npc_yaml)
        state = {"emotion": "hostile", "trust": 0.1, "stamina": "exhausted"}
        prompt = npc.build_state_prompt(state)
        assert "握紧了手中的酒瓶" in prompt

    def test_unknown_emotion_uses_default(self, complete_npc_yaml):
        """Unknown emotion should fall back to '正常反应'."""
        npc = load_npc(complete_npc_yaml)
        state = {"emotion": "nonexistent", "trust": 0.5, "stamina": "fresh"}
        prompt = npc.build_state_prompt(state)
        assert "正常反应" in prompt

    def test_empty_state_uses_defaults(self, complete_npc_yaml):
        """Empty state dict should use default values."""
        npc = load_npc(complete_npc_yaml)
        prompt = npc.build_state_prompt({})
        assert "calm" in prompt
        assert "0.5" in prompt
        assert "fresh" in prompt
        assert "笑眯眯地擦着酒杯" in prompt


# ===================================================================
#  NPCStore Tests
# ===================================================================

# -------------------------------------------------------------------
#  Fixture
# -------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    """Return an NPCStore backed by a temporary directory."""
    persist_dir = str(tmp_path / "chroma_npcs")
    return NPCStore(persist_dir=persist_dir)


@pytest.fixture
def sample_npc():
    """Return a fully populated NPCCharacter for store tests."""
    return NPCCharacter(
        name="老李",
        core=["酒馆老板，年过五旬", "见多识广，消息灵通"],
        attributes={"strength": 10, "agility": 8, "intelligence": 14},
        skills=[{"name": "说服", "value": 70}],
        personality={
            "tone": "温和热情",
            "verbal_tics": "喜欢用'哎呀'开头",
            "catchphrases": ["哎呀，这年头生意不好做啊"],
            "emotion_map": {
                "calm": "笑眯眯地擦着酒杯",
                "wary": "眼神闪烁",
                "hostile": "握紧了手中的酒瓶",
            },
        },
        few_shot=[
            {"input": "老板，最近有什么新闻吗？", "output": "哎呀，您可问对人了。"},
        ],
    )


@pytest.fixture
def second_npc():
    """Another NPC for multi-NPC store tests."""
    return NPCCharacter(
        name="铁匠老王",
        core=["铁匠铺的老板", "力大无穷，性格豪爽"],
        attributes={"strength": 18, "agility": 6, "intelligence": 10},
        skills=[{"name": "锻造", "value": 90}, {"name": "议价", "value": 40}],
        personality={
            "tone": "豪爽直率",
            "verbal_tics": "嗓门大",
            "emotion_map": {
                "calm": "叮叮当当打铁",
                "wary": "放下铁锤打量",
                "hostile": "举起了铁锤",
            },
        },
    )


# -------------------------------------------------------------------
#  Test: Save and Find by Name
# -------------------------------------------------------------------

class TestSaveAndFindByName:
    """Saving NPCs and looking them up by name."""

    def test_save_and_find(self, store, sample_npc):
        """After save, find_by_name should return the NPC."""
        store.save(sample_npc)
        found = store.find_by_name("老李")
        assert found is not None
        assert found.name == "老李"
        assert found.attributes["intelligence"] == 14
        assert found.skills[0]["name"] == "说服"
        assert found.personality["tone"] == "温和热情"
        assert len(found.few_shot) == 1

    def test_find_nonexistent_returns_none(self, store):
        """Looking up a name that was never saved returns None."""
        assert store.find_by_name("不存在的人") is None

    def test_save_overwrites_existing(self, store, sample_npc):
        """Saving an NPC with the same name should overwrite."""
        store.save(sample_npc)
        # Modify and re-save
        updated = NPCCharacter(
            name="老李",
            core=["更新后的背景"],
            attributes={"strength": 99},
            personality={"tone": "updated", "verbal_tics": "updated", "emotion_map": {}},
        )
        store.save(updated)
        found = store.find_by_name("老李")
        assert found is not None
        assert found.core == ["更新后的背景"]
        assert found.attributes["strength"] == 99

    def test_find_returns_correct_npc(self, store, sample_npc, second_npc):
        """Multiple NPCs saved, find returns the correct one."""
        store.save(sample_npc)
        store.save(second_npc)
        found = store.find_by_name("铁匠老王")
        assert found is not None
        assert found.name == "铁匠老王"
        assert found.skills[0]["value"] == 90


# -------------------------------------------------------------------
#  Test: Search by Query
# -------------------------------------------------------------------

class TestSearchByQuery:
    """Semantic search for NPCs."""

    def test_search_finds_relevant_npc(self, store, sample_npc):
        """Semantic search should find NPC with matching theme."""
        store.save(sample_npc)
        results = store.search("酒馆老板", n=5)
        assert len(results) >= 1
        assert any("老李" in r.name for r in results)

    def test_search_returns_multiple(self, store, sample_npc, second_npc):
        """Search with multiple NPCs stored should return multiple results."""
        store.save(sample_npc)
        store.save(second_npc)
        results = store.search("老板", n=5)
        assert len(results) >= 2

    def test_search_empty_returns_empty_list(self, store):
        """Search on empty store returns empty list."""
        results = store.search("anything", n=5)
        assert isinstance(results, list)
        assert len(results) == 0

    def test_search_no_match(self, store, sample_npc):
        """Completely unrelated query may return empty or low-relevance results."""
        store.save(sample_npc)
        results = store.search("量子物理", n=5)
        # ChromaDB may still return something; just assert it's a list
        assert isinstance(results, list)


# -------------------------------------------------------------------
#  Test: All NPCs
# -------------------------------------------------------------------

class TestAllNPCs:
    """Retrieving all NPCs."""

    def test_all_returns_saved_npcs(self, store, sample_npc, second_npc):
        """all() should return all saved NPCs."""
        store.save(sample_npc)
        store.save(second_npc)
        all_npcs = store.all()
        assert len(all_npcs) == 2
        names = {n.name for n in all_npcs}
        assert names == {"老李", "铁匠老王"}

    def test_all_empty_store(self, store):
        """Empty store should return empty list."""
        assert store.all() == []

    def test_all_after_overwrite(self, store, sample_npc):
        """Overwriting an NPC should not change the count."""
        store.save(sample_npc)
        assert len(store.all()) == 1
        updated = NPCCharacter(
            name="老李",
            core=["updated"],
            attributes={"strength": 99},
            personality={"tone": "t", "verbal_tics": "v", "emotion_map": {}},
        )
        store.save(updated)
        assert len(store.all()) == 1


# -------------------------------------------------------------------
#  Test: Create Dynamic NPC
# -------------------------------------------------------------------

class TestCreateDynamicNPC:
    """Dynamically creating and persisting NPCs."""

    def test_create_returns_npc(self, store):
        """create() should return an NPCCharacter instance."""
        npc = store.create(
            name="临时NPC",
            core=["动态创建的NPC"],
            attributes={"strength": 10, "intelligence": 10},
            skills=[{"name": "测试", "value": 50}],
            personality={
                "tone": "友好",
                "verbal_tics": "无",
                "emotion_map": {"calm": "正常"},
            },
        )
        assert isinstance(npc, NPCCharacter)
        assert npc.name == "临时NPC"

    def test_create_persists_to_store(self, store):
        """NPC created via create() should be findable."""
        store.create(
            name="临时NPC",
            core=["动态创建的NPC"],
            attributes={"strength": 10},
            personality={"tone": "t", "verbal_tics": "v", "emotion_map": {}},
        )
        found = store.find_by_name("临时NPC")
        assert found is not None
        assert found.core == ["动态创建的NPC"]

    def test_create_with_minimal_args(self, store):
        """create() should work with only name/core/attributes."""
        npc = store.create(
            name="极简NPC",
            core=["最简单的NPC"],
            attributes={"strength": 10},
        )
        assert npc.skills == []
        assert npc.personality == {}
        assert npc.few_shot == []
        found = store.find_by_name("极简NPC")
        assert found is not None


# -------------------------------------------------------------------
#  Test: Per-NPC Conversation History
# -------------------------------------------------------------------

class TestConversationHistory:
    """Per-NPC conversation history with sliding window."""

    def test_get_history_empty(self, store):
        """NPC with no history returns empty list."""
        assert store.get_history("老李") == []

    def test_append_and_get(self, store):
        """Appending turns should be retrievable."""
        store.append_history("老李", "user", "你好")
        store.append_history("老李", "assistant", "哎呀，客官您好啊！")
        history = store.get_history("老李")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "哎呀，客官您好啊！"

    def test_separate_npc_histories(self, store):
        """Different NPCs should have independent histories."""
        store.append_history("老李", "user", "来杯酒")
        store.append_history("铁匠老王", "user", "打个铁剑")
        assert len(store.get_history("老李")) == 1
        assert len(store.get_history("铁匠老王")) == 1
        assert store.get_history("老李")[0]["content"] == "来杯酒"
        assert store.get_history("铁匠老王")[0]["content"] == "打个铁剑"

    def test_sliding_window(self, store):
        """History beyond max_history * 2 should be trimmed."""
        # Append more than max_history * 2 turns
        for i in range(25):
            store.append_history("老李", "user", f"message {i}")
            store.append_history("老李", "assistant", f"response {i}")

        history = store.get_history("老李")
        # Should retain only the most recent max_history * 2 = 20 turns
        assert len(history) == 20
        # The oldest surviving entry should be message 5 (index 10 / 2 = 5)
        assert "message 5" in history[0]["content"]


# -------------------------------------------------------------------
#  Test: Store Persistence (ChromaDB restore)
# -------------------------------------------------------------------

class TestStorePersistence:
    """NPCs persisted to ChromaDB should restore on store reload."""

    def test_restore_on_reinit(self, tmp_path, sample_npc, second_npc):
        """NPCs saved to one store should be available after reinitialisation."""
        persist_dir = str(tmp_path / "chroma_persist")

        store1 = NPCStore(persist_dir=persist_dir)
        store1.save(sample_npc)
        store1.save(second_npc)

        # Re-create NPCStore with same directory
        store2 = NPCStore(persist_dir=persist_dir)

        assert store2.find_by_name("老李") is not None
        assert store2.find_by_name("铁匠老王") is not None
        assert len(store2.all()) == 2
        assert store2.find_by_name("老李").attributes["intelligence"] == 14

    def test_restore_no_npcs(self, tmp_path):
        """Initialising a store in an empty directory should not error."""
        persist_dir = str(tmp_path / "empty_chroma")
        store = NPCStore(persist_dir=persist_dir)
        assert store.all() == []
