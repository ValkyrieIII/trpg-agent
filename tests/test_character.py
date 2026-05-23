"""Tests for the character module — YAML loading, validation, and prompt building."""

import sys

import pytest
import yaml

from trpg_agent.character import Character, load_character


# ---------------------------------------------------------------------------
#  Fixtures — temporary YAML files
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_character_yaml(tmp_path):
    """Write a complete valid character YAML and return its path."""
    data = {
        "character": {
            "name": "艾琳",
            "core": [
                "你叫艾琳，北方荒原的游侠",
                "曾独自在荒野生存十年",
                "沉默寡言但行动敏锐，习惯在开口前先观察",
            ],
            "personality": {
                "tone": "短句，直接，偶尔冷幽默",
                "verbal_tics": "紧张时重复主语",
                "emotion_map": {
                    "anger": "压低声音，语速变慢",
                    "fear": "过度警觉，反问句增多",
                    "trust": "话变多，偶尔流露脆弱",
                },
                "catchphrases": [
                    "荒野教过我：生存的第一条法则——永远别信任看起来太安全的路。",
                ],
            },
            "few_shot": [
                {"input": "你好", "output": "嗯，活着就好。"},
                {"input": "你叫什么名字？", "output": "艾琳。北边来的。"},
                {"input": "前面有危险吗？", "output": "脚印还很新。小心点。"},
            ],
            "attributes": {
                "strength": 14,
                "agility": 18,
                "intelligence": 12,
                "willpower": 15,
            },
            "skills": [
                {"name": "追踪", "value": 75},
                {"name": "弓箭", "value": 80},
                {"name": "野外生存", "value": 90},
            ],
        }
    }
    path = tmp_path / "character.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)


@pytest.fixture
def minimal_character_yaml(tmp_path):
    """Minimal valid character (no few_shot, no skills)."""
    data = {
        "character": {
            "name": "测试角色",
            "core": ["只是一个测试角色"],
            "personality": {
                "tone": "普通",
                "verbal_tics": "无",
                "emotion_map": {},
                "catchphrases": [],
            },
            "attributes": {"strength": 10},
        }
    }
    path = tmp_path / "minimal.yaml"
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)


def _write_yaml(tmp_path, filename, data):
    """Helper: write a partial character YAML for validation tests."""
    path = tmp_path / filename
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)
    return str(path)


# ---------------------------------------------------------------------------
#  Loading & validation
# ---------------------------------------------------------------------------

class TestLoadCharacter:
    """Tests for load_character() and Character.load()."""

    def test_load_valid_character(self, valid_character_yaml):
        """Loading a complete YAML should return a fully populated Character."""
        c = load_character(valid_character_yaml)
        assert c.name == "艾琳"
        assert len(c.core) == 3
        assert c.personality["tone"] == "短句，直接，偶尔冷幽默"
        assert len(c.few_shot) == 3
        assert c.attributes["agility"] == 18
        assert len(c.skills) == 3
        assert c.skills[0]["name"] == "追踪"

    def test_load_class_method(self, valid_character_yaml):
        """Character.load() class method should behave identically."""
        c = Character.load(valid_character_yaml)
        assert isinstance(c, Character)
        assert c.name == "艾琳"

    def test_missing_name(self, tmp_path):
        """Missing 'name' should print a friendly error and exit."""
        data = {
            "character": {
                "core": ["测试"],
                "personality": {"tone": "普通", "verbal_tics": "无", "emotion_map": {}, "catchphrases": []},
                "attributes": {"strength": 10},
            }
        }
        path = _write_yaml(tmp_path, "no_name.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_core(self, tmp_path):
        """Missing 'core' should print a friendly error and exit."""
        data = {
            "character": {
                "name": "测试",
                "personality": {"tone": "普通", "verbal_tics": "无", "emotion_map": {}, "catchphrases": []},
                "attributes": {"strength": 10},
            }
        }
        path = _write_yaml(tmp_path, "no_core.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_personality(self, tmp_path):
        """Missing 'personality' should print a friendly error and exit."""
        data = {
            "character": {
                "name": "测试",
                "core": ["测试"],
                "attributes": {"strength": 10},
            }
        }
        path = _write_yaml(tmp_path, "no_personality.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_attributes(self, tmp_path):
        """Missing 'attributes' should print a friendly error and exit."""
        data = {
            "character": {
                "name": "测试",
                "core": ["测试"],
                "personality": {"tone": "普通", "verbal_tics": "无", "emotion_map": {}, "catchphrases": []},
            }
        }
        path = _write_yaml(tmp_path, "no_attributes.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_character_section(self, tmp_path):
        """Missing top-level 'character' key should print a friendly error and exit."""
        data = {"something_else": 42}
        path = _write_yaml(tmp_path, "no_character_section.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_file_not_found(self):
        """Non-existent file should print a friendly error and exit."""
        with pytest.raises(SystemExit) as exc:
            load_character("/nonexistent/path.yaml")
        assert exc.value.code == 1

    def test_minimal_character(self, minimal_character_yaml):
        """A character with only required fields (no few_shot/skills) should load."""
        c = load_character(minimal_character_yaml)
        assert c.name == "测试角色"
        assert c.few_shot == []
        assert c.skills == []


# ---------------------------------------------------------------------------
#  Prompt building
# ---------------------------------------------------------------------------

class TestBuildPersonalityPrompt:
    """Tests for Character.build_personality_prompt()."""

    def test_contains_core(self, valid_character_yaml):
        """Prompt should include all core background lines."""
        c = load_character(valid_character_yaml)
        prompt = c.build_personality_prompt()
        for line in c.core:
            assert line in prompt, f"Core line missing: {line}"

    def test_contains_tone(self, valid_character_yaml):
        """Prompt should include the personality tone setting."""
        c = load_character(valid_character_yaml)
        prompt = c.build_personality_prompt()
        assert c.personality["tone"] in prompt

    def test_contains_catchphrases(self, valid_character_yaml):
        """Prompt should include catchphrases."""
        c = load_character(valid_character_yaml)
        prompt = c.build_personality_prompt()
        for cp in c.personality["catchphrases"]:
            assert cp in prompt, f"Catchphrase missing: {cp}"

    def test_contains_few_shot_examples(self, valid_character_yaml):
        """Prompt must include all few_shot examples as formatted dialogue."""
        c = load_character(valid_character_yaml)
        prompt = c.build_personality_prompt()
        for example in c.few_shot:
            assert example["input"] in prompt, f"Few-shot input missing: {example['input']}"
            assert example["output"] in prompt, f"Few-shot output missing: {example['output']}"

    def test_minimal_still_has_core_and_tone(self, minimal_character_yaml):
        """Even minimal characters should produce a personality prompt."""
        c = load_character(minimal_character_yaml)
        prompt = c.build_personality_prompt()
        assert c.core[0] in prompt
        assert c.personality["tone"] in prompt

    def test_empty_few_shot_does_not_crash(self, minimal_character_yaml):
        """Character with no few_shot should still build prompt without error."""
        c = load_character(minimal_character_yaml)
        prompt = c.build_personality_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestBuildStatePrompt:
    """Tests for Character.build_state_prompt()."""

    def test_anger_emotion(self, valid_character_yaml):
        """Anger emotion should map to '压低声音，语速变慢'."""
        c = load_character(valid_character_yaml)
        state = {"emotion": "anger", "trust": 0.3, "stamina": "tired"}
        prompt = c.build_state_prompt(state)
        assert "压低声音，语速变慢" in prompt
        assert "anger" in prompt
        assert "0.3" in prompt
        assert "tired" in prompt

    def test_fear_emotion(self, valid_character_yaml):
        """Fear emotion should map to '过度警觉，反问句增多'."""
        c = load_character(valid_character_yaml)
        state = {"emotion": "fear", "trust": 0.5, "stamina": "fresh"}
        prompt = c.build_state_prompt(state)
        assert "过度警觉，反问句增多" in prompt

    def test_trust_emotion(self, valid_character_yaml):
        """Trust emotion should map to '话变多，偶尔流露脆弱'."""
        c = load_character(valid_character_yaml)
        state = {"emotion": "trust", "trust": 0.8, "stamina": "fresh"}
        prompt = c.build_state_prompt(state)
        assert "话变多，偶尔流露脆弱" in prompt

    def test_unknown_emotion_fallback(self, valid_character_yaml):
        """Unknown emotion should fall back to a default description."""
        c = load_character(valid_character_yaml)
        state = {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}
        prompt = c.build_state_prompt(state)
        # "calm" is not in the emotion_map, so it should fall back
        assert "calm" in prompt
        assert "0.5" in prompt
        assert "fresh" in prompt


class TestSummary:
    """Tests for Character.summary()."""

    def test_contains_name(self, valid_character_yaml):
        """Summary should include character name."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        assert c.name in s

    def test_contains_all_attributes(self, valid_character_yaml):
        """Summary should include all attribute names and values."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        for attr_name, attr_value in c.attributes.items():
            assert attr_name in s
            assert str(attr_value) in s

    def test_contains_all_skills(self, valid_character_yaml):
        """Summary should include all skill names and values."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        for skill in c.skills:
            assert skill["name"] in s
            assert str(skill["value"]) in s

    def test_minimal_summary(self, minimal_character_yaml):
        """Minimal character (no skills) should still produce a summary."""
        c = load_character(minimal_character_yaml)
        s = c.summary()
        assert c.name in s
        assert "strength" in s
        assert "10" in s
