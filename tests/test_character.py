"""Tests for the character module — YAML loading, validation, and summary."""

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
        "player": {
            "name": "罗恩",
            "pathway": "占卜家",
            "sequence": 9,
            "anchor": "守护家人",
            "core": [
                "贝克兰德东区出身的青年",
                "父亲生前是值夜者外围成员",
                "性格谨慎内敛",
            ],
            "attributes": {
                "力量": 10,
                "敏捷": 12,
                "体质": 11,
                "智力": 15,
                "感知": 14,
                "魅力": 10,
                "灵性": 16,
            },
            "skills": [
                {"name": "占卜", "value": 60},
                {"name": "观察", "value": 65},
                {"name": "灵性感知", "value": 70},
            ],
            "beyonder_abilities": [
                {"name": "灵摆占卜", "description": "使用灵摆进行简单的吉凶占卜"},
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
        "player": {
            "name": "测试角色",
            "core": ["只是一个测试角色"],
            "attributes": {"力量": 10},
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
        assert c.name == "罗恩"
        assert len(c.core) == 3
        assert c.attributes["敏捷"] == 12
        assert len(c.skills) == 3
        assert c.skills[0]["name"] == "占卜"

    def test_load_class_method(self, valid_character_yaml):
        """Character.load() class method should behave identically."""
        c = Character.load(valid_character_yaml)
        assert isinstance(c, Character)
        assert c.name == "罗恩"

    def test_pathway_loaded(self, valid_character_yaml):
        """Pathway, sequence, and anchor should be loaded from YAML."""
        c = load_character(valid_character_yaml)
        assert c.pathway == "占卜家"
        assert c.sequence == 9
        assert c.anchor == "守护家人"

    def test_beyonder_abilities_loaded(self, valid_character_yaml):
        """Beyonder abilities list should be loaded."""
        c = load_character(valid_character_yaml)
        assert len(c.beyonder_abilities) == 1
        assert c.beyonder_abilities[0]["name"] == "灵摆占卜"

    def test_backward_compatible(self, minimal_character_yaml):
        """Old-style YAML without pathway/sequence should load with defaults."""
        c = load_character(minimal_character_yaml)
        assert c.pathway == ""
        assert c.sequence == 9
        assert c.anchor == ""
        assert c.beyonder_abilities == []

    def test_missing_name(self, tmp_path):
        """Missing 'name' should print a friendly error and exit."""
        data = {
            "player": {
                "core": ["测试"],
                "attributes": {"力量": 10},
            }
        }
        path = _write_yaml(tmp_path, "no_name.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_core(self, tmp_path):
        """Missing 'core' should print a friendly error and exit."""
        data = {
            "player": {
                "name": "测试",
                "attributes": {"力量": 10},
            }
        }
        path = _write_yaml(tmp_path, "no_core.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_attributes(self, tmp_path):
        """Missing 'attributes' should print a friendly error and exit."""
        data = {
            "player": {
                "name": "测试",
                "core": ["测试"],
            }
        }
        path = _write_yaml(tmp_path, "no_attributes.yaml", data)
        with pytest.raises(SystemExit) as exc:
            load_character(path)
        assert exc.value.code == 1

    def test_missing_player_section(self, tmp_path):
        """Missing top-level 'player' key should print a friendly error and exit."""
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
        assert c.skills == []


class TestSummary:
    """Tests for Character.summary()."""

    def test_contains_name(self, valid_character_yaml):
        """Summary should include character name."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        assert c.name in s

    def test_contains_pathway_info(self, valid_character_yaml):
        """Summary should include pathway, sequence, and anchor."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        assert "占卜家" in s
        assert "序列9" in s
        assert "守护家人" in s

    def test_contains_beyonder_abilities(self, valid_character_yaml):
        """Summary should include beyonder abilities."""
        c = load_character(valid_character_yaml)
        s = c.summary()
        assert "灵摆占卜" in s
        assert "吉凶占卜" in s

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
        assert "10" in s
