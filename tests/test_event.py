"""Tests for event resolution module — :mod:`trpg_agent.event`.

Tests each trigger type (trap, environment, npc_reaction, discovery)
plus edge cases such as empty context, None context, and unknown trigger
types.  Dice outcomes are controlled via monkeypatch so all tests are
deterministic.
"""

from __future__ import annotations

import pytest

from trpg_agent.character import Character
from trpg_agent.event import resolve_trigger
from trpg_agent.state import StateMachine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(skills: list | None = None) -> Character:
    """Create a minimal Character for testing."""
    return Character(
        name="TestChar",
        core=["A test character."],
        personality={
            "tone": "neutral",
            "verbal_tics": "none",
            "emotion_map": {},
            "catchphrases": [],
        },
        attributes={"strength": 10},
        skills=skills or [],
    )


# ---------------------------------------------------------------------------
# Trap
# ---------------------------------------------------------------------------


class TestTrap:
    """``trigger_type="trap"`` — difficulty check DC 15."""

    def test_success_when_d20_ge_15(self, monkeypatch):
        """Roll >= 15 → success, no state change."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([15], 15))
        state = StateMachine()
        result = resolve_trigger("trap", _make_character(), state)

        assert result["outcome"] == "success"
        assert result["state_changes"] == []
        assert result["narrative"] == "察觉并避开"
        # State should be untouched
        assert state.get_state()["stamina"] == "fresh"

    def test_failure_when_d20_lt_15(self, monkeypatch):
        """Roll < 15 → failure, state gets 'combat' applied."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([10], 10))
        state = StateMachine()
        result = resolve_trigger("trap", _make_character(), state)

        assert result["outcome"] == "failure"
        assert result["state_changes"] == ["combat"]
        assert result["narrative"] == "触发陷阱受伤"
        # 'combat' reduces stamina: fresh → tired
        assert state.get_state()["stamina"] == "tired"

    def test_failure_on_exact_dc_boundary(self, monkeypatch):
        """Roll 14 (below 15) is still a failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([14], 14))
        state = StateMachine()
        result = resolve_trigger("trap", _make_character(), state)
        assert result["outcome"] == "failure"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironment:
    """``trigger_type="environment"`` — configurable DC."""

    def test_default_dc_success(self, monkeypatch):
        """Default DC 12, roll 12 → success."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([12], 12))
        result = resolve_trigger("environment", _make_character(), StateMachine())
        assert result["outcome"] == "success"
        assert result["narrative"] == "环境判定成功"

    def test_default_dc_failure(self, monkeypatch):
        """Default DC 12, roll 11 → failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([11], 11))
        result = resolve_trigger("environment", _make_character(), StateMachine())
        assert result["outcome"] == "failure"
        assert result["narrative"] == "环境判定失败"

    def test_custom_dc(self, monkeypatch):
        """DC 20 from context, roll 15 → failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([15], 15))
        result = resolve_trigger(
            "environment", _make_character(), StateMachine(), {"dc": 20}
        )
        assert result["outcome"] == "failure"

    def test_custom_narrative(self, monkeypatch):
        """Custom narrative strings from context."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([18], 18))
        result = resolve_trigger(
            "environment",
            _make_character(),
            StateMachine(),
            {"dc": 10, "narrative_success": "平安通过", "narrative_failure": "遭遇险情"},
        )
        assert result["outcome"] == "success"
        assert result["narrative"] == "平安通过"

    def test_empty_context(self, monkeypatch):
        """Empty dict context uses defaults (DC 12)."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([12], 12))
        result = resolve_trigger("environment", _make_character(), StateMachine(), {})
        assert result["outcome"] == "success"

    def test_no_state_change(self, monkeypatch):
        """Environment never mutates the state machine."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([5], 5))
        state = StateMachine()
        resolve_trigger("environment", _make_character(), state)
        assert state.get_state() == {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}


# ---------------------------------------------------------------------------
# NPC Reaction
# ---------------------------------------------------------------------------


class TestNpcReaction:
    """``trigger_type="npc_reaction"`` — trust thresholds."""

    def test_friendly_when_trust_ge_07(self):
        """Trust >= 0.7 → friendly."""
        state = StateMachine(trust=0.7)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "friendly"
        assert "友好" in result["narrative"]

    def test_friendly_above_threshold(self):
        """Trust 1.0 → friendly."""
        state = StateMachine(trust=1.0)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "friendly"

    def test_neutral_when_trust_between_03_and_07(self):
        """Trust 0.5 → neutral."""
        state = StateMachine(trust=0.5)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "neutral"
        assert "中立" in result["narrative"]

    def test_neutral_at_exact_03(self):
        """Trust exactly 0.3 → neutral (>= 0.3)."""
        state = StateMachine(trust=0.3)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "neutral"

    def test_hostile_when_trust_below_03(self):
        """Trust 0.29 → hostile."""
        state = StateMachine(trust=0.29)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "hostile"
        assert "敌对" in result["narrative"]

    def test_hostile_at_zero(self):
        """Trust 0.0 → hostile."""
        state = StateMachine(trust=0.0)
        result = resolve_trigger("npc_reaction", _make_character(), state)
        assert result["outcome"] == "hostile"

    def test_no_state_change(self):
        """NPC reaction never mutates the state machine."""
        state = StateMachine(trust=0.5)
        resolve_trigger("npc_reaction", _make_character(), state)
        assert state.get_state() == {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


class TestDiscovery:
    """``trigger_type="discovery"`` — skill check against character skill."""

    def test_success_with_matching_skill(self, monkeypatch):
        """Roll <= skill value → success."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([40], 40))
        char = _make_character(skills=[{"name": "侦查", "value": 60}])
        result = resolve_trigger("discovery", char, StateMachine())
        assert result["outcome"] == "success"
        assert result["narrative"] == "发现了线索"

    def test_failure_with_matching_skill(self, monkeypatch):
        """Roll > skill value → failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([90], 90))
        char = _make_character(skills=[{"name": "侦查", "value": 60}])
        result = resolve_trigger("discovery", char, StateMachine())
        assert result["outcome"] == "failure"
        assert result["narrative"] == "仔细观察后发现了..."

    def test_default_skill_when_no_skills(self, monkeypatch):
        """Character with no skills uses default value 50."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([40], 40))
        char = _make_character()  # no skills
        result = resolve_trigger("discovery", char, StateMachine())
        assert result["outcome"] == "success"

    def test_default_skill_failure(self, monkeypatch):
        """Default skill 50, roll 51 → failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([51], 51))
        char = _make_character()
        result = resolve_trigger("discovery", char, StateMachine())
        assert result["outcome"] == "failure"

    def test_custom_skill_name(self, monkeypatch):
        """Match a non-default skill name from context."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([30], 30))
        char = _make_character(skills=[{"name": "潜行", "value": 80}])
        result = resolve_trigger(
            "discovery", char, StateMachine(), {"skill_name": "潜行"}
        )
        assert result["outcome"] == "success"

    def test_skill_not_found_falls_back_to_default(self, monkeypatch):
        """Requested skill doesn't exist on character → default 50, roll 51 → failure."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([51], 51))
        char = _make_character(skills=[{"name": "开锁", "value": 90}])
        result = resolve_trigger(
            "discovery", char, StateMachine(), {"skill_name": "侦查"}
        )
        assert result["outcome"] == "failure"

    def test_empty_context(self, monkeypatch):
        """Empty context uses defaults (skill_name='侦查', default 50)."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([40], 40))
        char = _make_character()
        result = resolve_trigger("discovery", char, StateMachine(), {})
        assert result["outcome"] == "success"

    def test_custom_narrative(self, monkeypatch):
        """Custom narrative strings from context."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([10], 10))
        char = _make_character(skills=[{"name": "侦查", "value": 60}])
        result = resolve_trigger(
            "discovery",
            char,
            StateMachine(),
            {"narrative_success": "你发现了一个隐藏的开关！", "narrative_failure": "什么也没有"},
        )
        assert result["outcome"] == "success"
        assert result["narrative"] == "你发现了一个隐藏的开关！"

    def test_no_state_change(self, monkeypatch):
        """Discovery never mutates the state machine."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([50], 50))
        state = StateMachine()
        resolve_trigger("discovery", _make_character(), state)
        assert state.get_state() == {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}


# ---------------------------------------------------------------------------
# Unknown trigger type
# ---------------------------------------------------------------------------


class TestUnknownTrigger:
    """Unrecognised trigger_type values."""

    def test_unknown_type(self):
        result = resolve_trigger("unknown", _make_character(), StateMachine())
        assert result["outcome"] == "unknown_trigger"
        assert result["state_changes"] == []
        assert "unknown" in result["narrative"]

    def test_case_insensitivity_on_valid_types(self, monkeypatch):
        """Trigger type matching should be case-insensitive."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([15], 15))
        state = StateMachine()
        result = resolve_trigger("TRAP", _make_character(), state)
        assert result["outcome"] == "success"


# ---------------------------------------------------------------------------
# Context edge cases
# ---------------------------------------------------------------------------


class TestContextEdgeCases:
    """None and empty context handling."""

    def test_context_none_treated_as_empty(self, monkeypatch):
        """None context should be treated as an empty dict."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([12], 12))
        result = resolve_trigger("environment", _make_character(), StateMachine(), None)
        assert result["outcome"] == "success"

    def test_context_none_discovery(self, monkeypatch):
        """None context on discovery uses defaults."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([40], 40))
        result = resolve_trigger("discovery", _make_character(), StateMachine(), None)
        assert result["outcome"] == "success"

    def test_context_not_provided_defaults_none(self, monkeypatch):
        """Omitting context entirely is equivalent to None (default parameter)."""
        monkeypatch.setattr("trpg_agent.check.roll", lambda _: ([12], 12))
        result = resolve_trigger("environment", _make_character(), StateMachine())
        assert result["outcome"] == "success"
