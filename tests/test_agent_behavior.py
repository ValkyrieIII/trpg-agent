"""Agent behavior regression tests — prompt validation, tool selection, response parsing.

These tests focus on regression safety: verifying that prompt templates,
tool functions, and response parsing behave correctly without requiring
a live LLM connection. Mock LLMs are used where needed.
"""

import json
import re
import time
from unittest.mock import MagicMock, patch

import pytest

from trpg_agent.agent_config import (
    GameContext,
    build_gm_instructions,
    _GM_SYSTEM_PROMPT,
)
from trpg_agent.event_stream import TOOL_DISPLAY_NAMES, make_tool_event, StatusEvent


# ===================================================================
# Prompt template tests
# ===================================================================


class TestGMPrompt:
    """Verify the GM system prompt template is well-formed."""

    def test_prompt_contains_required_sections(self):
        """The GM prompt must include all critical sections."""
        required = [
            "## 工具使用纪律",
            "## 硬性规则",
            "## 流程规则",
            "## 叙事节奏",
            "## 叙事风格",
            "## 输出格式",
            "## 内部推理过程",
            "suggestions",
            "narration",
        ]
        for keyword in required:
            assert keyword in _GM_SYSTEM_PROMPT, f"Prompt missing section: {keyword}"

    def test_prompt_mentions_invoke_npcs(self):
        """The prompt should mention the parallel NPC tool."""
        assert "invoke_npcs" in _GM_SYSTEM_PROMPT

    def test_prompt_forbids_moral_judgment(self):
        """The prompt must instruct the GM not to make moral judgments."""
        assert "禁止道德评判" in _GM_SYSTEM_PROMPT

    def test_prompt_has_output_format_constraint(self):
        """The prompt must require JSON output with narration and suggestions."""
        assert '"narration"' in _GM_SYSTEM_PROMPT
        assert '"suggestions"' in _GM_SYSTEM_PROMPT
        assert 'json' in _GM_SYSTEM_PROMPT.lower()

    def test_build_gm_instructions_fills_placeholders(self):
        """build_gm_instructions() should fill all {placeholders}."""
        result = build_gm_instructions(
            world_setting="TestWorld",
            player_name="TestPlayer",
            player_card="TestCard",
            time_of_day="正午",
            weather="晴",
            scene_npcs="老酒保",
        )
        assert "{world_setting}" not in result
        assert "{player_name}" not in result
        assert "TestWorld" in result
        assert "TestPlayer" in result
        assert "TestCard" in result
        assert "正午" in result

    def test_prompt_includes_cot_reasoning_steps(self):
        """The prompt should include the internal reasoning process (CoT)."""
        assert "理解意图" in _GM_SYSTEM_PROMPT
        assert "评估状态" in _GM_SYSTEM_PROMPT
        assert "工具决策" in _GM_SYSTEM_PROMPT
        assert "输出自检" in _GM_SYSTEM_PROMPT


# ===================================================================
# Token estimation tests
# ===================================================================


class TestTokenEstimation:
    """Test the _estimate_tokens static method."""

    def test_pure_chinese(self):
        from trpg_agent.game_master import GameMaster
        # 4 Chinese chars ~= 2 tokens
        tokens = GameMaster._estimate_tokens("你好世界")
        assert 1 <= tokens <= 4

    def test_pure_english(self):
        from trpg_agent.game_master import GameMaster
        # "hello world" (11 chars) ~= 2-3 tokens
        tokens = GameMaster._estimate_tokens("hello world")
        assert 1 <= tokens <= 5

    def test_empty_string(self):
        from trpg_agent.game_master import GameMaster
        assert GameMaster._estimate_tokens("") == 0

    def test_mixed_text(self):
        from trpg_agent.game_master import GameMaster
        tokens = GameMaster._estimate_tokens("你好world")
        assert tokens >= 1


# ===================================================================
# JSON response parsing tests
# ===================================================================


class TestResponseParsing:
    """Test that the GM's JSON response parsing handles various formats.

    These tests verify the extraction logic in process() and process_streaming()
    by testing the core patterns directly.
    """

    @staticmethod
    def _parse_gm_response(text: str) -> dict:
        """Replicate the JSON extraction logic from process() (lines ~659-695).

        This is the actual logic used in game_master.py to parse GM responses.
        """
        # 1. Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, dict) and "narration" in result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # 2. Strip markdown code fences
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        try:
            result = json.loads(cleaned)
            if isinstance(result, dict) and "narration" in result:
                return result
        except (json.JSONDecodeError, TypeError):
            pass

        # 3. Find JSON object with "narration" key
        match = re.search(r'\{[^}]*"narration"[^}]*\}', text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group())
                if isinstance(result, dict) and "narration" in result:
                    return result
            except json.JSONDecodeError:
                pass

        # 4. Fallback: extract numbered suggestions from text
        narration = text
        suggestions = []
        suggestion_match = re.findall(r'^\d+\.\s+(.+)$', text, re.MULTILINE)
        if suggestion_match:
            suggestions = suggestion_match[:3]
            # Try to get narration before the first numbered item
            first_num = re.search(r'\n\d+\.\s+', text)
            if first_num:
                narration = text[:first_num.start()].strip()

        return {"narration": narration, "suggestions": suggestions}

    def test_parse_valid_json(self):
        result = self._parse_gm_response(
            '{"narration": "门打开了。", "suggestions": ["进去", "离开", "敲门"]}'
        )
        assert result["narration"] == "门打开了。"
        assert len(result["suggestions"]) == 3

    def test_parse_json_with_code_block(self):
        result = self._parse_gm_response(
            '```json\n{"narration": "风吹过树林。", "suggestions": ["前行", "返回", "观察"]}\n```'
        )
        assert result["narration"] == "风吹过树林。"
        assert len(result["suggestions"]) == 3

    def test_parse_json_with_extra_text(self):
        result = self._parse_gm_response(
            'I think the scene is ready.\n\n{"narration": "酒馆里很安静。", "suggestions": ["点酒", "离开", "交谈"]}\n\nThat should work.'
        )
        assert "酒馆" in result["narration"]
        assert len(result["suggestions"]) == 3

    def test_fallback_extracts_numbered_suggestions(self):
        """When JSON is completely broken, extract numbered lines as suggestions."""
        text = "酒馆很热闹。\n1. 走向吧台\n2. 观察角落\n3. 和老者搭话"
        result = self._parse_gm_response(text)
        assert len(result["suggestions"]) >= 1

    def test_narration_is_not_empty(self):
        result = self._parse_gm_response(
            '{"narration": "阳光洒在石板上。", "suggestions": ["前进", "等待", "搜索"]}'
        )
        assert result["narration"] != ""


# ===================================================================
# Tool availability tests
# ===================================================================


class TestToolAvailability:
    """Verify all expected tools are importable and properly decorated."""

    def test_all_tools_importable(self):
        """Every tool should be importable from trpg_agent.tools."""
        tool_names = [
            "roll_dice", "difficulty_check", "skill_check", "combat_attack",
            "get_player_state", "get_npc_state",
            "create_npc", "invoke_npc", "invoke_npcs",
            "remove_npc", "set_scene", "game_over",
            "search_knowledge", "search_memory",
        ]
        for name in tool_names:
            __import__("trpg_agent.tools", fromlist=[name])

    def test_core_tools_present(self):
        """Verify the 14 tools that the GM agent depends on exist."""
        from trpg_agent import tools as tmod
        expected = [
            "roll_dice", "difficulty_check", "skill_check", "combat_attack",
            "get_player_state", "get_npc_state",
            "create_npc", "invoke_npc", "invoke_npcs",
            "remove_npc", "set_scene", "game_over",
            "search_knowledge", "search_memory", "clear_tool_cache",
        ]
        for name in expected:
            assert hasattr(tmod, name), f"tools.py missing: {name}"


# ===================================================================
# Tool cache tests
# ===================================================================


class TestToolCache:
    """Tests for the tool result caching mechanism."""

    def test_cache_returns_cached_value(self):
        from trpg_agent.tools import _cached, clear_tool_cache
        clear_tool_cache()

        call_count = [0]

        def producer():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        r1 = _cached("test_tool", "key1", producer)
        r2 = _cached("test_tool", "key1", producer)
        assert r1 == r2 == "result-1"
        assert call_count[0] == 1  # producer only called once

    def test_cache_different_keys(self):
        from trpg_agent.tools import _cached, clear_tool_cache
        clear_tool_cache()

        call_count = [0]

        def producer():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        r1 = _cached("test_tool", "key_a", producer)
        r2 = _cached("test_tool", "key_b", producer)
        assert r1 != r2
        assert call_count[0] == 2

    def test_clear_cache_resets(self):
        from trpg_agent.tools import _cached, clear_tool_cache
        clear_tool_cache()

        call_count = [0]

        def producer():
            call_count[0] += 1
            return f"result-{call_count[0]}"

        _cached("test_tool", "key1", producer)
        clear_tool_cache()
        _cached("test_tool", "key1", producer)
        assert call_count[0] == 2  # producer called again after clear


# ===================================================================
# Event stream tests
# ===================================================================


class TestEventStream:
    """Tests for SSE event stream infrastructure."""

    def test_all_tools_have_display_names(self):
        """Every tool used by the GM should have a display name."""
        known_tools = [
            "roll_dice", "difficulty_check", "skill_check", "combat_attack",
            "get_player_state", "get_npc_state",
            "create_npc", "invoke_npc", "invoke_npcs",
            "remove_npc", "set_scene", "game_over",
            "search_knowledge", "search_memory",
        ]
        for tool in known_tools:
            assert tool in TOOL_DISPLAY_NAMES, f"Missing display name for: {tool}"

    def test_make_tool_event_start(self):
        event = make_tool_event("tool_call_start", "roll_dice", '{"expression":"2d6"}')
        assert event.type == "tool_call_start"
        assert event.tool == "roll_dice"
        assert event.display in ("投骰子(2d6)", "投骰子")
        assert event.result in (None, "")  # start events have no result yet

    def test_make_tool_event_end(self):
        event = make_tool_event("tool_call_end", "difficulty_check", '{"dc":12}', "d20=15 ≥ DC12 → 成功")
        assert event.type == "tool_call_end"
        assert event.tool == "difficulty_check"
        assert "成功" in event.result

    def test_status_event_fields(self):
        event = StatusEvent(
            type="thinking_start",
            tool="",
            display="",
            result="",
            npc_name=None,
            npc_text=None,
        )
        assert event.type == "thinking_start"
        assert event.tool == ""


# ===================================================================
# GameContext tests
# ===================================================================


class TestGameContext:
    """Tests for the GameContext dataclass."""

    def test_default_values(self):
        ctx = GameContext(
            player_state=None,
            npc_store=None,
            memory=None,
            knowledge=None,
            scene_npcs=[],
            time_of_day="正午",
            weather="晴",
            player_name="TestPlayer",
            player_card="TestCard",
        )
        assert ctx.time_of_day == "正午"
        assert ctx.debug is False
        assert ctx.debug_log == []
        assert ctx.game_over is False
        assert ctx.npc_agents == {}

    def test_tool_cache_module_available(self):
        """tools.py must export clear_tool_cache and _cached functions."""
        from trpg_agent.tools import clear_tool_cache, _cached
        assert callable(clear_tool_cache)
        assert callable(_cached)
