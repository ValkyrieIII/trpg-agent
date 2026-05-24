"""Structured status events for real-time Agent activity streaming via SSE.

StatusEvent objects flow through process_streaming() → api_server.py → SSE → frontend.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal

from agents import RunHooks

# ---------------------------------------------------------------------------
# StatusEvent
# ---------------------------------------------------------------------------

StatusEventType = Literal[
    "thinking_start",
    "thinking_end",
    "tool_call_start",
    "tool_call_end",
    "npc_message",
]


@dataclass
class StatusEvent:
    type: StatusEventType
    tool: str | None = None
    display: str | None = None          # human-readable e.g. "投骰子(2d6)"
    result: str | None = None           # tool result preview (tool_call_end)
    args_json: str | None = None        # raw arguments JSON
    npc_name: str | None = None         # NPC name for npc_message events
    npc_text: str | None = None         # NPC dialogue text for npc_message events


# ---------------------------------------------------------------------------
# Tool display names (Chinese)
# ---------------------------------------------------------------------------

TOOL_DISPLAY_NAMES: dict[str, str] = {
    "roll_dice":            "投骰子",
    "difficulty_check":     "难度检定",
    "skill_check":          "技能检定",
    "combat_attack":        "战斗攻击",
    "get_player_state":     "查询状态",
    "get_npc_state":        "查询NPC",
    "create_npc":           "创建NPC",
    "invoke_npc":           "NPC对话",
    "remove_npc":           "移除NPC",
    "set_scene":            "场景设定",
    "game_over":            "游戏结束",
    "search_knowledge":     "知识检索",
    "search_memory":        "记忆检索",
}


def _format_args(args_json: str) -> str:
    """Format tool arguments JSON into a short human-readable string.

    Example: '{"expr":"2d6"}' → '2d6'  |  '{"target":"老酒保"}' → 'target=老酒保'
    """
    try:
        parsed = json.loads(args_json)
    except (json.JSONDecodeError, TypeError):
        return args_json[:40]

    parsed.pop("ctx", None)

    if not parsed:
        return ""

    if len(parsed) == 1:
        val = list(parsed.values())[0]
        return str(val)[:40]

    parts = [f"{k}={v}" for k, v in parsed.items()]
    return ", ".join(parts)[:60]


def make_tool_event(event_type: StatusEventType, tool_name: str,
                    args_json: str = "", result: str = "") -> StatusEvent:
    """Build a StatusEvent for a tool call with human-readable display."""
    display_name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    if event_type == "tool_call_start":
        formatted = _format_args(args_json) if args_json else ""
        display = f"{display_name}({formatted})" if formatted else display_name
    else:
        display = f"{display_name} ✓"
    return StatusEvent(
        type=event_type,
        tool=tool_name,
        display=display,
        result=result[:120] if result else None,
        args_json=args_json,
    )


# ---------------------------------------------------------------------------
# AgentStatusHooks — for Runner.run() (non-streaming)
# ---------------------------------------------------------------------------

class AgentStatusHooks(RunHooks):
    """RunHooks that push lifecycle events to an asyncio.Queue for SSE streaming."""

    def __init__(self, event_queue: "asyncio.Queue"):
        self._queue = event_queue

    async def on_llm_start(self, context, agent, system_prompt=None, input_items=None):
        await self._queue.put(StatusEvent(type="thinking_start"))

    async def on_llm_end(self, context, agent, response):
        await self._queue.put(StatusEvent(type="thinking_end"))

    async def on_tool_start(self, context, agent, tool):
        tool_name = getattr(context, "tool_name", None) or getattr(tool, "qualified_name", "?")
        args_json = getattr(context, "tool_arguments", "{}") or "{}"
        await self._queue.put(make_tool_event("tool_call_start", tool_name, args_json))

    async def on_tool_end(self, context, agent, tool, result: str):
        tool_name = getattr(context, "tool_name", None) or getattr(tool, "qualified_name", "?")
        await self._queue.put(make_tool_event("tool_call_end", tool_name, result=result))

        # Emit npc_message event when invoke_npc returns, so frontend can
        # render NPC dialogue as a separate chat bubble (group-chat style).
        if tool_name == "invoke_npc" and result:
            npc_name, npc_text = _parse_npc_result(result)
            if npc_name and npc_text:
                await self._queue.put(StatusEvent(
                    type="npc_message",
                    npc_name=npc_name,
                    npc_text=npc_text,
                ))


def _parse_npc_result(result: str) -> tuple[str, str]:
    """Parse invoke_npc tool result into (npc_name, npc_text).

    The tool returns format:  'NPC名: "对话内容"'
    """
    import re
    match = re.match(r'^(.+?):\s*[""「](.+?)[""」]$', result.strip(), re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    # Fallback: split on first colon
    if ':' in result:
        name, text = result.split(':', 1)
        return name.strip(), text.strip().strip('"').strip('"').strip('「」')
    return "", result
