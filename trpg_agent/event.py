"""Event resolution logic — non-dialogue event triggers.

When a player triggers a trap, interacts with the environment, prompts an
NPC reaction, or makes a discovery, the GM uses this module to resolve the
outcome deterministically via checks + state machine rules — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from trpg_agent.check import difficulty_check, skill_check


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def resolve_trigger(
    trigger_type: str,
    character: Any,
    state: Any,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve a non-dialogue event trigger.

    Parameters
    ----------
    trigger_type : str
        One of ``"trap"``, ``"environment"``, ``"npc_reaction"``,
        ``"discovery"``.
    character : Character
        A :class:`~trpg_agent.character.Character` instance (used for
        skill values in discovery).
    state : StateMachine
        An active :class:`~trpg_agent.state.StateMachine` instance.
    context : dict or None
        Optional extra context.  Supported keys vary by trigger type:

        - ``environment``: ``dc`` (int, default 12),
          ``narrative_success`` / ``narrative_failure`` (str)
        - ``discovery``: ``skill_name`` (str, default ``"侦查"``),
          ``modifier`` (int, default 0),
          ``narrative_success`` / ``narrative_failure`` (str)

    Returns
    -------
    dict
        A dictionary with keys:

        - ``outcome`` (str): one of ``"success"``, ``"failure"``,
          ``"friendly"``, ``"neutral"``, ``"hostile"``,
          ``"unknown_trigger"``.
        - ``state_changes`` (list of str): trigger names that were applied
          to the state machine (e.g. ``["combat"]``).
        - ``narrative`` (str): human-readable result description.
    """
    if context is None:
        context = {}

    trigger_type = trigger_type.lower()

    if trigger_type == "trap":
        return _resolve_trap(state)
    elif trigger_type == "environment":
        return _resolve_environment(context)
    elif trigger_type == "npc_reaction":
        return _resolve_npc_reaction(state)
    elif trigger_type == "discovery":
        return _resolve_discovery(character, context)
    else:
        return {
            "outcome": "unknown_trigger",
            "state_changes": [],
            "narrative": f"未知的事件类型：{trigger_type}",
        }


# ---------------------------------------------------------------------------
# Internal resolvers
# ---------------------------------------------------------------------------


def _resolve_trap(state: Any) -> dict[str, Any]:
    """Trap trigger — difficulty check DC 15.

    On failure, applies ``"combat"`` to the state machine (reduces stamina).
    """
    result = difficulty_check(dc=15)
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": "察觉并避开",
        }
    else:
        state.apply("combat")
        return {
            "outcome": "failure",
            "state_changes": ["combat"],
            "narrative": "触发陷阱受伤",
        }


def _resolve_environment(context: dict[str, Any]) -> dict[str, Any]:
    """Environment trigger — difficulty check with configurable DC.

    Context keys
    ------------
    dc : int
        Difficulty class (default 12).
    narrative_success : str
        Narrative on success (default ``"环境判定成功"``).
    narrative_failure : str
        Narrative on failure (default ``"环境判定失败"``).
    """
    dc = context.get("dc", 12)
    custom_success = context.get("narrative_success", "环境判定成功")
    custom_failure = context.get("narrative_failure", "环境判定失败")

    result = difficulty_check(dc=dc)
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": custom_success,
        }
    else:
        return {
            "outcome": "failure",
            "state_changes": [],
            "narrative": custom_failure,
        }


def _resolve_npc_reaction(state: Any) -> dict[str, Any]:
    """NPC reaction based on trust threshold.

    - trust >= 0.7 → ``"friendly"``
    - trust >= 0.3 → ``"neutral"``
    - trust  < 0.3 → ``"hostile"``
    """
    trust = state.get_state()["trust"]
    if trust >= 0.7:
        return {
            "outcome": "friendly",
            "state_changes": [],
            "narrative": "NPC 态度友好，愿意提供帮助",
        }
    elif trust >= 0.3:
        return {
            "outcome": "neutral",
            "state_changes": [],
            "narrative": "NPC 态度中立，等待进一步行动",
        }
    else:
        return {
            "outcome": "hostile",
            "state_changes": [],
            "narrative": "NPC 态度敌对，随时可能攻击",
        }


def _resolve_discovery(character: Any, context: dict[str, Any]) -> dict[str, Any]:
    """Discovery trigger — skill check against a matching skill.

    If the character has no skill matching the requested name, a default
    value of 50 is used.

    Context keys
    ------------
    skill_name : str
        Name of the skill to check (default ``"侦查"``).
    modifier : int
        Modifier applied to the skill value (default 0).
    narrative_success : str
        Narrative on success (default ``"发现了线索"``).
    narrative_failure : str
        Narrative on failure (default ``"仔细观察后发现了..."``).
    """
    skill_name = context.get("skill_name", "侦查")
    modifier = context.get("modifier", 0)
    custom_success = context.get("narrative_success", "发现了线索")
    custom_failure = context.get("narrative_failure", "仔细观察后发现了...")

    # Find matching skill value; default to 50 if not found
    skill_value = 50
    for s in getattr(character, "skills", []):
        if isinstance(s, dict) and s.get("name", "").lower() == skill_name.lower():
            skill_value = int(s.get("value", 50))
            break

    result = skill_check(skill_value, modifier=modifier)
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": custom_success,
        }
    else:
        return {
            "outcome": "failure",
            "state_changes": [],
            "narrative": custom_failure,
        }
