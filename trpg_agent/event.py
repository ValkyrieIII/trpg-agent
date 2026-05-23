"""Event resolution logic — non-dialogue event triggers.

When a player triggers a trap, interacts with the environment, prompts an
NPC reaction, or makes a discovery, the GM uses this module to resolve the
outcome deterministically via checks + state machine rules — no LLM calls.
"""

from __future__ import annotations

from typing import Any

from trpg_agent.check import difficulty_check, skill_check
from trpg_agent.dice import roll


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
    elif trigger_type == "combat":
        attacker_attrs = getattr(character, "attributes", {}) if character else {}
        defender_state = context.get("defender_state") if context else None
        return _resolve_combat(
            state,
            attacker_attributes=attacker_attrs,
            defender_state=defender_state,
        )
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

    On failure, returns ``state_changes=["combat"]`` — caller applies.
    """
    result = difficulty_check(dc=15)
    roll_val = result["roll"]
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": f"陷阱判定: d20 = {roll_val} ≥ DC 15 → 察觉并避开",
            "check_detail": f"d20 = {roll_val} ≥ DC 15",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
    else:
        return {
            "outcome": "failure",
            "state_changes": ["combat"],
            "narrative": f"陷阱判定: d20 = {roll_val} < DC 15 → 触发陷阱受伤",
            "check_detail": f"d20 = {roll_val} < DC 15",
            "damage_dealt": 0,
            "damage_taken": 0,
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
    roll_val = result["roll"]
    total = result["total"]
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": f"环境判定: d20 = {roll_val} → {total} ≥ DC {dc} → {custom_success}",
            "check_detail": f"d20 = {roll_val} ≥ DC {dc}",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
    else:
        return {
            "outcome": "failure",
            "state_changes": [],
            "narrative": f"环境判定: d20 = {roll_val} → {total} < DC {dc} → {custom_failure}",
            "check_detail": f"d20 = {roll_val} < DC {dc}",
            "damage_dealt": 0,
            "damage_taken": 0,
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
            "check_detail": f"信任度={round(trust, 2)} ≥ 0.7",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
    elif trust >= 0.3:
        return {
            "outcome": "neutral",
            "state_changes": [],
            "narrative": "NPC 态度中立，等待进一步行动",
            "check_detail": f"信任度={round(trust, 2)} ≥ 0.3",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
    else:
        return {
            "outcome": "hostile",
            "state_changes": [],
            "narrative": "NPC 态度敌对，随时可能攻击",
            "check_detail": f"信任度={round(trust, 2)} < 0.3",
            "damage_dealt": 0,
            "damage_taken": 0,
        }


def _resolve_combat(
    state: Any,
    attacker_attributes: dict[str, int] | None = None,
    defender_state: Any = None,
) -> dict[str, Any]:
    """Combat trigger — difficulty check DC 12 + damage roll.

    On success: rolls d6 + strength bonus for damage, knocks down defender.
    On failure: defender may counter-attack.

    Parameters
    ----------
    state : StateMachine
        The attacker's state machine (for stamina/trust changes).
    attacker_attributes : dict or None
        Attacker attributes for damage calculation.
    defender_state : StateMachine or None
        The defender's state machine (for damage application).
    """
    result = difficulty_check(dc=12)
    roll_val = result["roll"]
    attacker_attributes = attacker_attributes or {}

    if result["success"]:
        # Damage: d6 + (strength-10)//2
        str_bonus = max(0, (attacker_attributes.get("strength", 10) - 10) // 2)
        _, dmg_roll = roll("1d6")
        damage = dmg_roll + str_bonus

        # Apply damage to defender
        def_status = "alive"
        def_hp_str = ""
        if defender_state is not None:
            def_status = defender_state.take_damage(damage)
            def_hp_str = f"{defender_state.hp}/{defender_state.max_hp}"

        narrative = (
            f"战斗判定: d20 = {roll_val} ≥ DC 12 → 攻击命中"
        )
        if defender_state is not None:
            narrative += f"（d6={dmg_roll}+{str_bonus}={damage}点伤害, HP剩余{def_hp_str}）"
            if def_status == "dead":
                narrative += " —— 目标倒下！"

        return {
            "outcome": "success",
            "state_changes": [],
            "npc_state_changes": ["threatened"],
            "narrative": narrative,
            "check_detail": f"d20 = {roll_val} ≥ DC 12",
            "damage_dealt": damage,
            "damage_taken": 0,
            "def_hp": def_hp_str,
            "target_status": def_status,
        }
    else:
        # Defender counter-attack
        counter_damage = 0
        counter_narrative = ""
        player_hp_str = f"{state.hp}/{state.max_hp}"
        if defender_state is not None:
            _, counter_roll = roll("1d6")
            counter_damage = counter_roll
            state.take_damage(counter_damage)
            player_hp_str = f"{state.hp}/{state.max_hp}"
            counter_narrative = (
                f"｜ 对方反击（d6={counter_roll}），你受到{counter_damage}点伤害"
                f"（HP剩余{player_hp_str}）"
            )

        narrative = f"战斗判定: d20 = {roll_val} < DC 12 → 攻击落空{counter_narrative}"

        return {
            "outcome": "failure",
            "state_changes": ["combat"],
            "narrative": narrative,
            "check_detail": f"d20 = {roll_val} < DC 12",
            "damage_dealt": 0,
            "damage_taken": counter_damage,
            "def_hp": "",
            "target_status": "alive",
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
    roll_val = result["roll"]
    effective = result["effective_skill"]
    if result["success"]:
        return {
            "outcome": "success",
            "state_changes": [],
            "narrative": f"发现判定: d100 = {roll_val} ≤ 技能{effective} → {custom_success}",
            "check_detail": f"d100 = {roll_val} ≤ 技能{effective}",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
    else:
        return {
            "outcome": "failure",
            "state_changes": [],
            "narrative": f"发现判定: d100 = {roll_val} > 技能{effective} → {custom_failure}",
            "check_detail": f"d100 = {roll_val} > 技能{effective}",
            "damage_dealt": 0,
            "damage_taken": 0,
        }
