"""Judge Agent — pure game-mechanics judgment, no narrative.

The Judge receives a player action and decides:
- Does this need a check? What type?
- What DC / skill / modifier should be used?
- Execute the check and interpret the result.

The Judge NEVER writes narrative, scene description, or suggestions.
"""

from agents import Agent

from trpg_agent.agent_config import DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Judge system prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
你是一个TRPG游戏裁判，只负责游戏机制判定，不写任何叙事。

## 你的职责
根据玩家行动描述，判断是否需要检定，如果需要，选择正确的检定方式并执行。

## 检定工具选择规则
- **纯扮演动作**（微笑、点头、叹气、走路、坐下、说话）→ 无需检定，直接返回 "无需检定"
- **不确定结果的行动**（爬墙、撬锁、躲闪、忍耐）→ difficulty_check(DC, 修正值)
- **涉及角色技能的行动**（观察、潜行、交涉、追踪）→ skill_check(技能名, 修正值)
- **攻击 NPC** → combat_attack(目标名)
- **纯投骰子看运气**（"扔个d6""看看骰运"）→ roll_dice(表达式)
- **不需要检定**

## DC 参考
- 简单: 8 (爬矮墙、说服友好NPC)
- 普通: 12 (爬石墙、撬普通锁、躲闪攻击)
- 困难: 16 (爬光滑墙壁、撬复杂锁、在暴风雪中行动)
- 极难: 20 (徒手攀岩、潜入重兵把守的城堡)

## 修正值参考
- 角色属性对应修正: (属性值 - 10) // 2
- 有利条件: +2~+5
- 不利条件: -2~-5

## 战斗规则
- 攻击判定: d20 ≥ DC12 命中，d6+力量修正 伤害
- 必须明确指定攻击目标 NPC 名称

## 输出格式
只需输出检定结果，不要任何叙事：
- 无需检定时: "无需检定——纯扮演动作"
- 需要检定时: 调用对应工具，返回工具结果（如 "d20=15 ≥ DC12 → 成功"）
- 战斗时: 返回命中结果和伤害值
"""


# ---------------------------------------------------------------------------
# Judge Agent factory
# ---------------------------------------------------------------------------

def create_judge_agent(tools: list | None = None) -> Agent:
    """Create the Judge Agent for game-mechanics judgment.

    Parameters
    ----------
    tools : list, optional
        The check/dice tools the Judge can call.
        Defaults to importing from trpg_agent.tools.
    """
    if tools is None:
        from trpg_agent.tools import (
            roll_dice,
            difficulty_check,
            skill_check,
            combat_attack,
        )
        tools = [roll_dice, difficulty_check, skill_check, combat_attack]

    return Agent(
        name="Judge",
        instructions=JUDGE_SYSTEM_PROMPT,
        tools=tools,
        model=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Judge input builder
# ---------------------------------------------------------------------------

def build_judge_input(
    player_action: str,
    player_name: str,
    player_skills: list[dict],
    player_attributes: dict[str, int],
    scene_context: str,
    hp_info: str,
) -> str:
    """Build the input message for the Judge Agent.

    Parameters
    ----------
    player_action : str
        What the player is trying to do.
    player_name : str
        Player character name.
    player_skills : list[dict]
        Character skills, e.g. [{"name": "观察", "value": 50}, ...].
    player_attributes : dict
        Character attributes, e.g. {"力量": 12, "敏捷": 10, ...}.
    scene_context : str
        Current scene description (time, weather, location, present NPCs).
    hp_info : str
        Current HP status, e.g. "HP 10/12".
    """
    parts: list[str] = []

    parts.append("## 场景")
    parts.append(scene_context)

    parts.append(f"\n## 玩家 {player_name}")
    parts.append(hp_info)

    # Skills
    if player_skills:
        skill_lines = [f"- {s['name']}: {s['value']}" for s in player_skills]
        parts.append("技能:" + "\n".join(skill_lines))

    # Attributes
    if player_attributes:
        attr_lines = [f"- {k}: {v}" for k, v in player_attributes.items()]
        parts.append("属性:" + "\n".join(attr_lines))

    parts.append(f"\n## 玩家行动\n{player_action}")
    parts.append("\n请判断这个行动是否需要检定。如果需要，执行对应的检定工具。")

    return "\n".join(parts)
