"""Narrator Agent — pure scene narration, no mechanics.

The Narrator receives all results (checks, NPC dialogue, knowledge search)
and writes the final scene narration + player action suggestions.

The Narrator has ZERO tools and NEVER invokes other agents.
"""

from agents import Agent

from trpg_agent.agent_config import DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Narrator system prompt
# ---------------------------------------------------------------------------

NARRATOR_SYSTEM_PROMPT = """\
你是一个TRPG游戏的叙述者，只负责把已经发生的结果写成场景叙述。你不需要做任何判定——所有检定和NPC决策都已经完成了。

## 你的输入
你会收到：
- 当前场景信息（时间、天气、在场NPC）
- 玩家做了什么
- 检定结果（如果做了检定）——由系统返回，你不需要计算
- NPC 的回应（如果NPC说话了）
- 相关知识（如果检索了）

## 叙事规则
- 每段叙述不超过 150 字
- 只描述角色动作和场景环境，绝对不写骰子数值
- 每次叙述引入新的推进元素：NPC的反应、环境变化、新线索的浮现
- 玩家是驱动故事的人——不要替玩家做决定
- 不要替玩家角色说话、做决定或执行动作
- 不要描述玩家角色的内心感受

## 输出格式（最高优先级）
你必须且只能以如下 JSON 格式回复：
{"narration": "场景叙述", "suggestions": ["建议1", "建议2", "建议3"]}

严格规则：
- narration: 只能包含场景叙述。禁止写入建议、提示、编号列表
- suggestions: 始终恰好 3 个，简短且基于当前场景
- 确保 JSON 合法。不要 markdown 代码块、不要额外文字
- 建议保持"此时此地"——基于玩家当前能看到、能做的
"""


# ---------------------------------------------------------------------------
# Narrator Agent factory
# ---------------------------------------------------------------------------

def create_narrator_agent() -> Agent:
    """Create the Narrator Agent for scene narration.

    The Narrator has zero tools — it's a pure LLM generation step
    that takes all collected results and writes the narrative output.
    """
    return Agent(
        name="Narrator",
        instructions=NARRATOR_SYSTEM_PROMPT,
        tools=[],  # zero tools — pure generation
        model=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Narrator input builder
# ---------------------------------------------------------------------------

def build_narrator_input(
    player_action: str,
    player_name: str,
    scene_context: str,
    check_results: str = "",
    npc_responses: str = "",
    knowledge_results: str = "",
    world_setting: str = "",
) -> str:
    """Build the input message for the Narrator Agent.

    Parameters
    ----------
    player_action : str
        What the player just did.
    player_name : str
        Player character name.
    scene_context : str
        Current scene (time, weather, location, present NPCs).
    check_results : str
        Results from Judge Agent (dice rolls, checks, combat).
    npc_responses : str
        Responses from NPC Agents (one or more NPCs).
    knowledge_results : str
        Results from knowledge/memory searches.
    world_setting : str
        World description from config.
    """
    parts: list[str] = []

    if world_setting:
        parts.append(f"## 世界设定\n{world_setting}")

    parts.append(f"## 当前场景\n{scene_context}")
    parts.append(f"\n## 玩家 {player_name}\n行动: {player_action}")

    if check_results:
        parts.append(f"\n## 检定结果\n{check_results}")

    if npc_responses:
        parts.append(f"\n## NPC 回应\n{npc_responses}")

    if knowledge_results:
        parts.append(f"\n## 相关知识\n{knowledge_results}")

    parts.append(
        "\n请基于以上所有信息，生成场景叙述和3个行动建议。"
        "按照 JSON 格式输出：{\"narration\": \"...\", \"suggestions\": [...]}"
    )

    return "\n".join(parts)
