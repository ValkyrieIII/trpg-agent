"""Orchestrator Agent — pure routing, zero narrative.

The Orchestrator analyzes player input and routes to specialized sub-agents:
Judge (game mechanics), NPC Agents (character dialogue), Narrator (scene writing).

It NEVER writes narrative, descriptions, or suggestions itself.
"""

from agents import Agent

from trpg_agent.agent_config import DEFAULT_MODEL

# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """\
你是TRPG游戏的流程调度器。你只负责分析玩家输入并路由到正确的子Agent，**不写任何叙事文字**。

## 你的职责
1. 分析玩家输入：玩家想做什么？
2. 决定需要哪些子Agent：需要检定吗？需要NPC回应吗？需要查知识吗？
3. 依次调用子Agent，收集结果
4. 最后调用 invoke_narrator 让叙述者写出场景叙事

## 路由规则

### 需要 Judge（裁判）的情况
- 玩家做了不确定结果的行动（攻击、攀爬、撬锁、躲闪等）
- 使用 invoke_judge(action_description) 调用
- 注意：如果玩家同时在和NPC说话，invoke_judge 可以和 invoke_npc 并行调用

### 需要 NPC 回应的情况
- 玩家明确提到NPC名称并与之互动 → invoke_npc(name, prompt)
- 玩家同时对多个NPC说话 → invoke_npcs([{"name":..., "prompt":...}, ...])
- 如果只是走过路过或描述场景，也需要广播事件给在场NPC → broadcast_event(event_description)

### 需要搜索知识/记忆的情况
- 玩家询问世界观设定 → search_knowledge(query)
- 玩家问"之前发生了什么" → search_memory(query)
- Orchestrator引入新地点/物品/势力 → 先 search_knowledge 确认

### 场景管理
- 玩家移动到新场景 → set_scene(location, present_npcs, time_of_day, weather)
- NPC离开 → remove_npc(name)
- 新NPC出现且有对话潜力 → create_npc(name, core, personality_tone)

### 异常状态
- HP ≤ 0 或叙事死亡 → game_over(cause)

## 执行顺序
1. 先并行调用：invoke_judge + invoke_npc/broadcast_event + search_knowledge/search_memory
2. 收集所有结果
3. 最后调用 invoke_narrator 生成场景叙事和行动建议

## 你的输出
你只是一个路由器。你的最终输出应该是 invoke_narrator 返回的 JSON 结果。
不要自己写叙事，不要自己写建议。把创作的工作交给 Narrator。

## 当前世界状态
时间: {time_of_day} | 天气: {weather}
场景NPC: {scene_npcs}
"""


# ---------------------------------------------------------------------------
# Orchestrator input builder
# ---------------------------------------------------------------------------

def build_orchestrator_input(
    player_input: str,
    player_name: str,
    scene_context: str,
    history_text: str = "",
    last_gm_response: str = "",
) -> str:
    """Build the input message for the Orchestrator.

    Parameters
    ----------
    player_input : str
        What the player just typed.
    player_name : str
        Player character name.
    scene_context : str
        Current scene summary (time, weather, NPCs, location).
    history_text : str
        Recent conversation history (last 3 exchanges).
    last_gm_response : str
        Previous turn's GM response for continuity.
    """
    parts: list[str] = []

    parts.append(f"## 当前场景\n{scene_context}")

    if history_text:
        parts.append(f"\n## 最近对话\n{history_text}")

    if last_gm_response:
        parts.append(f"\n## 上一轮GM回应\n{last_gm_response}")

    parts.append(f"\n## 玩家 {player_name}\n{player_input}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Orchestrator Agent factory
# ---------------------------------------------------------------------------

def create_orchestrator_agent(tools: list) -> Agent:
    """Create the Orchestrator Agent for game flow routing.

    Parameters
    ----------
    tools : list
        All tools the Orchestrator can call: invoke_judge, invoke_narrator,
        invoke_npc, invoke_npcs, broadcast_event, search_memory,
        search_knowledge, create_npc, remove_npc, set_scene,
        get_player_state, get_npc_state, game_over.
    """
    return Agent(
        name="Orchestrator",
        instructions="",  # filled dynamically via build_orchestrator_instructions
        tools=tools,
        model=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# Orchestrator instruction builder
# ---------------------------------------------------------------------------

def build_orchestrator_instructions(
    time_of_day: str = "",
    weather: str = "",
    scene_npcs: str = "",
) -> str:
    """Render the Orchestrator system prompt with current state placeholders."""
    return ORCHESTRATOR_SYSTEM_PROMPT.format(
        time_of_day=time_of_day or "未知",
        weather=weather or "未知",
        scene_npcs=scene_npcs or "无",
    )
