"""NPC Agent factory — each NPC becomes an independent Agent with its own reasoning.

Phase 2: invoke_npc routes to a real NPC Agent that independently decides how to respond.
"""

from agents import Agent, RunContextWrapper, function_tool

from trpg_agent.agent_config import GameContext, DEFAULT_MODEL


# ---------------------------------------------------------------------------
# NPC memory search tool (for NPC Agents)
# ---------------------------------------------------------------------------

@function_tool
def search_npc_memory(ctx: RunContextWrapper[dict], query: str) -> str:
    """Search this NPC's personal memories of past interactions with the player."""
    npc_context = ctx.context
    memory_store = npc_context.get("memory")
    npc_name = npc_context.get("npc_name", "")
    if not memory_store or not npc_name:
        return "无法检索记忆"

    try:
        memories = memory_store.npc_search(npc_name, query, n=3)
        if not memories:
            return "没有找到相关记忆"
        return "\n".join(f"- {m['content']}" for m in memories)
    except Exception as e:
        return f"记忆检索失败: {e}"


# ---------------------------------------------------------------------------
# NPC Agent factory
# ---------------------------------------------------------------------------

def create_npc_agent(
    name: str,
    personality_prompt: str,
    npc_store,
) -> Agent:
    """Create an independent Agent instance for an NPC.

    The Agent's instructions = static character settings (personality, speech style).
    Dynamic state (emotion, stamina, scene, relevant memories) is injected
    via the input message at call time.

    Parameters
    ----------
    name : str
        NPC name (used as Agent name and for memory lookups).
    personality_prompt : str
        Static personality prompt from NPCCharacter.build_personality_prompt().
    npc_store :
        NPCStore instance for retrieving NPC state and history.
    """
    return Agent(
        name=f"NPC_{name}",
        instructions=personality_prompt,
        tools=[search_npc_memory],
        model=DEFAULT_MODEL,
    )


# ---------------------------------------------------------------------------
# NPC input builder
# ---------------------------------------------------------------------------

def build_npc_input(
    prompt: str,
    npc_context: dict,
    npc_history: list[dict],
) -> str:
    """Build the input message for an NPC Agent.

    Parameters
    ----------
    prompt : str
        What the player said or the situation the NPC should respond to.
    npc_context : dict
        Dynamic context with keys: current_emotion, current_stamina, current_hp,
        scene, npc_name, memory.
    npc_history : list[dict]
        Recent conversation history with this NPC.
    """
    parts = []

    # Current state
    parts.append("## 你当前的状态")
    parts.append(f"情绪: {npc_context.get('current_emotion', 'calm')}")
    parts.append(f"体力: {npc_context.get('current_stamina', 'fresh')}")
    parts.append(f"HP: {npc_context.get('current_hp', '?/?')}")

    # Scene context
    scene = npc_context.get("scene", "")
    if scene:
        parts.append(f"\n## 当前场景\n{scene}")

    # Relevant memories
    memories = npc_context.get("relevant_memories", [])
    if memories:
        mem_text = "\n".join(
            f"- {m['content']}" if isinstance(m, dict) else f"- {m}"
            for m in memories
        )
        parts.append(f"\n## 你记得的过往交集\n{mem_text}")

    # Recent conversation history
    if npc_history:
        recent = npc_history[-6:]  # last 3 exchanges
        history_text = "\n".join(
            f"{'玩家' if h['role'] == 'user' else '你'}: {h['content'][:200]}"
            for h in recent
        )
        parts.append(f"\n## 最近的对话\n{history_text}")

    # The actual prompt
    parts.append(f"\n## 当前情境\n{prompt}")
    parts.append(
        "\n请以第一人称扮演你的角色，根据你的性格、当前情绪和记忆做出回应。"
        "你可以说话、做动作，也可以选择沉默或不理睬。只需输出你的回应（对话+动作描述），不需要JSON格式。"
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# NPC event input builder (broadcast + tick)
# ---------------------------------------------------------------------------

def build_npc_event_input(
    event: str,
    npc_name: str,
    npc_state: dict,
    scene: str,
    memory=None,
) -> str:
    """Build input for an NPC deciding whether/how to respond to an event.

    Unlike build_npc_input which is for explicit player→NPC dialogue,
    this is for broadcast/tick events where the NPC autonomously chooses
    whether to speak, act, or stay silent.

    Parameters
    ----------
    event : str
        Description of what happened.
    npc_name : str
        This NPC's name.
    npc_state : dict
        Current NPC state: emotion, stamina, hp.
    scene : str
        Current scene context.
    memory :
        MemoryStore for retrieving relevant memories.
    """
    parts: list[str] = []

    # Current state
    parts.append(f"## 你是 {npc_name}")
    parts.append(f"当前情绪: {npc_state.get('emotion', 'calm')}")
    parts.append(f"当前体力: {npc_state.get('stamina', 'fresh')}")
    parts.append(f"HP: {npc_state.get('hp', '?')}/{npc_state.get('max_hp', '?')}")

    # Scene
    if scene:
        parts.append(f"\n## 当前场景\n{scene}")

    # Relevant memories
    if memory:
        try:
            memories = memory.npc_full_retrieve(npc_name, event, n=3)
            if memories:
                mem_text = "\n".join(
                    f"- {m['content']}" if isinstance(m, dict) else f"- {m}"
                    for m in memories
                )
                parts.append(f"\n## 相关记忆\n{mem_text}")
        except Exception:
            pass

    # The event
    parts.append(f"\n## 发生了什么事\n{event}")

    # Decision prompt
    parts.append(
        "\n根据你的性格、当前情绪、以及和玩家的关系，你如何回应这件事？\n"
        "你可以选择：\n"
        "- 说话回应（用对话+动作描述）\n"
        "- 只做动作不说话（用动作描述）\n"
        "- 完全沉默（只回复 <silent>）\n\n"
        "如果你决定沉默，请只回复 <silent>。\n"
        "如果你决定回应，直接写你的对话和动作，不需要JSON格式。"
    )

    return "\n".join(parts)


def build_npc_tick_input(
    npc_name: str,
    npc_state: dict,
    scene: str,
    turns_since_last_action: int,
    memory=None,
) -> str:
    """Build input for an NPC's autonomous tick — time passes, what do you do?

    Called every N turns (default 3) for each NPC in the scene.

    Parameters
    ----------
    npc_name : str
        This NPC's name.
    npc_state : dict
        Current NPC state.
    scene : str
        Current scene context.
    turns_since_last_action : int
        How many turns since this NPC last acted or spoke.
    memory :
        MemoryStore for retrieving relevant memories.
    """
    parts: list[str] = []

    parts.append(f"## 你是 {npc_name}")
    parts.append(f"当前情绪: {npc_state.get('emotion', 'calm')}")
    parts.append(f"当前体力: {npc_state.get('stamina', 'fresh')}")
    parts.append(f"HP: {npc_state.get('hp', '?')}/{npc_state.get('max_hp', '?')}")

    if scene:
        parts.append(f"\n## 当前场景\n{scene}")

    if memory:
        try:
            memories = memory.npc_full_retrieve(npc_name, "", n=2)
            if memories:
                mem_text = "\n".join(
                    f"- {m['content']}" if isinstance(m, dict) else f"- {m}"
                    for m in memories
                )
                parts.append(f"\n## 最近发生的事情\n{mem_text}")
        except Exception:
            pass

    # Time passage
    parts.append(f"\n## 时间流逝\n距离你上次有所行动已经过了 {turns_since_last_action} 个回合。")

    # Decision
    parts.append(
        "你此刻在做什么？你想继续手头的事，还是有什么新的动作？\n"
        "你可以选择：\n"
        "- 做一个小动作（如巡视、擦杯子、整理货架）→ 描述动作\n"
        "- 主动和玩家说话 → 写对话+动作\n"
        "- 离开当前场景 → 描述离开\n"
        "- 什么都不做 → 回复 <silent>\n\n"
        "如果你觉得没什么特别的事，回复 <silent> 即可。"
    )

    return "\n".join(parts)
