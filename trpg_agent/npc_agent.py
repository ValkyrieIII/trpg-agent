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
