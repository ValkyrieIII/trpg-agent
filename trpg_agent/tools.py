"""Game tools migrated to @function_tool for the OpenAI Agents SDK.

All tools receive GameContext via ctx: RunContextWrapper[GameContext].
Phase 1: invoke_npc uses legacy LLM.chat() internally.
Phase 2: invoke_npc routes to real NPC Agents with autonomous decision-making.
"""

import os
import glob
import re

from agents import RunContextWrapper, function_tool, Runner

from trpg_agent.dice import roll as dice_roll
from trpg_agent.check import difficulty_check as check_dc, skill_check as check_skill
from trpg_agent.state import StateMachine, calc_max_hp

# Import GameContext for type hinting (avoid circular import)
from trpg_agent.agent_config import GameContext


# ---------------------------------------------------------------------------
# Dice / Check tools (stateless)
# ---------------------------------------------------------------------------

@function_tool
def roll_dice(ctx: RunContextWrapper[GameContext], expression: str) -> str:
    """Roll dice. expression like 'd20', '3d6+2', 'd100'."""
    try:
        results, total = dice_roll(expression)
        detail = " + ".join(str(r) for r in results)
        return f"投掷 {expression}: [{detail}] → {total}"
    except Exception as e:
        return f"骰子错误: {e}"


@function_tool
def difficulty_check(ctx: RunContextWrapper[GameContext], dc: int = 12, modifier: int = 0) -> str:
    """d20 difficulty check. d20 + modifier >= DC is success."""
    result = check_dc(dc=dc, modifier=modifier)
    roll_val = result["roll"]
    total = result["total"]
    if result["success"]:
        return f"d20={roll_val}+{modifier}={total} ≥ DC{dc} → 成功"
    else:
        return f"d20={roll_val}+{modifier}={total} < DC{dc} → 失败"


@function_tool
def skill_check(ctx: RunContextWrapper[GameContext], skill_name: str, modifier: int = 0) -> str:
    """d100 skill check. d100 <= skill_value is success. skill_name is the skill to check."""
    game_ctx = ctx.context

    # Find matching skill value from player skills list
    skill_value = 50  # default
    for s in game_ctx.player_skills:
        if isinstance(s, dict) and s.get("name", "").lower() == skill_name.lower():
            skill_value = int(s.get("value", 50))
            break

    result = check_skill(skill_value, modifier=modifier)
    roll_val = result["roll"]
    effective = result["effective_skill"]
    if result["success"]:
        return f"d100={roll_val} ≤ 技能{effective} → 成功"
    else:
        return f"d100={roll_val} > 技能{effective} → 失败"


# ---------------------------------------------------------------------------
# State query tools
# ---------------------------------------------------------------------------

@function_tool
def get_player_state(ctx: RunContextWrapper[GameContext]) -> str:
    """Query player HP/emotion/trust/stamina."""
    ps = ctx.context.player_state.get_state()
    return (
        f"HP {ps['hp']}/{ps['max_hp']} | "
        f"情绪 {ps['emotion']} | "
        f"信任 {ps['trust']} | "
        f"体力 {ps['stamina']}"
    )


@function_tool
def get_npc_state(ctx: RunContextWrapper[GameContext], name: str) -> str:
    """Query a specific NPC's full state."""
    game_ctx = ctx.context
    if not name:
        return "错误: 未指定 NPC 名称"

    npc = game_ctx.npc_store.find_by_name(name)
    if npc is None:
        return f"场景中找不到 NPC「{name}」"

    npc_state = game_ctx.npc_store.get_state(name)
    if npc_state is None:
        return f"{name}: 无状态记录"

    ns = npc_state.get_state()
    return (
        f"{name}: HP {ns['hp']}/{ns['max_hp']} | "
        f"情绪 {ns['emotion']} | "
        f"信任 {ns['trust']} | "
        f"体力 {ns['stamina']} | "
        f"存活: {'是' if ns['alive'] else '否'}"
    )


# ---------------------------------------------------------------------------
# NPC management tools
# ---------------------------------------------------------------------------

def _generate_npc_attributes(
    llm, name: str, core: list[str], personality_tone: str
) -> dict[str, int]:
    """Generate NPC attributes based on role description via LLM. (internal helper)"""
    if llm is None:
        return {
            "力量": 10, "敏捷": 10, "体质": 10,
            "智力": 10, "感知": 10, "魅力": 10,
        }

    core_text = "\n".join(core)
    try:
        result = llm.chat_json(
            system=(
                "你是TRPG角色设计师。根据角色描述，为该角色分配合理的属性值（1-20范围）。"
                "属性包括：力量、敏捷、体质、智力、感知、魅力。请根据角色特点合理分配，"
                "不要所有属性都一样。以JSON格式返回："
                "{\"力量\": 12, \"敏捷\": 10, \"体质\": 11, \"智力\": 14, \"感知\": 13, \"魅力\": 8}"
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"角色名称: {name}\n角色描述: {core_text}\n语调: {personality_tone}\n\n请为该角色分配属性。",
                }
            ],
        )
        attrs = {}
        for key in ["力量", "敏捷", "体质", "智力", "感知", "魅力"]:
            val = result.get(key, 10)
            attrs[key] = max(1, min(20, int(val)))
        return attrs
    except Exception:
        return {
            "力量": 10, "敏捷": 10, "体质": 10,
            "智力": 10, "感知": 10, "魅力": 10,
        }


@function_tool
def create_npc(
    ctx: RunContextWrapper[GameContext],
    name: str,
    core: str,
    personality_tone: str,
) -> str:
    """Create and register a new NPC in the game world.

    Args:
        name: NPC name (must contain letters or CJK characters).
        core: Character background description (one or more sentences).
        personality_tone: Speech tone description.
    """
    game_ctx = ctx.context

    # Validate name
    if not name:
        return "错误: NPC 名称不能为空"
    if not re.search(r"[a-zA-Z一-鿿]", name):
        return f"错误: NPC 名称「{name}」需包含有效字符（中文或字母）"

    # Convert core string to list of lines
    core_lines = [c.strip() for c in core.replace("；", ";").split(";") if c.strip()]
    if not core_lines:
        return f"错误: 创建 NPC「{name}」需要至少一条角色背景 (core)"

    # Validate personality_tone
    if not personality_tone:
        return f"错误: 创建 NPC「{name}」需要指定说话语调 (personality_tone)"

    # Check duplicate
    existing = game_ctx.npc_store.find_by_name(name)
    if existing is not None:
        if name not in game_ctx.scene_npcs:
            game_ctx.scene_npcs.append(name)
        return f"NPC「{name}」已存在，已加入场景"

    # Generate attributes
    attributes = _generate_npc_attributes(
        game_ctx.llm, name, core_lines, personality_tone
    )

    game_ctx.npc_store.create(
        name=name,
        core=core_lines,
        attributes=attributes,
        personality={
            "tone": personality_tone,
            "verbal_tics": "无特殊语言习惯",
            "emotion_map": {
                "calm": f"以{personality_tone}的态度说话",
                "wary": "警惕地观察",
                "hostile": "表现出敌意",
            },
        },
    )
    if name not in game_ctx.scene_npcs:
        game_ctx.scene_npcs.append(name)

    # ---- Phase 2: create NPC Agent for autonomous decision-making ----
    if name not in game_ctx.npc_agents:
        try:
            from trpg_agent.npc_agent import create_npc_agent

            npc_char = game_ctx.npc_store.find_by_name(name)
            if npc_char:
                game_ctx.npc_agents[name] = create_npc_agent(
                    name=name,
                    personality_prompt=npc_char.build_personality_prompt(),
                    npc_store=game_ctx.npc_store,
                )
        except Exception:
            pass  # NPC Agent creation is best-effort; fall back to legacy path

    core_summary = "；".join(core_lines[:2])
    return f"已创建 NPC「{name}」: {core_summary}（语调: {personality_tone}）"


@function_tool
def remove_npc(ctx: RunContextWrapper[GameContext], name: str) -> str:
    """Remove an NPC from the current scene (does not delete the character card)."""
    if not name:
        return "错误: 未指定 NPC 名称"
    if name in ctx.context.scene_npcs:
        ctx.context.scene_npcs.remove(name)
        return f"NPC「{name}」已从当前场景移除"
    return f"NPC「{name}」不在当前场景中"


@function_tool
def set_scene(
    ctx: RunContextWrapper[GameContext],
    location: str = "",
    present_npcs: str = "",
    time_of_day: str = "",
    weather: str = "",
) -> str:
    """Update current scene info: location, present NPCs (comma-separated), time, weather.
    Pass empty string for fields you don't want to change."""
    game_ctx = ctx.context
    changes = []

    if location:
        changes.append(f"场景: {location}")

    if present_npcs:
        npc_list = [n.strip() for n in present_npcs.split(",") if n.strip()]
        if npc_list:
            game_ctx.scene_npcs = npc_list
            changes.append(f"在场NPC: {', '.join(game_ctx.scene_npcs)}")

    if time_of_day:
        game_ctx.time_of_day = time_of_day
        changes.append(f"时间: {time_of_day}")

    if weather:
        game_ctx.weather = weather
        changes.append(f"天气: {weather}")

    if not changes:
        return "场景未变更"
    return "｜".join(changes)


# ---------------------------------------------------------------------------
# Combat tool
# ---------------------------------------------------------------------------

@function_tool
def combat_attack(ctx: RunContextWrapper[GameContext], target: str) -> str:
    """Attack an NPC. d20 >= DC12 hits, deals d6+STR damage."""
    game_ctx = ctx.context

    if not target:
        return "错误: 未指定攻击目标"

    target_npc = game_ctx.npc_store.find_by_name(target)
    if target_npc is None:
        return f"错误: 场景中找不到 NPC「{target}」"

    if target not in game_ctx.scene_npcs:
        game_ctx.scene_npcs.append(target)

    # Attack roll
    attack_result = check_dc(dc=12)
    roll_val = attack_result["roll"]

    if not attack_result["success"]:
        # Counter-attack
        _, counter_dmg = dice_roll("1d6")
        game_ctx.player_state.take_damage(counter_dmg)
        ps = game_ctx.player_state.get_state()
        return (
            f"d20={roll_val} < DC12 → 攻击落空｜"
            f"对方反击(d6={counter_dmg})，你受到{counter_dmg}点伤害 "
            f"(HP {ps['hp']}/{ps['max_hp']})"
        )

    # Hit — calculate damage
    # Strength bonus from player attributes
    str_bonus = max(0, (game_ctx.player_attributes.get("力量", 10) - 10) // 2)

    _, dmg_roll = dice_roll("1d6")
    damage = dmg_roll + str_bonus

    # Apply to defender
    npc_state = game_ctx.npc_store.get_state(target)
    if npc_state is None:
        npc_state = StateMachine(max_hp=calc_max_hp(target_npc.attributes))
        game_ctx.npc_store._states[target] = npc_state

    def_status = npc_state.take_damage(damage)
    def_hp = f"{npc_state.hp}/{npc_state.max_hp}"

    result = f"d20={roll_val} ≥ DC12 → 命中（d6={dmg_roll}+{str_bonus}={damage}点伤害, {target} HP {def_hp}）"
    if def_status == "dead":
        result += f" —— {target}倒下！"
        if target in game_ctx.scene_npcs:
            game_ctx.scene_npcs.remove(target)

    npc_state.apply("threatened")
    game_ctx.npc_store.save_state(target)

    return result


# ---------------------------------------------------------------------------
# Knowledge / Memory tools
# ---------------------------------------------------------------------------

@function_tool
def search_knowledge(ctx: RunContextWrapper[GameContext], query: str) -> str:
    """Search world knowledge base."""
    if not query:
        return "错误: 未指定搜索查询"
    results = ctx.context.knowledge.query(query, ctx.context.player_name)
    if not results:
        return f"未找到与「{query}」相关的知识"
    return "\n".join(f"- {r}" for r in results[:3])


@function_tool
def search_memory(ctx: RunContextWrapper[GameContext], query: str) -> str:
    """Search adventure memory (past events, NPC interactions, etc.)."""
    if not query:
        return "错误: 未指定搜索查询"
    memories = ctx.context.memory.search(query, n=5)
    if not memories:
        return f"未找到与「{query}」相关的记忆"
    return "\n".join(f"- {m['content']}" for m in memories[:5])


# ---------------------------------------------------------------------------
# Game over tool
# ---------------------------------------------------------------------------

@function_tool
def game_over(ctx: RunContextWrapper[GameContext], cause: str) -> str:
    """End the game (character death, etc.). Full reset."""
    game_ctx = ctx.context

    # Delete save file
    if os.path.exists("data/save.json"):
        os.remove("data/save.json")

    # Clear NPC state files
    for f in glob.glob("data/chroma/npcs/*_state.json"):
        try:
            os.remove(f)
        except Exception:
            pass

    # Clear run-time state
    game_ctx.scene_npcs.clear()
    game_ctx.time_of_day = "黄昏"
    game_ctx.weather = "阴"

    # Clear all memory collections
    try:
        all_ids = game_ctx.memory._collection.get()["ids"]
        if all_ids:
            game_ctx.memory._collection.delete(ids=all_ids)
    except Exception:
        pass
    try:
        all_npc_ids = game_ctx.memory._npc_collection.get()["ids"]
        if all_npc_ids:
            game_ctx.memory._npc_collection.delete(ids=all_npc_ids)
    except Exception:
        pass

    return f"游戏结束: {cause}"


# ---------------------------------------------------------------------------
# NPC speak tool (Phase 1: uses legacy LLM.chat)
# ---------------------------------------------------------------------------

@function_tool
def invoke_npc(ctx: RunContextWrapper[GameContext], name: str, prompt: str) -> str:
    """Have a specific NPC respond in-character.

    The NPC will reply based on their personality, current state,
    relevant memories, and conversation history.

    Phase 2: Uses the NPC's own Agent for autonomous decision-making,
    falling back to the legacy LLM.chat() path if no Agent exists.

    Args:
        name: The NPC name (must exist in the scene).
        prompt: What the player said or context for the NPC to respond to.
    """
    game_ctx = ctx.context

    if not name:
        return "错误: 未指定 NPC 名称"

    npc = game_ctx.npc_store.find_by_name(name)
    if npc is None:
        return f"场景中找不到 NPC「{name}」。请先使用 create_npc 创建。"

    if name not in game_ctx.scene_npcs:
        game_ctx.scene_npcs.append(name)

    # Get NPC state
    npc_state = game_ctx.npc_store.get_state(name)
    npc_state_dict = (
        npc_state.get_state()
        if npc_state
        else {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}
    )

    # ---- Phase 2: try NPC Agent first ----
    npc_agent = game_ctx.npc_agents.get(name)
    if npc_agent is not None:
        try:
            npc_reply = _invoke_npc_agent(
                game_ctx, npc_agent, name, prompt, npc_state_dict
            )
        except Exception:
            # Fall back to legacy path on Agent failure
            npc_reply = _invoke_npc_legacy(
                game_ctx, npc, name, prompt, npc_state_dict
            )
    else:
        # Phase 1 fallback: no NPC Agent created yet
        npc_reply = _invoke_npc_legacy(
            game_ctx, npc, name, prompt, npc_state_dict
        )

    # Record in NPC history
    game_ctx.npc_store.append_history(
        name, "user", f"{game_ctx.player_name}: {prompt}"
    )
    game_ctx.npc_store.append_history(name, "assistant", npc_reply)

    # Persist NPC state
    game_ctx.npc_store.save_state(name)

    return f'{name}: "{npc_reply}"'


# ---------------------------------------------------------------------------
# invoke_npc helpers (Phase 1 + 2)
# ---------------------------------------------------------------------------

def _invoke_npc_agent(
    game_ctx: GameContext,
    npc_agent,
    name: str,
    prompt: str,
    npc_state_dict: dict,
) -> str:
    """Phase 2: route to NPC Agent for autonomous decision-making."""
    import asyncio
    from trpg_agent.npc_agent import build_npc_input

    # Build NPC-specific context
    npc_context = {
        "npc_name": name,
        "memory": game_ctx.memory,
        "current_emotion": npc_state_dict.get("emotion", "calm"),
        "current_stamina": npc_state_dict.get("stamina", "fresh"),
        "current_hp": f"{npc_state_dict.get('hp', '?')}/{npc_state_dict.get('max_hp', '?')}",
        "scene": f"时间: {game_ctx.time_of_day}, 天气: {game_ctx.weather}",
    }

    # Retrieve relevant memories
    try:
        memories = game_ctx.memory.npc_full_retrieve(name, prompt, n=3)
        npc_context["relevant_memories"] = memories
    except Exception:
        pass

    # Build input
    npc_history = list(game_ctx.npc_store.get_history(name))
    npc_input = build_npc_input(prompt, npc_context, npc_history)

    # Run NPC Agent (use sync wrapper since we're inside a sync tool)
    result = Runner.run_sync(
        npc_agent,
        input=npc_input,
        context=npc_context,
        max_turns=3,
    )
    return result.final_output


def _invoke_npc_legacy(
    game_ctx: GameContext,
    npc,
    name: str,
    prompt: str,
    npc_state_dict: dict,
) -> str:
    """Phase 1 fallback: use legacy LLM.chat() for NPC response."""
    # Build NPC system prompt
    npc_system = (
        npc.build_personality_prompt()
        + "\n\n"
        + npc.build_state_prompt(npc_state_dict)
    )

    # Inject NPC relations
    if npc.relations:
        rel_lines = [f"- 与{k}的关系: {v}" for k, v in npc.relations.items()]
        npc_system += "\n\n【人物关系】\n" + "\n".join(rel_lines)

    # Inject NPC-specific memories
    try:
        npc_memories = game_ctx.memory.npc_full_retrieve(name, prompt, n=3)
        if npc_memories:
            npc_mem_text = "\n".join(
                f"- {m['content']} （{m.get('relation', '')}）" if m.get('relation')
                else f"- {m['content']}"
                for m in npc_memories
            )
            npc_system += (
                f"\n\n【与该玩家的过往交集（与当前情境最相关）】\n{npc_mem_text}"
            )
    except Exception:
        pass

    # Build messages from NPC history
    npc_messages = list(game_ctx.npc_store.get_history(name))

    # Legacy LLM call
    llm = game_ctx.llm
    if llm is not None:
        try:
            return llm.chat(system=npc_system, messages=npc_messages)
        except Exception:
            return f"(NPC「{name}」暂时不可用)"
    return "(LLM 未连接)"
