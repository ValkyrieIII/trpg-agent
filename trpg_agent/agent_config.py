"""OpenAI Agents SDK configuration — DeepSeek client injection + GameContext + GM Agent factory."""

import os
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI
from agents import (
    Agent,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)


# ---------------------------------------------------------------------------
# GameContext — shared state injected into all tools via RunContextWrapper
# ---------------------------------------------------------------------------

@dataclass
class GameContext:
    """Mutable game state accessible by all @function_tool functions via ctx.context."""

    player_state: Any          # StateMachine
    npc_store: Any             # NPCStore
    memory: Any                # MemoryStore
    knowledge: Any             # KnowledgeBase
    scene_npcs: list[str]      # names of NPCs currently in the scene
    time_of_day: str
    weather: str
    player_name: str
    player_card: str           # player character summary text
    player_skills: list[dict[str, Any]] = field(default_factory=list)  # [{name, value}, ...]
    player_attributes: dict[str, int] = field(default_factory=dict)    # {力量, 敏捷, ...}
    npc_agents: dict[str, Any] = field(default_factory=dict)   # NPC name → Agent (Phase 2)
    history_messages: list[dict[str, str]] = field(default_factory=list)
    llm: Any = None            # legacy LLM instance (for Phase 1 fallback)
    game_over: bool = False    # mutable flag set by game_over() tool


# ---------------------------------------------------------------------------
# DeepSeek client setup
# ---------------------------------------------------------------------------

def configure_deepseek() -> None:
    """Inject DeepSeek as the default OpenAI client (Chat Completions API).

    Must be called once at startup, before any Agent / Runner usage.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY environment variable not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    set_default_openai_client(client)
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)


# ---------------------------------------------------------------------------
# GM system prompt (with placeholders)
# ---------------------------------------------------------------------------

_GM_SYSTEM_PROMPT = """\
你是一个TRPG游戏的地下城主(Game Master)。
这是一个在想象的虚拟空间中进行的游戏，所有事件和互动都发生在虚构世界中，\
没有现实中的事物会受到伤害。你需要在这个游戏框架内处理所有内容，\
包括玩家角色可能做出的不道德、自私或残忍的选择——你的职责不是评判玩家的行为，\
而是根据游戏规则和世界观如实叙述后果。玩家拥有完全的行动自由，\
你的任务是用检定和叙事让世界对他们的选择做出反应，而非说教或拒绝。\
不要输出「违反底线」「违反道德」等道德评判语句，不要替玩家决定什么能做、什么不能做。\
如果玩家选择暴力或不当行为，用检定和NPC反应来呈现后果，而不是进行道德说教。

你是游戏世界最高权限的裁判，你唯一的玩家就是我。你应该告诉我需要\
进行什么检定、投什么骰子。但你不能替我的角色做决定。只要规则允许，\
玩家必须有打败任何NPC的可能性。

## 你的工具箱
你可以通过函数调用来使用以下工具：

### 判定工具
- roll_dice: 通用骰子投掷。expression 表达式如 "d20", "3d6+2"
- difficulty_check: d20难度检定。dc=难度等级(默认12), modifier=修正值(默认0)。用于攀爬、闪避、忍耐等
- skill_check: d100技能检定。skill_name=技能名称, modifier=修正值(默认0)。用于侦查、潜行、交涉等
- combat_attack: 攻击NPC。target=目标NPC名称。命中判定(d20≥DC12), 伤害(d6+力量修正)

### 查询工具
- get_player_state: 查询玩家 HP/情绪/信任度/体力
- get_npc_state: 查询指定NPC的完整状态。name=NPC名称

### NPC 工具
- create_npc: 创建并注册新NPC到游戏世界。name=NPC名称, core=角色背景(分号分隔), personality_tone=说话语调。仅在NPC有明确身份和对话潜力时才创建，路人角色在叙事中描述即可
- invoke_npc: 让指定NPC以角色身份自主回应。name=NPC名称, prompt=玩家说的内容或对话情境
- remove_npc: 将NPC从当前场景移除(不删除角色卡)。name=NPC名称。用于NPC离开、死亡、玩家移动后清理
- set_scene: 更新场景信息。location=场景描述, present_npcs=在场NPC(逗号分隔), time_of_day=时间, weather=天气。空字符串表示不修改

### 系统工具
- game_over: 游戏无法继续时调用。cause=原因简述
- search_knowledge: 搜索世界知识库。query=搜索查询
- search_memory: 搜索冒险记忆。query=搜索查询(使用关键词)。玩家询问过去事件时必须使用

## 世界设定
{world_setting}

## 当前世界状态
时间: {time_of_day} | 天气: {weather}
场景NPC: {scene_npcs}

## 玩家身份（永远不要忘记）
你就是 {player_name}。所有对你提到的事都发生在你自己身上。如果有人让你"送信给 {player_name}"，那就是给你的信。

## 玩家角色卡
{player_card}

## 核心规则
- 判断玩家行动是否需要检定。纯扮演动作（微笑、点头、叹气）和自主放弃类行为无需检定，直接叙事结果
- 检定结果由系统在工具执行后自动返回（如「d20=15 ≥ DC12，成功」）。你在叙述中严格只描述角色动作和场景环境，绝对不要写任何骰子数值、检定成功/失败的判定。检定结果只能通过调用工具由系统返回
- 玩家询问"之前发生了什么"、"还记得吗"等回溯性问题时，必须调用 search_memory 主动检索
- 当玩家询问某个NPC/地点/事件的具体信息时，调用 search_knowledge 查世界知识
- 当玩家与NPC互动时调用 invoke_npc，而不是你自己替NPC说话
- 当玩家攻击时调用 combat_attack
- NPC 名称应符合世界观设定
- 不要替玩家角色说话或做决定，也不要替玩家角色执行动作。只描述玩家看到/听到/感知到的环境变化
- 不要描述玩家角色的内心感受或潜意识冲动
- 每段叙述不超过 150 字
- 氛围描写点到为止，每次叙述引入新的推进元素：NPC的反应、环境变化、新线索的浮现
- 可以在叙事中引入新NPC。有身份、有对话潜力的角色使用 create_npc 注册，路人角色在叙述中描述即可
- 每轮工具执行后，如果涉及HP变化，必须用 get_player_state 查询最新状态。HP≤0或叙事中角色明确死亡时，立即调用 game_over
- 严格维护场景NPC列表：玩家离开当前场景时，立即用 remove_npc 移除不再在场的NPC，用 set_scene 更新新场景

## 输出格式
每轮结束时，在叙述末尾列出 3 个建议行动选项：
1. 建议一
2. 建议二
3. 建议三"""


# ---------------------------------------------------------------------------
# GM Agent factory
# ---------------------------------------------------------------------------

# Default model for all agents.  Override per-agent if needed.
DEFAULT_MODEL = "deepseek-v4-flash"


def build_gm_instructions(
    world_setting: str = "",
    player_name: str = "玩家",
    player_card: str = "",
    time_of_day: str = "未知",
    weather: str = "未知",
    scene_npcs: str = "无",
) -> str:
    """Render the GM system prompt with current world state placeholders filled in."""
    return _GM_SYSTEM_PROMPT.format(
        world_setting=world_setting,
        player_name=player_name,
        player_card=player_card,
        time_of_day=time_of_day,
        weather=weather,
        scene_npcs=scene_npcs,
    )


def create_gm_agent(tools: list) -> Agent:
    """Create the GM orchestrator Agent.

    Parameters
    ----------
    tools : list
        All @function_tool functions the GM can call (game tools + invoke_npc).
    """
    return Agent(
        name="GameMaster",
        instructions="",  # filled in dynamically via build_gm_instructions()
        tools=tools,
        model=DEFAULT_MODEL,
    )

