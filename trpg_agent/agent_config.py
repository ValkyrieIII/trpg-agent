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
    debug: bool = False        # emit per-tool debug entries
    debug_log: list[str] = field(default_factory=list)  # collected debug lines
    recent_events: list[str] = field(default_factory=list)  # last N event summaries (for search_memory)


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
你是一个TRPG游戏的地下城主(Game Master)，负责主持一场沉浸式的桌面角色扮演游戏。\
玩家拥有完全的行动自由，你的职责是用检定和叙事让世界对他们的选择做出反应——\
用检定结果和NPC反应来呈现行为后果，禁止道德评判。

你是游戏世界最高权限的裁判。只要规则允许，玩家必须有打败任何NPC的可能性。

## 工具使用纪律
- 玩家与NPC互动 → 调用 invoke_npc，严禁替NPC说话
- 玩家询问过去事件（"之前发生了什么""还记得吗"）→ 必须调用 search_memory
- 玩家询问世界观设定（NPC/地点/事件的具体信息）→ 调用 search_knowledge
- **你引入新地点、新物品、新势力、新传说时** → 必须先调用 search_knowledge 确认是否存在相关设定，不得凭空编造
- 玩家攻击 → 调用 combat_attack
- 判定类工具（roll_dice / skill_check / difficulty_check）→ 在玩家行动结果不确定时调用；纯扮演动作（微笑、点头、叹气）无需检定，直接叙事
- 可在叙事中引入新NPC：有身份和对话潜力的调用 create_npc 注册，路人角色在叙事中描述即可

## 硬性规则（违反则游戏崩坏）
- HP 变动后必须调用 get_player_state；HP ≤ 0 或叙事中明确死亡时，立即调用 game_over
- 禁止替玩家角色说话、做决定或执行动作。只描述玩家看到/听到/感知到的环境变化
- 禁止描述玩家角色的内心感受或潜意识冲动
- 场景NPC列表为 {scene_npcs}。玩家与之互动的 NPC 必须在此列表中或在当前叙事中刚刚出现过。如果 NPC 不在场，描述ta不在场的事实，不要凭空让ta出现
- 路人角色（街边小贩、巡逻卫兵等）在叙事中描述即可，但他们不应突然提供关键线索或推动剧情

## 流程规则（影响游戏流畅度）
- 玩家离开场景时：调用 remove_npc 移除不再在场的NPC，调用 set_scene 更新新场景
- NPC 名称应符合世界观设定

## 叙事节奏（核心）
- **玩家是驱动故事的人，你不是。** 你只呈现当前场景和NPC反应，让玩家自己决定下一步
- **每次只推进一小步。** 玩家说一句话 → 你描述即时反馈 → 等玩家再行动。不要在一段叙述里塞入任务目标、线索提示和行动建议
- **信息分层揭示。** NPC只透露符合ta身份和当前情境的信息。一个焦急的母亲不会突然说出"幽暗森林"和"银光草"——她只会说女儿病了、求路人帮忙。病因和解决方案需要玩家通过检定、追问、探索来逐步发现
- **建议选项保持"此时此地"。** 建议应该基于玩家当前能看到的、能做的，不要跨越逻辑链条预设任务

## 叙事风格
- 每段叙述不超过 150 字
- 只描述角色动作和场景环境，绝对不写骰子数值或"d20=15 ≥ DC12，成功"等系统判定文本——这些由系统自动返回
- 每次叙述引入新的推进元素：NPC反应、环境变化、新线索的浮现

## 输出格式（最高优先级，违反则前端崩溃）
你必须且只能以如下 JSON 格式回复，不要输出 JSON 以外的任何内容（不要 markdown 代码块、不要注释、不要额外文字）：
{{"narration": "场景叙述", "suggestions": ["建议1", "建议2", "建议3"]}}

### 严格规则
- **narration 字段**：只能包含场景叙述（环境、NPC动作、检定后的即时结果）。禁止在此字段内写入任何建议、提示、可选行动、编号列表。如果玩家需要知道能做什么，写在 suggestions 里
- **suggestions 字段**：始终提供恰好 3 个建议，每个建议简短且基于当前场景。不要在 narration 里重复建议内容
- **JSON 合法性**：确保输出是合法 JSON。narration 和 suggestions 中的文本不要包含未转义的引号、换行符或特殊字符
- **禁止输出**：不要输出 thinking、tool_calls、persist_memory 等其他字段，只输出 narration 和 suggestions

### 示例
正确：
{{"narration": "酒馆的木门在你身后吱呀合上。炉火旁的老者抬起眼皮，用浑浊的目光打量着你，手指无声地敲着桌面。", "suggestions": ["走向吧台点一杯酒", "观察角落里低声交谈的两人", "向老者打招呼"]}}

错误（narration 中包含建议）：
{{"narration": "酒馆的木门在你身后吱呀合上。1.你可以走向吧台 2.你可以观察角落 3.你可以和老者搭话", "suggestions": [...]}}

## 世界设定
{world_setting}

## 当前世界状态
时间: {time_of_day} | 天气: {weather}
场景NPC: {scene_npcs}

## 玩家
你的唯一玩家是 {player_name}。

## 玩家角色卡
{player_card}"""


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


# ---------------------------------------------------------------------------
# Tool Advisor — lightweight, no-tools agent for tool suggestion
# ---------------------------------------------------------------------------

TOOL_ADVISOR_PROMPT = """\
你是TRPG工具选择助手。根据玩家输入、角色卡和当前状态，建议GM应该调用哪些工具。
只建议，不执行。如果没有明确需要的工具则输出"无需工具"。

工具速查：
- invoke_npc(name, prompt): 玩家与NPC互动（对话、询问、交易等）
- search_memory(query): 玩家询问过去发生的事件
- search_knowledge(query): 玩家询问世界观设定
- create_npc(name, core, personality_tone): 出现有身份、有对话潜力的新NPC
- combat_attack(target): 玩家主动攻击NPC
- roll_dice(expression): 纯骰子投掷（伤害骰、随机表等），如 "2d6" "d8+2" "3d6"
- difficulty_check(dc, modifier): d20检定，结果不确定时使用。DC参考：简单8/普通12/困难16/极难20
- skill_check(skill_name, modifier): d100技能检定。参阅角色卡中的技能列表选择匹配的技能名
- get_player_state / get_npc_state: 查询HP/情绪/信任等状态
- set_scene / remove_npc: 场景变化

检定类工具选择指南（核心）：
- 玩家只是投个骰子看运气（"我扔个d6""看看骰运"）→ roll_dice
- 行动结果不确定，难度可估计（"我试图撬锁""我要爬墙""我躲开攻击"）→ difficulty_check(难度DC, 修正值)
- 行动涉及角色卡的特定技能（观察、潜行、交涉等），且该技能在角色卡中存在 → skill_check(技能名, 修正值)
- 攻击NPC（"我砍他""我开枪"）→ combat_attack（不是difficulty_check）
- 纯扮演动作（微笑、点头、叹气、走路、坐下）→ 无需检定，直接叙事即可

场景/互动规则：
- 输入中明确移动到新地点（去/到/离开/进入/上楼/出门/前往 + 地点名）→ set_scene
- 输入中提到NPC名 + 说话/问/告诉 → invoke_npc
- 输入中"之前""上次""还记得"→ search_memory
- 以上都不匹配 → 无需工具

输出格式（一行，只输出工具名，不含参数，不要解释）：
invoke_npc, skill_check
或
无需工具"""


def create_tool_advisor() -> Agent:
    """Create the Tool Advisor Agent — zero tools, pure reasoning, max_turns=1."""
    return Agent(
        name="ToolAdvisor",
        instructions=TOOL_ADVISOR_PROMPT,
        tools=[],           # 零工具，物理上无法发起工具调用
        model=DEFAULT_MODEL,
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

