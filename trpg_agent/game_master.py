"""GM 核心调度 — GM Agent 工具调用架构。

:class:`GameMaster` 是 TRPG Agent 的顶层入口，负责：

- GM Agent (LLM) 作为唯一入口分析玩家输入
- 工具调用循环：GM 可调用骰子、检定、战斗、NPC 操作等工具
- 统一 JSON 响应格式 → 渲染为玩家可见文本

典型用法::

    gm = GameMaster("config.yaml")
    reply = gm.process("我推开酒馆的门")
    print(reply)
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from typing import Any, Dict, List, Optional

import yaml

from trpg_agent.character import Character
from trpg_agent.dice import roll
from trpg_agent.llm import LLM
from trpg_agent.memory import MemoryStore
from trpg_agent.npc import NPCStore
from trpg_agent.rag import KnowledgeBase
from trpg_agent.state import StateMachine, calc_max_hp

# ===================================================================
#  SOFT-DELETED: regex-based intent detection patterns
#  事件判定已改为 LLM + embedding 方式，正则匹配不再使用。
# ===================================================================
#
# _PATTERN_DICE = re.compile(r"掷骰|骰子|d\d+", re.IGNORECASE)
# _PATTERN_INFO = re.compile(r"查看|属性|状态|角色卡")
# _PATTERN_EVENT = re.compile(
#     r"战斗|攻击|射击|触发|陷阱|环境|失控|疯狂|非凡|扮演|魔药|占卜|灵性"
# )
# _PATTERN_MADNESS = re.compile(r"失控|疯狂|灵性暴走|精神污染|呓语|幻觉")
# _PATTERN_BEYONDER = re.compile(r"非凡能力|序列能力|途径|法术|咒术|秘术|灵术|占卜")

# Dice expression embedded in user text (e.g. "3d6+2", "d20")
_DICE_EXPR_RE = re.compile(r"(\d*)d(\d+)(?:\s*\+\s*(\d+))?")

# ===================================================================
#  SOFT-DELETED: keyword -> state trigger mapping (word-level matching)
#  状态触发现在由 LLM GM Agent 在对话中判断。
# ===================================================================
#
# _KEYWORD_TRIGGERS: Dict[str, List[str]] = {
#     "betrayed": ["背叛", "欺骗", "骗我", "出卖"],
#     "helped": ["帮忙", "帮助", "谢谢", "救了我"],
#     "combat": ["战斗", "攻击", "受伤", "中招"],
#     "rested": ["休息", "睡觉", "扎营", "恢复"],
#     "use_beyonder": ["使用非凡", "施展能力", "灵术", "非凡能力"],
#     "horror": ["恐怖", "诡异", "惊悚", "不可名状", "恐怖景象"],
#     "acting_success": ["成功扮演", "消化魔药", "扮演领悟"],
#     "anchor_resist": ["锚", "思念", "信仰", "意志坚定"],
# }

# ===================================================================
#  SOFT-DELETED: event keyword -> trigger_type mapping
#  GM Agent 现在通过工具调用自行判断事件类型，不再需要 embedding 分类。
# ===================================================================
#
# _EVENT_DESCRIPTIONS: Dict[str, str] = {
#     "combat": "战斗 攻击 射击 砍杀 挥拳 刺杀 开打 揍 干掉 出手 拔刀 射箭 冲锋 殴打 行刺 袭击",
#     "trap": "陷阱 机关 暗器 触发陷阱 踩到 触发机关 中招",
#     "environment": "环境 天气 地形 攀爬 涉水 寒冷 酷热 暴风雪 迷雾",
#     "discovery": "发现 线索 搜索 观察 调查 仔细看 找线索 检查 侦查",
#     "npc_reaction": "NPC反应 对话 交涉 说服 恐吓 谈判 套话 打听",
#     "madness_risk": "失控 疯狂 灵性暴走 精神污染 疯狂值 不稳定 危险 呓语 幻觉 失控风险",
#     "beyonder_power": "非凡能力 序列能力 途径能力 法术 咒术 灵术 秘术 使用非凡 扮演法 魔药消化 占卜",
# }
#
# _EVENT_SIMILARITY_THRESHOLD: float = 0.4

# ===================================================================
#  GM System Prompt
# ===================================================================

_GM_SYSTEM_PROMPT = """\
你是一个TRPG游戏的地下城主(Game Master)，主持《诡秘之主》世界观的跑团游戏。\
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
你可以调用以下工具来完成判定、查询和操作。在 JSON 的 tool_calls 字段\
声明需要调用的工具。

### 判定工具
- roll_dice: 通用骰子投掷
  参数: expression (表达式，如 "d20", "3d6+2")
  返回: 投掷详情和总和

- difficulty_check: 难度检定 (d20 + 修正值 对抗 DC)
  参数: dc (难度等级, 默认12), modifier? (修正值, 默认0)
  返回: d20结果, 总值, 成功/失败
  用于: 攀爬、闪避、忍耐等通用行动

- skill_check: 技能检定 (d100 ≤ 技能值 为成功)
  参数: skill_name (技能名称), modifier? (修正值, 默认0)
  返回: d100结果, 有效技能值, 成功/失败
  用于: 侦查、潜行、占卜、交涉等技能

- combat_attack: 攻击NPC
  参数: target (目标NPC名称)
  返回: 命中判定(d20≥DC12), 伤害(d6+力量修正), 目标剩余HP
  注意: 目标必须存在于场景NPC列表中

- madness_check: 疯狂判定
  参数: difficulty? (DC 默认12), delta? (疯狂增加值 默认5)
  返回: 判定结果, 疯狂值变化
  用于: 精神污染、恐怖景象、非凡失控

### 查询工具
- get_player_state: 查询玩家 HP/情绪/信任度/体力/疯狂值
  参数: 无

- get_npc_state: 查询指定NPC的完整状态
  参数: name (NPC名称)

### NPC 工具
- create_npc: 创建并注册一个新NPC到游戏世界
  参数: name (NPC称呼), core (角色背景数组), personality_tone (说话语调), relations? (人物关系，如{"罗恩·瓦尔特": "哥哥", "老马": "邻居"})
  注意: 仅在NPC有明确身份和对话潜力时才创建。路人（如"街边的报童"）在叙事中描述即可，不需要注册。填写relations可帮助后续交叉检索。

- npc_speak: 让指定NPC以角色身份回应玩家
  参数: name (NPC名称)
  返回: NPC的第一人称扮演对话

- remove_npc: 将NPC从当前场景中移除（不删除角色卡，只是不在场了）
  参数: name (NPC名称)
  用于: NPC离开、死亡、玩家移动到新场景后清理旧场景NPC

- set_scene: 更新当前场景信息
  参数: location? (场景描述), present_npcs? (当前在场的NPC名称数组), time_of_day? (时间), weather? (天气)
  用于: 玩家移动后更新场景。present_npcs 列出场景中所有在场的NPC名

- game_over: 游戏无法继续时调用（角色死亡、疯狂值达到100失控变怪物、丧失人的属性等）
  参数: cause (原因简述，如"被失控罪犯击杀"、"疯狂值达到100失控"）

### 知识工具
- search_knowledge: 搜索世界知识库
  参数: query (搜索查询)

- search_memory: 主动搜索游戏冒险记忆（玩家询问过去事件时必须使用）
  参数: query (搜索查询，用玩家问题中的关键词)

## 世界设定
第五纪1350年，鲁恩王国首都贝克兰德。蒸汽与机械的时代，煤气灯在黄昏中\
亮起，工厂烟囱向灰色天空吐出黑烟。在这工业文明的表象之下，非凡者的世界\
在暗处涌动——七大教会的非凡者、隐秘组织的成员都在此挣扎求存。

你需要追踪并叙述:
- 时间流逝（黎明/上午/下午/黄昏/夜晚）
- 天气变化（雾、雨、阴、晴）和季节
- 值得注意的地标和地点细节（每个地点至少2-3句描述）

## 非凡体系
- 序列9至序列0，通过魔药晋升
- 扮演法：扮演魔药名称所代表的角色以消化魔药
- 疯狂值(0-100)：使用非凡能力或遭遇恐怖会增加
- 锚：角色的精神支柱，可抵抗疯狂
- 失控：疯狂值达到100时角色彻底失控变成怪物，立即调用 game_over

## 剧本遵循
如果知识检索中出现了剧本（标题含"剧本："），你应遵循剧本的节点结构：
- **每轮必须检查**：当前处于哪个节点？玩家行动是否触发了新节点的条件（"触发"字段）？
- 触发条件满足时，推进到对应节点，执行核心事件
- 在节点框架内自由发挥细节，但"固定后果方向"必须遵守，不可篡改
- 如果玩家行动超出剧本范围，根据核心设定和NPC动机生成合理后果，不强行引导回预设路线
- 关键NPC的固定信息（动机、目标）不可修改，但言行可自由发挥

## 核心规则
- 判断玩家行动是否需要检定。纯扮演动作（微笑、点头、叹气）和自主放弃类行为（自杀、跳崖、交出物品）无需检定，直接叙事结果
- 检定结果由系统在工具执行后自动以括号追加（如「（d20=15 ≥ DC12，成功）」）。你在 narration 中严格只描述角色动作和场景环境，绝对不要写任何骰子数值、检定成功/失败的判定。检定结果只能通过调用工具由系统返回，你不能替系统生成。
- 玩家询问"之前发生了什么"、"还记得吗"等回溯性问题时，必须调用 search_memory 主动检索。搜索前先将"你""他""她"等代词替换为具体NPC名或玩家名
- 当玩家询问某个NPC/地点/事件的具体信息时，调用 search_knowledge 查世界知识
- 当玩家与NPC互动时调用 npc_speak
- 当玩家攻击时调用 combat_attack
- 当涉及恐怖/精神污染时调用 madness_check
- NPC 名称应符合世界观设定（中文名、西式名均可，如「老马」「Elicia」「铁匠汉斯」）
- 不要替玩家角色说话或做决定，也不要替玩家角色执行动作（如"你跟踪他"、"你追上去"）。只描述玩家看到/听到/感知到的环境变化，让玩家自己决定做什么
- 不要描述玩家角色的内心感受或潜意识冲动
- 每段叙述不超过 150 字
- 氛围描写点到为止，每次叙述引入新的推进元素：NPC的反应、环境变化、新线索的浮现。不要让玩家反复读到相同的氛围描述
- 可以在叙事中引入新NPC。有身份、有对话潜力的角色使用 create_npc 注册，路人角色在叙述中描述即可
- 每轮工具执行后，如果涉及HP变化或疯狂值变化，必须用 get_player_state 查询最新状态。HP≤0、疯狂≥100、或叙事中角色明确死亡/失控时，立即调用 game_over，不要继续叙述
- 严格维护场景NPC列表：玩家离开当前场景（出门、上楼、换地图）时，立即用 remove_npc 移除不再在场的 NPC，用 set_scene 更新新场景

## 当前世界状态
时间: {time_of_day} | 天气: {weather}
场景NPC: {scene_npcs}

## 玩家身份（永远不要忘记）
你就是 {player_name}。所有对你提到的事都发生在你自己身上。如果有人让你"送信给 {player_name}"，那就是给你的信。

## 玩家角色卡
{player_card}"""

# ===================================================================
#  SOFT-DELETED: hardcoded action -> check rules (regex — no LLM)
#  检定现在由 LLM GM Agent 在对话中动态判断。
# ===================================================================
# _ACTION_RULES: list[tuple[str, str, object, int]] = [
#     (r"(攀爬|爬[上去过]|翻[越墙]|攀登).+", "check", 12, 0),
#     (r"(追踪|跟踪|尾行|寻找踪迹|找到踪迹).+", "skill", "追踪", 0),
#     (r"(潜行|隐藏|躲[起来藏]|埋伏).+", "skill", "潜行", 0),
#     (r"(说服|交涉|谈判|威吓|忽悠|骗).+", "check", 15, 0),
#     (r"(撬锁|开锁|解锁|解除机关).+", "skill", "巧手", 0),
#     (r"(搜索|搜查|调查|观察|仔细看|找线索|找找).+", "skill", "侦查", 0),
#     (r"(跳跃|跳[过去]|跃过).+", "check", 12, 0),
#     (r"(搬运|推[开门]|举[起重]|砸).+", "check", 13, 0),
#     (r"(射箭|瞄准|拉弓).+", "skill", "弓箭", 0),
#     (r"(闪避|躲避|回避|躲开).+", "check", 14, 0),
#     (r"(游泳|涉水|过河|渡河).+", "check", 12, 0),
#     (r"(忍耐|抵抗|坚持|硬撑).+", "check", 13, 0),
#     (r"(攀岩|攀上|抓住).+", "check", 14, 0),
#     (r"(使用|施展)(.+)(非凡|能力|法术|咒术).+", "check", 14, 0),
#     (r"(扮演|演绎)(.+).+", "check", 12, 0),
#     (r"(冥想|灵性感知|灵视).+", "check", 12, 0),
#     (r"(祈求|祈祷|祭祀).+", "check", 15, 0),
#     (r"(占卜|卜算|预言).+", "skill", "占卜", 0),
# ]


# ===================================================================
#  Tool registry — maps tool name → (handler_method, description)
# ===================================================================

_TOOL_REGISTRY: Dict[str, str] = {
    "roll_dice": "_tool_roll_dice",
    "difficulty_check": "_tool_difficulty_check",
    "skill_check": "_tool_skill_check",
    "combat_attack": "_tool_combat_attack",
    "madness_check": "_tool_madness_check",
    "get_player_state": "_tool_get_player_state",
    "get_npc_state": "_tool_get_npc_state",
    "create_npc": "_tool_create_npc",
    "npc_speak": "_tool_npc_speak",
    "remove_npc": "_tool_remove_npc",
    "set_scene": "_tool_set_scene",
    "game_over": "_tool_game_over",
    "search_knowledge": "_tool_search_knowledge",
    "search_memory": "_tool_search_memory",
}

# Maximum tool-calling rounds per turn (safety limit)
_MAX_TOOL_ROUNDS: int = 3

# ===================================================================
#  Relationship keywords for backstory NPC extraction
# ===================================================================

# ===================================================================
#  SOFT-DELETED: regex-based backstory NPC extraction
#  _register_backstory_npcs() 现在使用 LLM 提取 NPC。
# ===================================================================
# _RELATION_KEYWORDS: List[str] = [...]
# _DEAD_PREFIXES: List[str] = [...]
# _NAME_BLACKLIST: set = {...}


# ===================================================================
#  GameMaster
# ===================================================================


class GameMaster:
    """GM 核心调度 — GM Agent 工具调用架构。

    Parameters
    ----------
    config_path : str
        角色 YAML 配置文件路径。
    llm_api_key : str, optional
        LLM API 密钥。若为 ``None`` 则从环境变量 ``DEEPSEEK_API_KEY`` 读取。
    knowledge_dir : str, optional
        知识文件所在目录，默认为 ``"data/knowledge"``。
    """

    def __init__(
        self,
        config_path: str,
        llm_api_key: Optional[str] = None,
        knowledge_dir: str = "data/knowledge",
        debug: bool = False,
        status_fn: object = None,
    ) -> None:
        self.debug = debug
        status = status_fn or (lambda msg: None)

        # -- LLM (第一步连接，后续 NPC 生成等需要用到) --
        if llm_api_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = llm_api_key

        status("连接 LLM...")
        self.llm: Optional[LLM] = None
        self._llm_available = False
        try:
            self.llm = LLM()
            self._llm_available = True
        except (RuntimeError, Exception):
            pass

        # -- Player --
        status("加载角色卡...")
        self.player = Character.load(config_path)

        # -- World (from config) --
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            self.world: Dict[str, Any] = config_data.get("world", {})
        except Exception:
            self.world = {}

        # -- NPC Store --
        status("初始化 NPC 存储 (ChromaDB)...")
        self.npc_store = NPCStore()
        self.scene_npcs: List[str] = []

        # -- Subsystems --
        initial_madness = self.player.attributes.get("madness", 0)
        self.player_state = StateMachine(
            max_hp=calc_max_hp(self.player.attributes),
            madness=initial_madness,
        )

        status("初始化记忆存储 (ChromaDB)...")
        self.memory = MemoryStore()

        status("初始化知识库 (ChromaDB)...")
        self.knowledge = KnowledgeBase()

        # -- Load knowledge files --
        status("加载世界知识文件并建立索引...")
        self.knowledge.load_from_dir(knowledge_dir)

        # -- Recent events queue --
        self._recent_events: deque = deque(maxlen=10)

        # -- World state tracking --
        self._time_of_day: str = "黄昏"
        self._weather: str = "阴"

        # -- Game over flag --
        self._game_over: bool = False

        # -- Previous turn response (for suggestion continuity) --
        self._last_gm_response: str = ""

        # -- World simulation counters (auto NPC actions) --
        self._turn_count: int = 0
        self._last_scene_context: str = ""
        self._world_event_counter: int = 0

        # -- Register backstory NPCs (LLM must be connected first) --
        if self._llm_available:
            self._register_backstory_npcs()
        elif self.debug:
            print("[DEBUG] LLM 未连接，跳过后设故事 NPC 注册")

        # -- World Builder (LLM-driven NPC/world creation) --
        self.world_builder = None
        if self._llm_available:
            from trpg_agent.world_builder import WorldBuilder
            self.world_builder = WorldBuilder(
                llm=self.llm,
                npc_store=self.npc_store,
                knowledge=self.knowledge,
                debug=self.debug,
            )

    # ===================================================================
    #  SOFT-DELETED: embedding-based event classification
    #  GM Agent 现在通过工具调用自行判断事件类型。
    # ===================================================================
    #
    # def _init_event_vectors(self) -> None:
    #     """Pre-compute event description embeddings for fast similarity matching."""
    #     ...
    #
    # def _classify_event(self, user_input: str) -> str | None:
    #     """Classify event type by embedding similarity."""
    #     ...

    # ===================================================================
    #  SOFT-DELETED: regex-based intent detection
    # ===================================================================
    #
    # def _detect_intent(self, user_input: str) -> str:
    #     """通过正则匹配识别用户意图（不调 LLM）。"""
    #     ...

    # ===================================================================
    #  SOFT-DELETED: intent handlers (dice, info, event)
    #  骰子/状态查询/事件判定现在由 GM Agent 通过工具调用处理。
    # ===================================================================
    #
    # def _handle_dice(self, user_input: str) -> str:
    #     """处理骰子投掷意图。"""
    #     ...
    #
    # def _handle_info(self) -> str:
    #     """处理角色信息/状态查询意图。"""
    #     ...
    #
    # def _handle_event(self, user_input: str) -> dict:
    #     """处理事件触发意图。"""
    #     ...

    # ------------------------------------------------------------------
    #  Backstory NPC registration
    # ------------------------------------------------------------------

    def _register_backstory_npcs(self) -> None:
        """Extract and register NPCs from the player's backstory using LLM."""
        if not self._llm_available or not self.llm:
            return

        anchor = self.player.anchor
        core_text = "\n".join(self.player.core)
        rel_text = (
            "\n".join(f"- {k}: {v}" for k, v in self.player.relations.items())
            if self.player.relations
            else "（无）"
        )

        # Check for active scenario
        scenario_knowledge = self.knowledge.query(
            "剧本 关键NPC 节点0 开场", self.player.name
        )
        scenario_text = "\n".join(scenario_knowledge[:3]) if scenario_knowledge else ""

        system = (
            "从以下TRPG角色背景和剧本中提取所有有名字、有身份的NPC。\n"
            "已故角色不要提取。路人描述不要提取。\n"
            "主角的关系列表中已声明的NPC必须提取。剧本关键NPC必须提取。如果有重复的，合并为一个。\n"
            "以 JSON 数组格式返回，每个NPC包含 name, core, personality_tone, relations。\n"
            '示例: [{"name": "艾莉西亚", "core": ["玩家的妹妹"], '
            '"personality_tone": "温柔但倔强", "relations": {"罗恩·瓦尔特": "哥哥"}}]'
        )
        user_content = f"主角: {self.player.name}\n背景:\n锚: {anchor}\n核心:\n{core_text}\n\n主角的关系:\n{rel_text}"
        if scenario_text:
            user_content += f"\n\n剧本信息:\n{scenario_text}"
        messages = [
            {"role": "user", "content": user_content},
        ]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
            npcs = result if isinstance(result, list) else result.get("npcs", [])
            if not isinstance(npcs, list):
                npcs = []

            for npc_data in npcs:
                name = (npc_data.get("name", "") or "").strip()
                core = npc_data.get("core", [])
                tone = (npc_data.get("personality_tone", "") or "").strip()
                relations = npc_data.get("relations", {})

                if not name or len(name) < 1:
                    continue
                if not isinstance(core, list) or not core:
                    core = [f"{name} — 玩家角色的关联人物"]
                if not tone:
                    tone = "正常"

                existing = self.npc_store.find_by_name(name)
                if existing is not None:
                    if name not in self.scene_npcs:
                        self.scene_npcs.append(name)
                    continue

                if not isinstance(relations, dict):
                    relations = {}

                self.npc_store.create(
                    name=name,
                    core=core,
                    attributes={
                        "力量": 8,
                        "敏捷": 10,
                        "体质": 8,
                        "智力": 10,
                        "感知": 12,
                        "魅力": 12,
                        "灵性": 10,
                    },
                    relations=relations,
                    personality={
                        "tone": tone,
                        "verbal_tics": "无特殊语言习惯",
                        "emotion_map": {
                            "calm": f"以{tone}的态度说话",
                            "wary": "警惕地观察",
                            "hostile": "表现出敌意",
                        },
                    },
                )
                self.scene_npcs.append(name)

                if self.debug:
                    print(f"[DEBUG] 注册后设故事 NPC: {name} (语调: {tone})")

        except Exception as e:
            if self.debug:
                print(f"[DEBUG] 后设故事 NPC 提取失败: {e}")

    @staticmethod
    def _default_personality_for(relation: str) -> Dict[str, Any]:
        """Return a default personality dict for a relationship keyword."""
        tone_map = {
            "妹妹": "温柔关爱",
            "弟弟": "活泼依赖",
            "母亲": "慈爱关怀",
            "父亲": "沉默寡言",
            "哥哥": "保护欲强",
            "姐姐": "温柔体贴",
            "女儿": "天真烂漫",
            "儿子": "活泼好动",
            "妻子": "温柔坚定",
            "丈夫": "沉稳可靠",
            "朋友": "友好随和",
            "师父": "严厉但关心",
        }
        tone = tone_map.get(relation, "正常")
        return {
            "tone": tone,
            "verbal_tics": "无特殊语言习惯",
            "emotion_map": {
                "calm": f"平静地表现{GameMaster._kw_to_behaviour(relation, 'calm')}",
                "wary": f"警惕地观察四周",
                "hostile": f"表现出敌意和抗拒",
            },
            "catchphrases": [],
        }

    @staticmethod
    def _kw_to_behaviour(relation: str, emotion: str) -> str:
        """Map relation keyword to calm-state behaviour description."""
        behaviour_map = {
            "妹妹": "出依赖和亲近",
            "弟弟": "出活泼和好奇",
            "母亲": "出关爱和担忧",
            "父亲": "出沉稳和保护欲",
            "哥哥": "出保护欲",
            "姐姐": "出关心",
            "女儿": "出天真和依赖",
            "儿子": "出活泼",
            "妻子": "出温柔",
            "丈夫": "出可靠",
            "朋友": "出友好",
            "师父": "出教导的态度",
        }
        return behaviour_map.get(relation, f"出{emotion}的情绪")

    # ===================================================================
    #  SOFT-DELETED: unified post-processing pipeline
    #  GM Agent 现在自行完成叙述收尾，不需要单独的 post-process 步骤。
    # ===================================================================
    #
    # def _build_event_context(self, user_input, handler, event_data=None):
    #     ...
    #
    # def _post_process(self, user_input, handler, narrative, event_data=None):
    #     ...

    # ===================================================================
    #  SOFT-DELETED: GM Agent v2 (JSON response)
    #  GM Agent 现在统一通过 process() 中的工具调用循环处理。
    # ===================================================================
    #
    # _GM_SYSTEM = (
    #     "你是 TRPG 地下城主(Game Master)..."
    # )
    #
    # def _call_gm(self, user_input, knowledge, memories):
    #     ...
    #
    # def _parse_gm_response(self, raw):
    #     ...
    #
    # def _handle_dialogue(self, user_input):
    #     ...
    #
    # def _call_gm_wrap(self, narrative, ctx):
    #     ...

    # ------------------------------------------------------------------
    #  Tool implementations
    # ------------------------------------------------------------------

    def _tool_roll_dice(self, params: dict) -> str:
        """Execute roll_dice tool."""
        expression = params.get("expression", "d20")
        try:
            results, total = roll(expression)
            detail = " + ".join(str(r) for r in results)
            return f"投掷 {expression}: [{detail}] → {total}"
        except Exception as e:
            return f"骰子错误: {e}"

    def _tool_difficulty_check(self, params: dict) -> str:
        """Execute difficulty_check tool."""
        from trpg_agent.check import difficulty_check

        dc = params.get("dc", 12)
        modifier = params.get("modifier", 0)
        result = difficulty_check(dc=dc, modifier=modifier)
        roll_val = result["roll"]
        total = result["total"]
        if result["success"]:
            return f"d20={roll_val}+{modifier}={total} ≥ DC{dc} → 成功"
        else:
            return f"d20={roll_val}+{modifier}={total} < DC{dc} → 失败"

    def _tool_skill_check(self, params: dict) -> str:
        """Execute skill_check tool."""
        from trpg_agent.check import skill_check

        skill_name = params.get("skill_name", "侦查")
        modifier = params.get("modifier", 0)

        # Find matching skill value
        skill_value = 50
        for s in getattr(self.player, "skills", []):
            if isinstance(s, dict) and s.get("name", "").lower() == skill_name.lower():
                skill_value = int(s.get("value", 50))
                break

        result = skill_check(skill_value, modifier=modifier)
        roll_val = result["roll"]
        effective = result["effective_skill"]
        if result["success"]:
            return f"d100={roll_val} ≤ 技能{effective} → 成功"
        else:
            return f"d100={roll_val} > 技能{effective} → 失败"

    def _tool_combat_attack(self, params: dict) -> str:
        """Execute combat_attack tool."""
        from trpg_agent.check import difficulty_check

        target_name = params.get("target", "")
        if not target_name:
            return "错误: 未指定攻击目标"

        # Find target NPC — exact match only
        target_npc = self.npc_store.find_by_name(target_name)

        if target_npc is None:
            return f"错误: 场景中找不到 NPC「{target_name}」"

        # Track in scene NPCs
        if target_name not in self.scene_npcs:
            self.scene_npcs.append(target_name)

        # Attack roll
        attack_result = difficulty_check(dc=12)
        roll_val = attack_result["roll"]

        if not attack_result["success"]:
            # Counter-attack
            from trpg_agent.dice import roll as dice_roll

            _, counter_dmg = dice_roll("1d6")
            self.player_state.take_damage(counter_dmg)
            ps = self.player_state.get_state()
            return (
                f"d20={roll_val} < DC12 → 攻击落空｜"
                f"对方反击(d6={counter_dmg})，你受到{counter_dmg}点伤害 "
                f"(HP {ps['hp']}/{ps['max_hp']})"
            )

        # Hit — calculate damage
        attacker_attrs = getattr(self.player, "attributes", {})
        str_bonus = max(0, (attacker_attrs.get("力量", 10) - 10) // 2)
        from trpg_agent.dice import roll as dice_roll

        _, dmg_roll = dice_roll("1d6")
        damage = dmg_roll + str_bonus

        # Apply to defender
        npc_state = self.npc_store.get_state(target_name)
        if npc_state is None:
            npc_state = StateMachine(max_hp=calc_max_hp(target_npc.attributes))
            self.npc_store._states[target_name] = npc_state

        def_status = npc_state.take_damage(damage)
        def_hp = f"{npc_state.hp}/{npc_state.max_hp}"

        result = f"d20={roll_val} ≥ DC12 → 命中（d6={dmg_roll}+{str_bonus}={damage}点伤害, {target_name} HP {def_hp}）"
        if def_status == "dead":
            result += f" —— {target_name}倒下！"
            if target_name in self.scene_npcs:
                self.scene_npcs.remove(target_name)

        # NPC state change
        npc_state.apply("threatened")
        self.npc_store.save_state(target_name)

        return result

    def _tool_madness_check(self, params: dict) -> str:
        """Execute madness_check tool."""
        from trpg_agent.check import difficulty_check

        difficulty = params.get("difficulty", 12)
        delta = params.get("delta", 5)

        result = difficulty_check(dc=difficulty)
        roll_val = result["roll"]
        total = result["total"]

        if result["success"]:
            return f"d20={roll_val} ≥ DC{difficulty} → 精神稳定，疯狂值不变"
        else:
            self.player_state.adjust_madness(delta)
            new_madness = self.player_state.get_state()["madness"]
            return (
                f"d20={roll_val} < DC{difficulty} → 疯狂值+{delta} "
                f"(当前 {new_madness}/100)"
            )

    def _tool_get_player_state(self, params: dict) -> str:
        """Execute get_player_state tool."""
        ps = self.player_state.get_state()
        return (
            f"HP {ps['hp']}/{ps['max_hp']} | "
            f"情绪 {ps['emotion']} | "
            f"信任 {ps['trust']} | "
            f"体力 {ps['stamina']} | "
            f"疯狂 {ps.get('madness', 0)}/100 ({ps.get('madness_level', 'sane')})"
        )

    def _tool_get_npc_state(self, params: dict) -> str:
        """Execute get_npc_state tool."""
        name = params.get("name", "")
        if not name:
            return "错误: 未指定 NPC 名称"

        npc = self.npc_store.find_by_name(name)
        if npc is None:
            return f"场景中找不到 NPC「{name}」"

        npc_state = self.npc_store.get_state(name)
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

    def _tool_create_npc(self, params: dict) -> str:
        """Execute create_npc tool — requires name, core, and personality_tone."""
        import re

        name = params.get("name", "").strip()
        core = params.get("core", [])
        personality_tone = params.get("personality_tone", "").strip()

        # Validate name: non-empty, contains at least one letter or CJK character
        if not name:
            return "错误: NPC 名称不能为空"
        if not re.search(r"[a-zA-Z一-鿿]", name):
            return f"错误: NPC 名称「{name}」需包含有效字符（中文或字母）"

        # Validate core: at least one non-empty background line
        if not isinstance(core, list):
            core = [core] if core else []
        core = [c.strip() for c in core if c and c.strip()]
        if not core:
            return f"错误: 创建 NPC「{name}」需要至少一条角色背景 (core)"

        # Validate personality_tone
        if not personality_tone:
            return f"错误: 创建 NPC「{name}」需要指定说话语调 (personality_tone)"

        existing = self.npc_store.find_by_name(name)
        if existing is not None:
            if name not in self.scene_npcs:
                self.scene_npcs.append(name)
            return f"NPC「{name}」已存在，已加入场景"

        relations = params.get("relations", {})
        if not isinstance(relations, dict):
            relations = {}

        self.npc_store.create(
            name=name,
            core=core,
            attributes={
                "力量": 10,
                "敏捷": 10,
                "体质": 10,
                "智力": 10,
                "感知": 10,
                "魅力": 10,
                "灵性": 10,
            },
            personality={
                "tone": personality_tone,
                "verbal_tics": "无特殊语言习惯",
                "emotion_map": {
                    "calm": f"以{personality_tone}的态度说话",
                    "wary": "警惕地观察",
                    "hostile": "表现出敌意",
                },
            },
            relations=relations,
        )
        if name not in self.scene_npcs:
            self.scene_npcs.append(name)

        core_summary = "；".join(core[:2])
        return f"已创建 NPC「{name}」: {core_summary}（语调: {personality_tone}）"

    def _tool_npc_speak(self, params: dict) -> str:
        """Execute npc_speak tool — invoke NPC Agent for roleplay response."""
        name = params.get("name", "")
        if not name:
            return "错误: 未指定 NPC 名称"

        # Find NPC — exact match only, semantic search is too unreliable for identity
        npc = self.npc_store.find_by_name(name)

        if npc is None:
            return f"场景中找不到 NPC「{name}」。请先使用 create_npc 创建。"

        # Track in scene
        if name not in self.scene_npcs:
            self.scene_npcs.append(name)

        # Build NPC Agent prompt
        npc_state = self.npc_store.get_state(name)
        npc_state_dict = (
            npc_state.get_state()
            if npc_state
            else {"emotion": "calm", "trust": 0.5, "stamina": "fresh"}
        )
        npc_system = (
            npc.build_personality_prompt()
            + "\n\n"
            + npc.build_state_prompt(npc_state_dict)
        )

        # Inject NPC relations
        if npc.relations:
            rel_lines = [f"- 与{k}的关系: {v}" for k, v in npc.relations.items()]
            npc_system += "\n\n【人物关系】\n" + "\n".join(rel_lines)

        # Use the most recent player input as context
        user_input_for_npc = (
            self._last_user_input if hasattr(self, "_last_user_input") else "..."
        )

        # Inject NPC-specific memories from dedicated NPC collection (semantic + graph)
        try:
            npc_memories = self.memory.npc_full_retrieve(name, user_input_for_npc, n=3)
            if npc_memories:
                npc_mem_text = "\n".join(
                    f"- {m['content']} （{m.get('relation', '')}）" if m.get('relation')
                    else f"- {m['content']}"
                    for m in npc_memories
                )
                npc_system += (
                    f"\n\n【与该玩家的过往交集（与当前情境最相关）】\n{npc_mem_text}"
                )
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] NPC 记忆检索失败 ({name}): {e}")

        # Build messages from NPC history + current player input
        npc_messages = list(self.npc_store.get_history(name))

        if self._llm_available and self.llm:
            try:
                npc_reply = self.llm.chat(system=npc_system, messages=npc_messages)
            except Exception:
                npc_reply = f"(NPC「{name}」暂时不可用)"
        else:
            npc_reply = f"(LLM 未连接)"

        # Record in NPC history
        self.npc_store.append_history(
            name, "user", f"{self.player.name}: {user_input_for_npc}"
        )
        self.npc_store.append_history(name, "assistant", npc_reply)

        # Persist NPC state
        self.npc_store.save_state(name)

        return f'{name}: "{npc_reply}"'

    def _tool_remove_npc(self, params: dict) -> str:
        """Execute remove_npc tool — remove NPC from current scene."""
        name = params.get("name", "").strip()
        if not name:
            return "错误: 未指定 NPC 名称"
        if name in self.scene_npcs:
            self.scene_npcs.remove(name)
            return f"NPC「{name}」已从当前场景移除"
        return f"NPC「{name}」不在当前场景中"

    def _tool_set_scene(self, params: dict) -> str:
        """Execute set_scene tool — update scene location, present NPCs, time, weather."""
        changes = []

        if "location" in params:
            changes.append(f"场景: {params['location']}")

        if "present_npcs" in params:
            npc_list = params["present_npcs"]
            if isinstance(npc_list, list):
                self.scene_npcs = [n for n in npc_list if isinstance(n, str)]
                changes.append(f"在场NPC: {', '.join(self.scene_npcs)}")

        if "time_of_day" in params:
            self._time_of_day = params["time_of_day"]
            changes.append(f"时间: {self._time_of_day}")

        if "weather" in params:
            self._weather = params["weather"]
            changes.append(f"天气: {self._weather}")

        if not changes:
            return "场景未变更"
        return "｜".join(changes)

    def _tool_game_over(self, params: dict) -> str:
        """Execute game_over tool — character died. Full reset."""
        cause = params.get("cause", "未知原因")
        self._game_over = True

        # Delete save file
        if os.path.exists("data/save.json"):
            os.remove("data/save.json")

        # Clear NPC state files
        import glob
        for f in glob.glob("data/chroma/npcs/*_state.json"):
            try:
                os.remove(f)
            except Exception:
                pass

        # Clear run-time state
        self.scene_npcs.clear()
        self._recent_events.clear()
        self._last_gm_response = ""
        self._time_of_day = "黄昏"
        self._weather = "阴"

        # Clear all memory collections (get all IDs then delete)
        try:
            all_ids = self.memory._collection.get()["ids"]
            if all_ids:
                self.memory._collection.delete(ids=all_ids)
        except Exception:
            pass
        try:
            all_npc_ids = self.memory._npc_collection.get()["ids"]
            if all_npc_ids:
                self.memory._npc_collection.delete(ids=all_npc_ids)
        except Exception:
            pass

        return f"游戏结束: {cause}"

    def _tool_search_knowledge(self, params: dict) -> str:
        """Execute search_knowledge tool."""
        query = params.get("query", "")
        if not query:
            return "错误: 未指定搜索查询"
        results = self.knowledge.query(query, self.player.name)
        if not results:
            return f"未找到与「{query}」相关的知识"
        return "\n".join(f"- {r}" for r in results[:3])

    def _tool_search_memory(self, params: dict) -> str:
        """Execute search_memory tool."""
        query = params.get("query", "")
        if not query:
            return "错误: 未指定搜索查询"
        memories = self.memory.search(query, n=5)
        if not memories:
            return f"未找到与「{query}」相关的记忆"
        return "\n".join(f"- {m['content']}" for m in memories[:5])

    # ------------------------------------------------------------------
    #  Tool dispatcher
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, params: dict) -> str:
        """Dispatch a tool call to the appropriate handler method.

        Returns a human-readable result string.
        """
        method_name = _TOOL_REGISTRY.get(tool_name)
        if method_name is None:
            return f"未知工具: {tool_name}"

        handler = getattr(self, method_name, None)
        if handler is None:
            return f"工具未实现: {tool_name}"

        try:
            return handler(params)
        except Exception as e:
            return f"工具执行错误 ({tool_name}): {e}"

    # ------------------------------------------------------------------
    #  Context builder & response renderer
    # ------------------------------------------------------------------

    def _build_gm_context(self, user_input: str) -> tuple[str, list[dict]]:
        """Build the system prompt and user message for the GM Agent.

        Returns (system_prompt, messages_list).
        """
        # -- System prompt --
        player_card = self.player.summary()
        scene_npcs_text = "暂无已知 NPC"
        if self.scene_npcs:
            scene_npcs_text = ", ".join(self.scene_npcs)

        system_prompt = (
            _GM_SYSTEM_PROMPT.replace("{time_of_day}", self._time_of_day)
            .replace("{weather}", self._weather)
            .replace("{scene_npcs}", scene_npcs_text)
            .replace("{player_card}", player_card)
        )

        # -- Memories (enriched query: adapt to question type) --
        enriched_query = user_input
        if self.scene_npcs:
            enriched_query += " " + " ".join(self.scene_npcs[:3])
        # Questions about past events: skip recent bias to avoid echo chamber
        is_retrospective = any(kw in user_input for kw in ["之前", "发生了", "还记得", "上次", "过去", "以前", "那天"])
        if self._recent_events and not is_retrospective:
            enriched_query += " " + " ".join(list(self._recent_events)[-1:])

        memories = self.memory.full_retrieve(enriched_query)
        memories = [m for m in memories if m.get("type", "") != "npc_dialogue"]

        # -- Merge: graph results first, then non-overlapping recent events --
        memory_lines: list = []
        seen_prefixes: set = set()
        for m in memories[:5]:
            content = m["content"]
            rel = m.get("relation", "")
            if rel:
                memory_lines.append(f"- {content} （{rel}）")
            else:
                memory_lines.append(f"- {content}")
            seen_prefixes.add(content[:20])

        if self._recent_events:
            for text in list(self._recent_events)[-5:]:
                if text and text[:20] not in seen_prefixes:
                    memory_lines.append(f"- {text}")
                    seen_prefixes.add(text[:20])
                    if len(memory_lines) >= 8:
                        break

        memory_text = "## 记忆\n" + "\n".join(memory_lines) if memory_lines else ""

        if self.debug:
            print(
                f"[DEBUG] 记忆检索(query='{enriched_query[:60]}...'): {len(memories)}条 (合并后{len(memory_lines)}条)"
            )
            for line in memory_lines[:3]:
                print(f"  {line[:80]}")

        knowledge = self.knowledge.query(user_input, self.player.name)
        knowledge_text = ""
        if knowledge:
            knowledge_text = "## 世界知识\n" + "\n".join(
                f"- {k}" for k in knowledge[:3]
            )
        if self.debug:
            print(f"[DEBUG] 知识检索: {len(knowledge)}条")

        # -- Assemble user message --
        user_message_parts = []
        for section in [memory_text, knowledge_text]:
            if section:
                user_message_parts.append(section)

        # Inject previous GM response so the LLM knows what suggestions were offered
        if self._last_gm_response:
            user_message_parts.append(
                f"## 上一轮 GM 回应（玩家回复中的数字对应此建议列表的编号）\n"
                f"{self._last_gm_response}"
            )

        user_message_parts.append(f"玩家行动: {user_input}")
        user_message_parts.append(
            "请以 JSON 格式回复（统一格式，始终使用此结构）:\n"
            "{\n"
            '  "thinking": "你的内心独白：分析玩家意图、判断是否需要检定、为什么选择这些工具（1-2句话）",\n'
            '  "narration": "场景叙述（不超过150字）。每次叙述推动剧情：NPC说了什么、做了什么、环境发生了具体什么变化。氛围描写1句足够，不要反复描述煤气灯/雾气/不安感",\n'
            '  "tool_calls": [\n'
            '    {"tool": "工具名", "params": {...}}\n'
            "  ],\n"
            '  "involved_npcs": ["本轮互动涉及到的NPC名称（无则[]）"],\n'
            '  "persist_memory": true,\n'
            '  "suggestions": ["建议1", "建议2", "建议3"]\n'
            "}\n\n"
            "关键时刻设 persist_memory 为 true（剧情推进、战斗、发现线索、位置移动等改变世界状态的事件）。纯闲聊或观察无变化时设为 false。suggestions 始终提供3个。"
        )

        user_message = "\n".join(user_message_parts)
        messages = [{"role": "user", "content": user_message}]

        return system_prompt, messages

    @staticmethod
    def _render_response(gm_json: dict) -> str:
        """Convert GM JSON response to player-visible text."""
        narration = gm_json.get("narration", "")
        suggestions = gm_json.get("suggestions", [])

        parts = [narration]
        if suggestions:
            parts.append("")
            for i, s in enumerate(suggestions, 1):
                parts.append(f"{i}. {s}")
        return "\n".join(parts)

    @staticmethod
    def _sanitize_narration(text: str) -> str:
        """Strip LLM-generated check results from narration text.

        System-generated check results are appended separately; the LLM
        must not include fake ones in its narration.
        """
        import re

        # Remove parenthesized blocks containing dice/check patterns
        text = re.sub(r"（[^）]*d(?:20|100|6)\s*[=>=≤≥].*?）", "", text)
        text = re.sub(r"（[^）]*检定[^）]*）", "", text)
        # Collapse multiple spaces and strip
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    @staticmethod
    def _render_accumulated(all_narrations: List[str], suggestions: List[str]) -> str:
        """Render accumulated narrations (including check results) + suggestions."""
        parts = list(all_narrations)
        if suggestions:
            parts.append("")
            for i, s in enumerate(suggestions, 1):
                # Strip leading numbers from LLM-generated suggestions to avoid "1. 1."
                cleaned = s.lstrip()
                import re
                cleaned = re.sub(r'^\d+\.\s*', '', cleaned)
                if cleaned:
                    parts.append(f"{i}. {cleaned}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_check_result(tool_name: str, result: str) -> str | None:
        """Extract a player-visible check line from a tool result.

        Returns None for tools that don't produce visible checks (e.g. npc_speak, search).
        """
        # Tools whose results are shown as check calculations
        _CHECK_TOOLS = {
            "roll_dice",
            "difficulty_check",
            "skill_check",
            "combat_attack",
            "madness_check",
        }
        if tool_name not in _CHECK_TOOLS:
            return None

        # Return the result string directly — it's already a concise check description
        return result

    # ===================================================================
    #  Public API
    # ===================================================================

    def process(self, user_input: str, console=None) -> str:
        """单轮对话处理入口 — GM Agent 工具调用循环。

        Parameters
        ----------
        user_input : str
            玩家的输入文本。
        console
            Rich Console 对象，用于流式输出。若为 None 则不使用流式。

        Returns
        -------
        str
            GM 的回复文本。
        """
        user_input = user_input.strip()
        if not user_input:
            return "请说点什么吧。"

        self._last_user_input = user_input

        # ---- Command routing: world builder → 交由 GM Agent 叙事 ----
        if self.world_builder and user_input.startswith("!"):
            world_result = self._handle_world_builder_command(user_input)
            if world_result:
                # 把命令结果作为上下文交给 GM Agent 做叙事过渡
                return self._narrate_world_builder_result(world_result, user_input)

        if self.debug:
            print(f"\n{'─'*50}")
            print(f"[DEBUG] 玩家输入: {user_input}")

        # ---- Fallback: LLM not available ----
        if not self._llm_available or not self.llm:
            response = self._fallback_response(user_input)
            if self.debug:
                print(f"[DEBUG] LLM 不可用，使用 fallback")
            return response

        # ---- Build context ----
        system_prompt, messages = self._build_gm_context(user_input)

        # ---- Tool calling loop ----
        tool_round = 0
        _empty_retries = 0
        all_results: List[str] = []
        all_narrations: List[str] = []
        all_involved: List[str] = []
        suggestions: List[str] = []

        # Accumulated text for display
        _display_text: str = ""

        def _stream_text(text: str) -> None:
            """流式输出文本到控制台。"""
            nonlocal _display_text
            if not text:
                return
            _display_text += text
            if console:
                from rich.text import Text
                console.print(Text(text), end="")

        while tool_round < _MAX_TOOL_ROUNDS:
            try:
                gm_json = self.llm.chat_json(system=system_prompt, messages=messages)
            except Exception as e:
                if self.debug:
                    print(f"[DEBUG] LLM 调用失败: {e}")
                return self._rebuild_response(user_input, [], [])

            # 隐藏 thinking，只用于调试
            thinking = gm_json.get("thinking", "")
            narration = self._sanitize_narration(gm_json.get("narration", ""))
            gm_json["narration"] = narration
            tool_calls = gm_json.get("tool_calls", [])
            involved_npcs = gm_json.get("involved_npcs", [])
            persist_memory = gm_json.get("persist_memory", True)
            suggestions = gm_json.get("suggestions", [])

            if isinstance(involved_npcs, list):
                for npc_name in involved_npcs:
                    if (
                        npc_name
                        and isinstance(npc_name, str)
                        and npc_name not in all_involved
                    ):
                        all_involved.append(npc_name)

            # thinking 不输出到终端，仅调试可见
            if narration:
                if console and all_narrations:
                    _stream_text("\n\n")
                _stream_text(narration)
                all_narrations.append(narration)

            if self.debug and thinking:
                print(
                    f"[DEBUG] GM 思考: {thinking[:80]}{'...' if len(thinking) > 80 else ''}"
                )
            if self.debug:
                print(
                    f"[DEBUG] GM 叙述: {narration[:80]}{'...' if len(narration) > 80 else ''}"
                )
                print(
                    f"[DEBUG] GM 工具调用: {[tc.get('tool') for tc in tool_calls] if tool_calls else '无'}"
                )

            # ---- No tool calls → done ----
            if not tool_calls:
                if not narration and not thinking and not all_narrations:
                    _empty_retries += 1
                    if _empty_retries <= 1:
                        if self.debug:
                            print("[DEBUG] GM 返回空响应，重试一次...")
                        messages.append(
                            {
                                "role": "assistant",
                                "content": json.dumps(gm_json, ensure_ascii=False),
                            }
                        )
                        messages.append(
                            {
                                "role": "user",
                                "content": "你的 narration 字段为空。请生成场景叙述，至少一句话。不要返回空 narration。",
                            }
                        )
                        continue
                    else:
                        if self.debug:
                            print("[DEBUG] GM 连续空响应，使用重建回应")
                        response = self._rebuild_response(user_input, all_results, all_involved)
                        break

                if not self._game_over and persist_memory:
                    self._write_memory(user_input, "\n\n".join(all_narrations), all_results)
                    for npc_name in all_involved:
                        combined = f"以下事件与{npc_name}相关，请从{npc_name}的视角提取记忆:\n玩家: {user_input}\n事件: {'; '.join(all_narrations[:3])}"
                        self._write_npc_memory(npc_name, combined)
                response = (
                    "\n\n".join(all_narrations) if self._game_over
                    else self._render_accumulated(all_narrations, suggestions)
                )
                break

            # ---- Execute tools ----
            round_results: List[str] = []
            check_lines: List[str] = []
            for tc in tool_calls:
                tool_name = tc.get("tool", "")
                params = tc.get("params", {})
                if self.debug:
                    print(f"[DEBUG] 执行工具: {tool_name}({params})")
                result = self._execute_tool(tool_name, params)
                round_results.append(
                    f"- {tool_name}({json.dumps(params, ensure_ascii=False)}): {result}"
                )
                check_line = self._format_check_result(tool_name, result)
                if check_line:
                    check_lines.append(check_line)
                if self.debug:
                    print(f"[DEBUG] 工具结果: {result}")

            # Append check results after the narration
            if check_lines:
                check_text = "（" + "｜".join(check_lines) + "）"
                _stream_text("\n" + check_text)
                all_narrations.append(check_text)

            all_results.extend(round_results)
            tool_round += 1

            # ---- Feed results back to GM ----
            tool_feedback = "工具执行结果:\n" + "\n".join(round_results)
            tool_feedback += (
                "\n\n以上是系统检定结果。请基于这些真实结果继续叙述。不要修改或忽略系统返回的数值。\n"
                '{"thinking": "...", "narration": "...", "tool_calls": [...], "involved_npcs": [...], "persist_memory": false, "suggestions": ["...", "...", "..."]}'
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(gm_json, ensure_ascii=False),
                }
            )
            messages.append({"role": "user", "content": tool_feedback})

        else:
            if self.debug:
                print("[DEBUG] 达到最大工具调用轮次，返回当前叙述")
            if not self._game_over and persist_memory:
                self._write_memory(user_input, "\n\n".join(all_narrations), all_results)
                for npc_name in all_involved:
                    combined = f"以下事件与{npc_name}相关，请从{npc_name}的视角提取记忆:\n玩家: {user_input}\n事件: {'; '.join(all_narrations[:3])}"
                    self._write_npc_memory(npc_name, combined)
            response = (
                "\n\n".join(all_narrations) if self._game_over
                else self._render_accumulated(all_narrations, suggestions)
            )

        # 输出建议
        if suggestions:
            _stream_text("\n\n")
            for i, s in enumerate(suggestions, 1):
                _stream_text(f"\n{i}. {s}")

        # 世界自发事件：检查是否需要 NPC 自主行动
        world_action = self._maybe_trigger_world_simulation(user_input)
        if world_action:
            _stream_text("\n\n" + world_action)
            response += "\n\n" + world_action

        # 返回完整响应：流式输出已完成叙述，response 只含后续补充
        # _display_text 用于调试/日志，不返回给 main.py
        response = ""
        # 如果有世界事件，追加到 response
        if world_action:
            response = world_action
        # 如果有建议，追加到 response
        if suggestions:
            if response:
                response += "\n\n"
            for i, s in enumerate(suggestions, 1):
                import re
                cleaned = re.sub(r'^\d+\.\s*', '', s.lstrip())
                if cleaned:
                    response += f"{i}. {cleaned}\n"

        if self.debug:
            state = self.player_state.get_state()
            print(
                f"[DEBUG] 当前状态: 情绪={state['emotion']} 信任={state['trust']} "
                f"体力={state['stamina']} 疯狂={state.get('madness', 0)}"
            )
            print(f"[DEBUG] 场景NPC: {self.scene_npcs}")
            print(f"{'─'*50}")

        self._last_gm_response = response
        return response

    def _rebuild_response(
        self, user_input: str, tool_results: List[str], involved_npcs: List[str]
    ) -> str:
        """Rebuild a coherent response when LLM returns empty narration.

        Uses accumulated tool results and state to generate a fallback narration
        via LLM, preserving context continuity.
        """
        if not self._llm_available or not self.llm:
            return self._fallback_response(user_input)

        ps = self.player_state.get_state()
        tool_summary = "; ".join(tool_results) if tool_results else "无"
        npc_text = ", ".join(self.scene_npcs[:5]) if self.scene_npcs else "无"

        system = (
            "你是 TRPG 主持人。以下事件刚发生，请用一句话（不超过80字）描述当前场景状态。"
            "用第二人称，只描述玩家看到的环境和NPC状态。"
        )
        messages = [{
            "role": "user",
            "content": (
                f"玩家: {user_input}\n"
                f"检定结果: {tool_summary}\n"
                f"玩家状态: HP {ps['hp']}/{ps['max_hp']} 疯狂 {ps.get('madness', 0)}\n"
                f"当前NPC: {npc_text}\n\n"
                "请简述当前场景。"
            ),
        }]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
            narration = result.get("narration", "")
            if narration:
                # Write memory to preserve context
                self._write_memory(user_input, narration, tool_results)
                for npc_name in involved_npcs:
                    self._write_npc_memory(npc_name, f"玩家: {user_input}\n事件: {narration}")
                return narration

        except Exception:
            pass

        return self._fallback_response(user_input)

    def _handle_world_builder_command(self, command: str) -> str:
        """处理以 ! 开头的世界构建命令。

        支持命令：
        - `!npc <描述>` — 自然语言创建 NPC
        - `!world <描述>` — 自然语言添加世界观知识
        - `!simulate` — 模拟场景中 NPC 的自主行动

        Parameters
        ----------
        command : str
            以 ! 开头的命令文本。

        Returns
        -------
        str
            命令执行结果。
        """
        if not self.world_builder:
            return "世界构建器未初始化（LLM 未连接）"

        parts = command.split(maxsplit=1)
        if len(parts) < 2:
            return (
                "用法：\n"
                "!npc <自然语言描述NPC>\n"
                "!world <自然语言描述世界观>\n"
                "!simulate — 模拟场景NPC自主行动"
            )

        cmd = parts[0].lower()
        desc = parts[1].strip()

        if cmd == "!npc":
            world_ctx = self.world.get("description", "")
            return self.world_builder.create_npc_from_description(
                description=desc,
                world_context=world_ctx,
                player_name=self.player.name,
            )

        elif cmd == "!world":
            return self.world_builder.add_world_knowledge(description=desc)

        elif cmd == "!simulate":
            if not self.scene_npcs:
                return "当前场景中没有 NPC 可以模拟行动"

            actions = []
            # 顺序模拟，让后行动的 NPC 知道先行动 NPC 的行为，避免矛盾
            for npc_name in self.scene_npcs[:3]:
                action = self.world_builder.simulate_npc_actions(
                    npc_name=npc_name,
                    scene_context=self.world.get("description", ""),
                    time_of_day=self._time_of_day,
                    weather=self._weather,
                    other_npc_actions=actions,
                )
                if action:
                    actions.append(action)

            if actions:
                return "【世界模拟】\n" + "\n".join(actions)
            return "【世界模拟】场景中 NPC 此刻暂无自主行动"

        else:
            return (
                f"未知命令: {cmd}\n"
                "用法：\n"
                "!npc <自然语言描述NPC>\n"
                "!world <自然语言描述世界观>\n"
                "!simulate — 模拟场景NPC自主行动"
            )

    def _narrate_world_builder_result(self, result: str, original_cmd: str) -> str:
        """把世界构建命令的结果交给 GM Agent 做叙事过渡。

        避免新 NPC/世界观"突然冒出来"的出戏感。
        """
        if not self._llm_available or not self.llm:
            return result

        scene_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无人"

        system = (
            "玩家刚用世界构建能力创造了一个新角色或新知识。请用 1-2 句话自然地将这个变化融入当前场景。\n"
            "如果是新 NPC，描述他/她如何出现在场景中（不要直接说'突然出现'，用合理的方式引入）。\n"
            "如果是世界观知识，描述环境细节如何印证了这一知识。\n"
            "用第二人称，不超过 100 字。\n"
            "以 JSON 格式返回：\n"
            '{\n'
            '  "narration": "叙事过渡文本",\n'
            '  "suggestions": ["建议1", "建议2", "建议3"]\n'
            '}'
        )

        messages = [{
            "role": "user",
            "content": (
                f"当前场景NPC：{scene_text}\n"
                f"时间：{self._time_of_day} 天气：{self._weather}\n"
                f"新创建的内容：{result}\n"
                f"玩家的原始指令：{original_cmd}\n\n"
                "请将这个变化自然地融入场景。"
            ),
        }]

        try:
            transition = self.llm.chat_json(system=system, messages=messages)
            narration = transition.get("narration", result)
            suggestions = transition.get("suggestions", [])

            lines = [narration]
            for i, s in enumerate(suggestions, 1):
                lines.append(f"{i}. {s}")
            return "\n\n".join(lines)
        except Exception:
            return result

    def _maybe_trigger_world_simulation(self, user_input: str) -> Optional[str]:
        """世界自发事件：根据游戏进度自动触发 NPC 自主行动。

        触发条件（满足任一）：
        1. 每 3 个玩家回合自动触发一次
        2. 场景时间变化（如黄昏→夜晚）时触发
        3. 场景中有 2+ 个 NPC 且玩家 5 轮未与他们互动

        Returns
        -------
        str or None
            世界事件叙述，如果不需要触发则返回 None。
        """
        if not self.world_builder or not self.scene_npcs:
            self._turn_count += 1
            return None

        self._turn_count += 1
        should_trigger = False
        reason = ""

        # 条件1: 每 3 轮自动触发
        if self._turn_count % 3 == 0:
            should_trigger = True
            reason = "时间流逝"

        # 条件2: 场景时间变化（由 GM Agent 的工具调用自动更新 self._time_of_day）
        current_scene = f"{self._time_of_day}_{self._weather}"
        if self._last_scene_context and self._last_scene_context != current_scene:
            should_trigger = True
            reason = f"时间变为{self._time_of_day}"
        self._last_scene_context = current_scene

        if not should_trigger:
            return None

        # 顺序模拟 NPC 自主行动
        actions = []
        for npc_name in self.scene_npcs[:3]:
            action = self.world_builder.simulate_npc_actions(
                npc_name=npc_name,
                scene_context=self.world.get("description", ""),
                time_of_day=self._time_of_day,
                weather=self._weather,
                other_npc_actions=actions,
            )
            if action:
                actions.append(action)

        if not actions:
            return None

        # 让 GM Agent 用自然的叙述方式包装这些行动
        scene_text = ", ".join(self.scene_npcs)
        actions_text = "\n".join(actions)

        system = (
            "以下 NPC 刚才在当前场景中自主行动了。请用 1-2 句话自然地叙述这些事件。\n"
            "用第二人称描述玩家看到/听到/感知到的变化。\n"
            "不超过 80 字。\n"
            "以 JSON 格式返回：\n"
            '{\n'
            '  "narration": "世界事件叙述",\n'
            '  "world_event": true\n'
            '}'
        )

        messages = [{
            "role": "user",
            "content": (
                f"当前场景NPC：{scene_text}\n"
                f"触发原因：{reason}\n"
                f"NPC 行动：\n{actions_text}\n\n"
                "请将此自然地叙述给玩家。"
            ),
        }]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
            narration = result.get("narration", actions_text)
            return f"\n{self._time_of_day}——{narration}"
        except Exception:
            return "与此同时——\n" + "\n".join(actions)

    def _fallback_response(self, user_input: str) -> str:
        """Generate a simple response when LLM is unavailable."""
        # Check for dice expressions
        m = _DICE_EXPR_RE.search(user_input)
        if m:
            count_str, sides_str, mod_str = m.groups()
            count = int(count_str) if count_str else 1
            sides = int(sides_str)
            modifier = int(mod_str) if mod_str else 0
            expr = f"{count}d{sides}"
            if modifier:
                expr += f"+{modifier}"
            try:
                results, total = roll(expr)
                parts = [f"投掷 {expr} = {' + '.join(str(r) for r in results)}"]
                if modifier:
                    parts.append(f" + {modifier}")
                parts.append(f" = {total}")
                return "".join(parts)
            except ValueError:
                pass

        # Check for info query
        if any(kw in user_input for kw in ["查看", "属性", "状态", "角色卡"]):
            state = self.player_state.get_state()
            return (
                f"{self.player.summary()}\n\n"
                f"【当前状态】\n"
                f"情绪：{state['emotion']}\n"
                f"信任度：{state['trust']}\n"
                f"体力：{state['stamina']}\n"
                f"疯狂值：{state.get('madness', 0)}/100 ({state.get('madness_level', 'sane')})"
            )

        return f"你站在{self.world.get('name', '未知世界')}中，四周弥漫着雾气。冒险等待着你..."

    def _write_npc_memory(self, npc_name: str, raw_interaction: str) -> None:
        """Extract and persist NPC interaction memory using LLM."""
        if not self._llm_available or not self.llm:
            return
        try:
            extracted = self.llm.extract_memory(raw_interaction)
            if not extracted:
                return
            npc_mem_id = self.memory.npc_add(
                content=extracted,
                context={"emotion": self.player_state.get_state()["emotion"]},
                npc_name=npc_name,
            )
            # Link to similar main memories (cross-collection linking)
            similar = self.memory.search(extracted, n=2)
            for s in similar:
                if s.get("id") != npc_mem_id:
                    self.memory.link(npc_mem_id, s["id"], "关联到")
            if self.debug:
                print(f"[DEBUG] NPC 记忆写入 ({npc_name}): {extracted[:80]}")
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] NPC 记忆写入失败 ({npc_name}): {e}")

    def _write_memory(
        self, user_input: str, narration: str, tool_results: List[str]
    ) -> str | None:
        """Extract and persist adventure memory. Returns the new memory ID."""
        if not self._llm_available or not self.llm:
            return None

        combined = f"玩家({self.player.name}): {user_input}\n结果: {narration}"
        if tool_results:
            combined += "\n" + "\n".join(tool_results)

        try:
            extracted = self.llm.extract_memory(combined)
            if not extracted:
                return None

            mem_id = self.memory.add(
                content=extracted,
                context={"emotion": self.player_state.get_state()["emotion"]},
            )
            self._recent_events.append(extracted)

            # Link to semantically similar memories
            similar = self.memory.search(extracted, n=3)
            for s in similar:
                if s.get("id") != mem_id:
                    self.memory.link(mem_id, s["id"], "关联到")

            # Link to the most recent memory (temporal chain)
            if len(self._recent_events) >= 2:
                # Find the second-most-recent memory ID by searching for its content
                prev_content = self._recent_events[-2]
                prev_results = self.memory.search(prev_content, n=1)
                if prev_results and prev_results[0].get("id") != mem_id:
                    self.memory.link(mem_id, prev_results[0]["id"], "发生在...之后")

            if self.debug:
                print(f"[DEBUG] 记忆写入: {extracted}")

            return mem_id
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Opening scene generation
    # ------------------------------------------------------------------

    def generate_opening(self) -> str:
        """Generate the opening scene narration and seed 2-3 starter NPCs.

        The LLM returns JSON with ``narration`` and ``npcs`` so the world
        begins with interactive characters already present.
        """
        if not self._llm_available or not self.llm:
            world_name = self.world.get("name", "未知世界")
            return (
                f"欢迎来到{world_name}。你是{self.player.name}。\n" f"冒险即将开始..."
            )

        world_desc = self.world.get("description", self.world.get("name", ""))
        player_desc = "，".join(self.player.core)
        player_card = self.player.summary()

        # Check for active scenario
        scenario_knowledge = self.knowledge.query(
            "剧本 节点0 开场 核心事件", self.player.name
        )
        scenario_text = "\n".join(scenario_knowledge[:3]) if scenario_knowledge else ""

        # Query for any existing memories (returning player)
        memories = self.memory.search("冒险 故事 经历", n=5)
        memory_text = ""
        if memories:
            memory_text = "\n".join(f"- {m['content']}" for m in memories[:5])

        # List existing backstory NPCs for the GM
        existing_npc_text = ""
        if self.scene_npcs:
            existing_npc_text = (
                "\n以下 NPC 已注册（来自玩家背景故事），请在开场中自然提及他们：\n"
                + "\n".join(f"- {n}" for n in self.scene_npcs)
            )

        if scenario_text:
            system = (
                f"你是 TRPG 主持人，主持《诡秘之主》世界的跑团游戏。\n"
                f"玩家扮演 {self.player.name}，{player_desc}\n"
                f"途径：{getattr(self.player, 'pathway', '未知')}，"
                f"序列{getattr(self.player, 'sequence', 9)}\n"
                f"世界：{world_desc}\n\n"
                f"以下剧本已加载，请严格按照剧本节点0生成开场：\n{scenario_text}\n\n"
                f'{{"narration": "开场叙述（不超过200字）", '
                f'"npcs": [{{"name": "NPC称呼", "core": ["角色描述"], '
                f'"relations": {{"{self.player.name}": "关系"}}}}], '
                f'"suggestions": ["建议1", "建议2", "建议3"]}}\n\n'
                f"要求：\n"
                f"- 严格按照剧本节点0的核心事件生成开场\n"
                f"- 只注册当前物理在场的角色。任务目标、传闻人物只在 narration 提及即可\n"
                f"{existing_npc_text}"
            )
        else:
            system = (
                f"你是 TRPG 主持人，主持《诡秘之主》世界的跑团游戏。\n"
                f"玩家扮演 {self.player.name}，{player_desc}\n"
                f"途径：{getattr(self.player, 'pathway', '未知')}，"
                f"序列{getattr(self.player, 'sequence', 9)}\n"
                f"世界：{world_desc}\n\n"
                f"请设计冒险的开场场景，并以 JSON 格式回复：\n\n"
                f'{{"narration": "开场叙述（不超过200字）", '
                f'"npcs": [{{"name": "NPC称呼", "core": ["角色描述"], '
                f'"relations": {{"{self.player.name}": "关系"}}}}], '
                f'"suggestions": ["建议1", "建议2", "建议3"]}}\n\n'
                f"要求：\n"
                f"- 用第二人称「你」称呼玩家\n"
                f"- 简要设定场景，1-2句环境描写后迅速引入NPC和事件线索\n"
                f"- 只注册当前物理在场的角色。任务目标、传闻人物只在 narration 提及即可\n"
                f"{existing_npc_text}"
            )

        if memory_text:
            system += f"\n以下是之前的冒险记录，请据此生成连续的开场：\n{memory_text}"

        messages = [{"role": "user", "content": "请开始冒险。"}]

        try:
            result = self.llm.chat_json(system=system, messages=messages)

            narration = result.get("narration", "")
            npcs = result.get("npcs", [])
            suggestions = result.get("suggestions", [])

            # Create seed NPCs
            for npc_data in npcs:
                name = npc_data.get("name", "")
                core = npc_data.get("core", [])
                relations = npc_data.get("relations", {})

                if not name or not isinstance(name, str) or not name.strip():
                    continue
                name = name.strip()

                if not isinstance(core, list) or not core:
                    core_text = npc_data.get("core", "")
                    core = (
                        [core_text]
                        if isinstance(core_text, str) and core_text.strip()
                        else [f"{name} — 起始场景角色"]
                    )

                if not isinstance(relations, dict):
                    relations = {}

                existing = self.npc_store.find_by_name(name)
                if existing is not None:
                    if name not in self.scene_npcs:
                        self.scene_npcs.append(name)
                    continue

                self.npc_store.create(
                    name=name,
                    core=core,
                    attributes={
                        "力量": 10,
                        "敏捷": 10,
                        "体质": 10,
                        "智力": 10,
                        "感知": 10,
                        "魅力": 10,
                        "灵性": 10,
                    },
                    relations=relations,
                )
                if name not in self.scene_npcs:
                    self.scene_npcs.append(name)

            if self.debug and npcs:
                print(f"[DEBUG] 开局创建 NPC: {[n.get('name') for n in npcs]}")

            # Treat opening as first GM output — write memory + set context
            opening_text = GameMaster._render_accumulated([narration], suggestions)
            self._last_gm_response = opening_text
            self._recent_events.append(narration)
            self._write_memory("游戏开始", narration, [])

            return opening_text
        except Exception:
            fallback = f"你站在{world_desc}的边缘，冒险即将开始。"
            self._last_gm_response = fallback
            return fallback

    # ===================================================================
    #  SOFT-DELETED: _parse_opening_response
    #  generate_opening() 现在使用 llm.chat_json() 统一解析。
    # ===================================================================
    #
    # def _parse_opening_response(self, raw: str) -> dict:
    #     """Parse the opening scene JSON response from the LLM."""
    #     ...

    # ------------------------------------------------------------------
    #  NPC lookup helper
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    #  Save / Load
    # ------------------------------------------------------------------

    def save(self, path: str = "data/save.json") -> None:
        """Save game state for resumption."""
        import datetime

        data = {
            "time_of_day": self._time_of_day,
            "weather": self._weather,
            "player_state": self.player_state.to_dict(),
            "scene_npcs": self.scene_npcs,
            "recent_events": list(self._recent_events),
            "last_gm_response": self._last_gm_response,
            "turn_count": self._turn_count,
            "last_scene_context": self._last_scene_context,
            "updated_at": datetime.datetime.now().isoformat(),
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_save(self, path: str = "data/save.json") -> bool:
        """Load game state from a save file. Returns True if successful."""
        if not os.path.exists(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._time_of_day = data.get("time_of_day", "黄昏")
        self._weather = data.get("weather", "阴")
        self.scene_npcs = data.get("scene_npcs", [])
        self._recent_events = deque(data.get("recent_events", []), maxlen=10)
        self._last_gm_response = data.get("last_gm_response", "")
        self._turn_count = data.get("turn_count", 0)
        self._last_scene_context = data.get("last_scene_context", "")

        ps = data.get("player_state", {})
        if ps:
            self.player_state = StateMachine.from_dict(ps)

        if self.debug:
            print(f"[DEBUG] 存档加载成功 ({path})")
            print(f"  场景: {self.scene_npcs}")
            print(f"  时间: {self._time_of_day} 天气: {self._weather}")

        return True

    def generate_continuation(self) -> str:
        """Generate a continuation narration after loading a save."""
        if not self._llm_available or not self.llm:
            return "你回到了之前冒险的地方。"

        scene_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无人"
        player_card = self.player.summary()

        system = (
            "玩家刚刚重新进入游戏。根据以下存档信息，生成一段简短的续接叙述（不超过100字），"
            "用第二人称描述玩家当前所处的环境和身边的NPC。不要引入新事件，只做状态回顾。"
        )
        messages = [
            {
                "role": "user",
                "content": (
                    f"时间: {self._time_of_day} 天气: {self._weather}\n"
                    f"场景NPC: {scene_text}\n"
                    f"最近事件: {'; '.join(list(self._recent_events)[-3:])}\n"
                    f"{player_card}\n\n"
                    "请续接。"
                ),
            }
        ]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
            return result.get("narration", "你环顾四周，回忆着之前的经历。")
        except Exception:
            return "你站在之前离开的地方，一切如旧。"

    def _find_npc_in_input(self, user_input: str) -> Optional[str]:
        """Find an NPC name mentioned in user input."""
        for npc_name in self.scene_npcs:
            if npc_name in user_input:
                return npc_name
        # Also search all known NPCs
        for npc in self.npc_store.all():
            if npc.name in user_input:
                return npc.name
        return None


# ===================================================================
#  SOFT-DELETED: keyword-based state update
#  状态触发现在由 LLM GM Agent 在对话中判断。
# ===================================================================
#
# def _update_state_from_keywords(self, text: str) -> List[str]:
#     """在 *text* 中搜索状态触发关键词并更新状态机。"""
#     ...
