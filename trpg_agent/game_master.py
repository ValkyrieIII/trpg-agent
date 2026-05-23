"""GM 核心调度 — 对话调度中枢，聚合所有子系统。

:class:`GameMaster` 是 TRPG Agent 的顶层入口，负责：

- 意图识别（正则匹配，不调 LLM）
- 骰子投掷（dice 意图）
- 角色信息查询（info 意图）
- 事件判定（event 意图）
- 完整对话管线（dialogue 意图）：GM Agent 分析输入 -> 执行检定 -> NPC Agent 扮演
  -> 状态更新 -> 记忆记录 -> 历史管理

典型用法::

    gm = GameMaster("config.yaml")
    reply = gm.process("我推开酒馆的门")
    print(reply)
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import yaml

from trpg_agent.character import Character
from trpg_agent.check import difficulty_check, skill_check
from trpg_agent.dice import roll
from trpg_agent.event import resolve_trigger
from trpg_agent.llm import LLM
from trpg_agent.memory import MemoryStore
from trpg_agent.npc import NPCStore
from trpg_agent.rag import KnowledgeBase
from trpg_agent.state import StateMachine


# ---------------------------------------------------------------------------
#  Intent detection patterns  (regex only — no LLM)
# ---------------------------------------------------------------------------

_PATTERN_DICE = re.compile(r"掷骰|骰子|d\d+", re.IGNORECASE)
_PATTERN_INFO = re.compile(r"查看|属性|状态|角色卡")
_PATTERN_EVENT = re.compile(r"战斗|攻击|射击|触发|陷阱|环境")

# Dice expression embedded in user text (e.g. "3d6+2", "d20")
_DICE_EXPR_RE = re.compile(r"(\d*)d(\d+)(?:\s*\+\s*(\d+))?")

# ---------------------------------------------------------------------------
#  Keyword -> state trigger mapping  (word-level matching)
# ---------------------------------------------------------------------------

_KEYWORD_TRIGGERS: Dict[str, List[str]] = {
    "betrayed": ["背叛", "欺骗", "骗我", "出卖"],
    "helped": ["帮忙", "帮助", "谢谢", "救了我"],
    "combat": ["战斗", "攻击", "受伤", "中招"],
    "rested": ["休息", "睡觉", "扎营", "恢复"],
}

# ---------------------------------------------------------------------------
#  Event keyword -> trigger_type mapping
# ---------------------------------------------------------------------------

_EVENT_DESCRIPTIONS: Dict[str, str] = {
    "combat":        "战斗 攻击 射击 砍杀 挥拳 刺杀 开打 揍 干掉 出手 拔刀 射箭 冲锋 殴打 行刺 袭击",
    "trap":          "陷阱 机关 暗器 触发陷阱 踩到 触发机关 中招",
    "environment":   "环境 天气 地形 攀爬 涉水 寒冷 酷热 暴风雪 迷雾",
    "discovery":     "发现 线索 搜索 观察 调查 仔细看 找线索 检查 侦查",
    "npc_reaction":  "NPC反应 对话 交涉 说服 恐吓 谈判 套话 打听",
}

# Minimum cosine similarity to accept an event match
_EVENT_SIMILARITY_THRESHOLD: float = 0.4

# ---------------------------------------------------------------------------
#  Action -> check rules  (hardcoded regex — no LLM)
# ---------------------------------------------------------------------------
# Each entry: (regex_pattern, check_type, skill_or_dc, modifier)
#   check_type="skill" → skill_check(skill_value, mod)
#   check_type="check" → difficulty_check(dc, mod)

_ACTION_RULES: list[tuple[str, str, object, int]] = [
    (r"(攀爬|爬[上去过]|翻[越墙]|攀登).+", "check", 12, 0),
    (r"(追踪|跟踪|尾行|寻找踪迹|找到踪迹).+", "skill", "追踪", 0),
    (r"(潜行|隐藏|躲[起来藏]|埋伏).+", "skill", "潜行", 0),
    (r"(说服|交涉|谈判|威吓|忽悠|骗).+", "check", 15, 0),
    (r"(撬锁|开锁|解锁|解除机关).+", "skill", "巧手", 0),
    (r"(搜索|搜查|调查|观察|仔细看|找线索|找找).+", "skill", "侦查", 0),
    (r"(跳跃|跳[过去]|跃过).+", "check", 12, 0),
    (r"(搬运|推[开门]|举[起重]|砸).+", "check", 13, 0),
    (r"(射箭|瞄准|拉弓).+", "skill", "弓箭", 0),
    (r"(闪避|躲避|回避|躲开).+", "check", 14, 0),
    (r"(游泳|涉水|过河|渡河).+", "check", 12, 0),
    (r"(忍耐|抵抗|坚持|硬撑).+", "check", 13, 0),
    (r"(攀岩|攀上|抓住).+", "check", 14, 0),
]


# ===================================================================
#  GameMaster
# ===================================================================

class GameMaster:
    """GM 核心调度 — 聚合所有子系统，提供单轮对话入口。

    Parameters
    ----------
    config_path : str
        角色 YAML 配置文件路径。
    llm_api_key : str, optional
        LLM API 密钥。若为 ``None`` 则从环境变量 ``DEEPSEEK_API_KEY`` 读取。
        当 API 密钥不可用时，LLM 操作自动降级为占位符行为。
    knowledge_dir : str, optional
        知识文件所在目录，默认为 ``"data/knowledge"``。
    """

    def __init__(
        self,
        config_path: str,
        llm_api_key: Optional[str] = None,
        knowledge_dir: str = "data/knowledge",
        debug: bool = False,
    ) -> None:
        self.debug = debug
        # -- Player --
        self.player = Character.load(config_path)

        # -- World (from config) --
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            self.world: Dict[str, Any] = config_data.get("world", {})
        except Exception:
            self.world = {}

        # -- NPC Store --
        self.npc_store = NPCStore()
        self.scene_npcs: List[str] = []

        # -- Subsystems --
        self.state = StateMachine()
        # NOTE: MemoryStore 和 KnowledgeBase 各自内部创建了
        # SentenceTransformerEmbeddingFunction("BAAI/bge-small-zh-v1.5") 。
        # 由于 sentence-transformers 库自带模型级缓存，实际上并没有重复
        # 加载模型，此处不做特殊共享处理。
        self.memory = MemoryStore()
        self.knowledge = KnowledgeBase()

        # -- Load knowledge files --
        self.knowledge.load_from_dir(knowledge_dir)

        # -- LLM (graceful fallback to placeholder) --
        if llm_api_key is not None:
            os.environ["DEEPSEEK_API_KEY"] = llm_api_key

        self.llm: Optional[LLM] = None
        self._llm_available = False
        try:
            self.llm = LLM()
            self._llm_available = True
        except (RuntimeError, Exception):
            pass

        # -- Event embedding vectors (pre-computed once) --
        self._event_vectors: Dict[str, Any] = {}
        self._init_event_vectors()

        # -- Player conversation history --
        self.max_history: int = 10
        self.player_history: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    #  Intent detection
    # ------------------------------------------------------------------

    def _detect_intent(self, user_input: str) -> str:
        """通过正则匹配识别用户意图（不调 LLM）。

        Returns
        -------
        str
            ``"dice"`` / ``"info"`` / ``"event"`` / ``"dialogue"``
        """
        if _PATTERN_DICE.search(user_input):
            return "dice"
        if _PATTERN_INFO.search(user_input):
            return "info"
        if _PATTERN_EVENT.search(user_input):
            return "event"
        return "dialogue"

    # ------------------------------------------------------------------
    #  Intent handlers
    # ------------------------------------------------------------------

    def _handle_dice(self, user_input: str) -> str:
        """处理骰子投掷意图。

        - 从用户输入中提取骰子表达式
        - 调用 :func:`dice.roll` 进行投掷
        - 返回格式化后的投掷结果
        """
        m = _DICE_EXPR_RE.search(user_input)
        if not m:
            return "请说出要掷的骰子，例如：d20、3d6、2d6+3"

        count_str, sides_str, mod_str = m.groups()
        count = int(count_str) if count_str else 1
        sides = int(sides_str)
        modifier = int(mod_str) if mod_str else 0

        expr = f"{count}d{sides}"
        if modifier:
            expr += f"+{modifier}"

        try:
            results, total = roll(expr)
        except ValueError as e:
            return f"骰子表达式错误：{e}"

        parts = [f"投掷 {expr} = {' + '.join(str(r) for r in results)}"]
        if modifier:
            parts.append(f" + {modifier}")
        parts.append(f" = {total}")
        return "".join(parts)

    def generate_opening(self) -> str:
        """Generate the opening scene narration based on world, player, and memories.

        Returns a GM-narrated opening to display when the game starts.
        """
        if not self._llm_available or not self.llm:
            world_name = self.world.get("name", "未知世界")
            return (
                f"欢迎来到{world_name}。你是{self.player.name}。\n"
                f"冒险即将开始..."
            )

        world_desc = self.world.get("description", self.world.get("name", ""))
        player_desc = "，".join(self.player.core)
        player_card = self.player.summary()

        # Query for any existing memories (returning player)
        memories = self.memory.search("冒险 故事 经历", n=5)
        memory_text = ""
        if memories:
            memory_text = "\n".join(f"- {m['content']}" for m in memories[:5])

        system = (
            f"你是 TRPG 地下城主。\n"
            f"玩家扮演 {self.player.name}，{player_desc}\n"
            f"世界：{world_desc}\n\n"
            f"请为这次冒险写一段开场白（不超过200字）：\n"
            f"- 设定场景氛围和起始地点\n"
            f"- 用第二人称「你」称呼玩家\n"
            f"- 让玩家知道周围环境，暗示可能的行动方向\n"
        )

        if memory_text:
            system += (
                f"\n以下是之前的冒险记录，请据此生成连续的开场：\n{memory_text}"
            )

        messages = [{"role": "user", "content": "请开始冒险。"}]

        try:
            narration = self.llm.chat(system=system, messages=messages)
            return narration
        except Exception:
            return f"你站在{world_desc}的边缘，冒险即将开始。"

    def _handle_info(self) -> str:
        """处理角色信息/状态查询意图。

        返回角色摘要 + 当前情绪/信任度/体力状态。
        """
        state = self.state.get_state()
        return (
            f"{self.player.summary()}\n\n"
            f"【当前状态】\n"
            f"情绪：{state['emotion']}\n"
            f"信任度：{state['trust']}\n"
            f"体力：{state['stamina']}"
        )

    # ------------------------------------------------------------------
    #  Embedding-based event routing
    # ------------------------------------------------------------------

    def _init_event_vectors(self) -> None:
        """Pre-compute event description embeddings for fast similarity matching."""
        import numpy as np

        try:
            embed_fn = self.memory._embedding_fn
            for event_type, desc in _EVENT_DESCRIPTIONS.items():
                vectors = embed_fn([desc])
                if vectors and len(vectors) > 0:
                    self._event_vectors[event_type] = np.array(vectors[0])
        except Exception:
            pass

    def _classify_event(self, user_input: str) -> str | None:
        """Classify event type by embedding similarity.

        Returns the best-matching event type, or ``None`` if no type exceeds
        the similarity threshold.
        """
        import numpy as np

        if not self._event_vectors:
            return None

        try:
            embed_fn = self.memory._embedding_fn
            input_vecs = embed_fn([user_input])
            if not input_vecs or len(input_vecs) == 0:
                return None

            input_vec = np.array(input_vecs[0])
            best_type = None
            best_sim = -1.0
            for event_type, center_vec in self._event_vectors.items():
                sim = np.dot(input_vec, center_vec) / (
                    np.linalg.norm(input_vec) * np.linalg.norm(center_vec)
                )
                if sim > best_sim:
                    best_sim = sim
                    best_type = event_type

            if best_sim < _EVENT_SIMILARITY_THRESHOLD:
                return None
            return best_type
        except Exception:
            return None

    def _handle_event(self, user_input: str) -> str:
        """处理事件触发意图。

        通过 embedding 语义相似度确定触发类型，调用 :func:`event.resolve_trigger`
        进行判定，返回判定结果叙事描述。
        """
        trigger_type = self._classify_event(user_input)
        if trigger_type is None:
            trigger_type = "environment"

        if self.debug:
            print(f"[DEBUG] 事件分类: {trigger_type}")

        result = resolve_trigger(
            trigger_type=trigger_type,
            character=self.player,
            state=self.state,
        )

        narrative = result.get("narrative", "")
        state_changes = result.get("state_changes", [])
        if state_changes:
            narrative += f"（状态变化：{', '.join(state_changes)}）"

        if self.debug:
            print(f"[DEBUG] 事件结果: {narrative}")

        return narrative

    # ------------------------------------------------------------------
    #  Action matching & check execution
    # ------------------------------------------------------------------

    def _match_action(self, user_input: str) -> dict | None:
        """硬编码正则匹配行动，返回检定参数。无匹配返回 None。"""
        for pattern, check_type, skill_or_dc, mod in _ACTION_RULES:
            m = re.search(pattern, user_input)
            if m:
                action_desc = m.group(0)
                # 提取行动关键词
                action_key = m.group(1) if m.lastindex else action_desc
                return {
                    "action_desc": action_desc,
                    "action_key": action_key,
                    "check_type": check_type,
                    "skill_or_dc": skill_or_dc,
                    "modifier": mod,
                }
        return None

    def _execute_check(self, action: dict) -> dict:
        """执行检定，返回检定结果叙事片段。"""
        check_type = action["check_type"]
        skill_or_dc = action["skill_or_dc"]
        mod = action["modifier"]
        action_desc = action["action_desc"]

        if check_type == "skill":
            skill_name = skill_or_dc
            # skills 是 list[dict]，查找匹配技能名
            skill_value = 50
            for s in self.player.skills:
                if s["name"] == skill_name:
                    skill_value = s["value"]
                    break
            result = skill_check(skill_value, mod)
            check_label = f"{skill_name}检定"
            target_str = str(skill_value)
        else:
            dc = int(skill_or_dc)
            result = difficulty_check(dc, mod)
            check_label = f"难度检定 DC{dc}"
            target_str = str(dc)

        if result["success"]:
            return {
                "success": True,
                "narrative": (
                    f"【行动】{action_desc}\n"
                    f"【{check_label}】{result['detail']}"
                ),
            }
        else:
            return {
                "success": False,
                "stamina_cost": True,
                "narrative": (
                    f"【行动】{action_desc}\n"
                    f"【{check_label}】{result['detail']}"
                ),
            }

    # ------------------------------------------------------------------
    #  GM agent system prompt
    # ------------------------------------------------------------------

    _GM_SYSTEM = (
        "你是 TRPG 地下城主(Game Master)。\n"
        "玩家扮演 {player_name}，{player_desc}\n\n"
        "## 职责\n"
        "1. 叙述场景 — 用第三人称客观描述玩家看到、听到、感受到的一切。\n"
        "2. 判断检定 — 玩家用括号声明行动时（如'(我拔出短刀)'），判断是否需要检定并执行。纯扮演动作（微笑、点头）无需检定。\n"
        "3. 扮演 NPC — 当玩家对 NPC 说话或互动时，你需要以该 NPC 的身份说话。\n\n"
        "## 规则\n"
        "- 不要替玩家角色说话或替 ta 做决定。\n"
        "- 不要描述玩家角色的内心感受（'你觉得...'），只描述客观事实。\n"
        "- 每段叙述不超过 150 字。\n"
        "- 当玩家询问不在场的 NPC 时，直接说明 ta 不在。\n"
        "- 你可以在叙事中引入新 NPC，引入后该 NPC 将持续存在于世界中。\n\n"
        "## 当前场景已知 NPC\n"
        "{scene_npcs}\n\n"
        "## 玩家角色卡\n"
        "{player_card}"
    )

    # ------------------------------------------------------------------
    #  GM Agent (v2 — JSON response)
    # ------------------------------------------------------------------

    def _call_gm(self, user_input: str, knowledge: list[str], memories: list[dict]) -> dict:
        """GM Agent: 分析玩家输入，返回结构化 JSON。

        Returns
        -------
        dict
            Keys: ``narration``, ``check``, ``responding_npc``, ``new_npc``.
        """
        player_desc = " ".join(self.player.core)

        scene_npcs_text = "暂无已知 NPC"
        if self.scene_npcs:
            scene_npcs_text = "\n".join(f"- {n}" for n in self.scene_npcs)

        player_card = self.player.summary()

        # -- Build system prompt --
        system_parts = [
            self._GM_SYSTEM.format(
                player_name=self.player.name,
                player_desc=player_desc,
                scene_npcs=scene_npcs_text,
                player_card=player_card,
            ),
        ]

        # Inject knowledge
        if knowledge:
            system_parts.append("\n## 相关知识\n" + "\n".join(knowledge[:3]))

        # Inject memories
        if memories:
            system_parts.append(
                "\n## 最近事件\n"
                + "\n".join(m["content"] for m in memories[:3])
            )

        # Inject check rules
        system_parts.append(
            "\n\n## 检定规则\n"
            "玩家用括号声明行动时（如'(我拔出短刀)'）判断是否需要检定。\n"
            "纯扮演动作（微笑、点头、摇头、叹气等）无需检定。\n"
            "如果你认为需要检定，请在 check 字段返回包含检定信息的对象。\n"
            "如果你认为不需要，check 字段为 null。"
        )

        system_prompt = "\n".join(system_parts)

        gm_messages = [
            {
                "role": "user",
                "content": (
                    f"玩家说：{user_input}\n\n"
                    "请以 JSON 格式回复，包含以下字段：\n"
                    '{"narration": "场景叙述", "check": null, '
                    '"responding_npc": "NPC短名（2-6字）或null", '
                    '"new_npc": "新NPC短名（2-6字）或null"}\n\n'
                    "注意：responding_npc 和 new_npc 必须用简短的人名（如'老马'、'老板娘'），"
                    "不要用长描述。"
                ),
            },
        ]

        if self._llm_available and self.llm:
            try:
                raw = self.llm.chat(system=system_prompt, messages=gm_messages)
                return self._parse_gm_response(raw)
            except Exception:
                pass

        return {"narration": "", "check": None, "responding_npc": None, "new_npc": None}

    def _parse_gm_response(self, raw: str) -> dict:
        """解析 GM 的 JSON 响应，失败时将全部文本视为 narration。"""
        text = raw.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        # Attempt JSON parse
        try:
            result = json.loads(text)
            if isinstance(result, dict):
                return {
                    "narration": result.get("narration", ""),
                    "check": result.get("check"),
                    "responding_npc": result.get("responding_npc"),
                    "new_npc": result.get("new_npc"),
                }
        except json.JSONDecodeError:
            pass

        # Fallback: entire response is narration
        return {"narration": raw, "check": None, "responding_npc": None, "new_npc": None}

    # ------------------------------------------------------------------
    #  Dialogue handler (v2 — GM + NPC Agent pipeline)
    # ------------------------------------------------------------------

    def _handle_dialogue(self, user_input: str) -> str:
        """处理对话意图 — GM Agent + NPC Agent 管线。

        流程
        ----
        1. 硬编码行动匹配 (_match_action)
        2. 检索（记忆 + 知识）
        3. GM Agent (_call_gm) → {narration, check, responding_npc, new_npc}
        4. 需要时执行检定
        5. 新 NPC 创建
        6. NPC Agent 扮演
        7. 组装输出
        8. 状态更新
        9. 记忆记录
        10. 更新 player_history
        """
        # ---- Step 0: 硬编码匹配行动 ----
        action = self._match_action(user_input)

        if self.debug and action:
            print(f"[DEBUG] 硬编码行动匹配: {action['action_desc']} → {action['check_type']}({action['skill_or_dc']})")

        # ---- Step 1: 检索 ----
        memories = self.memory.full_retrieve(user_input)
        knowledge = self.knowledge.query(user_input, self.player.name)

        if self.debug:
            print(f"[DEBUG] 记忆检索: {len(memories)}条")
            print(f"[DEBUG] 知识检索: {len(knowledge)}条")

        # ---- Step 2: GM Agent ----
        gm_result = self._call_gm(user_input, knowledge, memories)

        narration = gm_result.get("narration", "")
        check_info = gm_result.get("check")
        responding_npc = gm_result.get("responding_npc")
        new_npc_name = gm_result.get("new_npc")

        if self.debug:
            print(f"[DEBUG] GM 叙事: {narration[:60]}{'...' if len(narration) > 60 else ''}")
            print(f"[DEBUG] GM 检定判定: {check_info}")
            print(f"[DEBUG] GM 指定 NPC: {responding_npc or '无'}")
            if new_npc_name:
                print(f"[DEBUG] GM 创建新 NPC: {new_npc_name}")

        # ---- Step 3: 执行检定（GM 判断需要时） ----
        check_result = None
        if check_info is not None and action is not None:
            check_result = self._execute_check(action)
            if self.debug:
                print(f"[DEBUG] 检定结果: {check_result['narrative']}")

        # ---- Step 4: 处理新 NPC 创建 ----
        if new_npc_name and isinstance(new_npc_name, str):
            existing = self.npc_store.find_by_name(new_npc_name)
            if existing is None:
                self.npc_store.create(
                    name=new_npc_name,
                    core=[f"{new_npc_name} — 由 GM 引入的角色"],
                    attributes={
                        "strength": 10,
                        "agility": 10,
                        "intelligence": 10,
                        "willpower": 10,
                    },
                )
            if new_npc_name not in self.scene_npcs:
                self.scene_npcs.append(new_npc_name)

        # ---- Step 5: NPC Agent 扮演 ----
        npc_reply = None
        if responding_npc and isinstance(responding_npc, str):
            npc = self.npc_store.find_by_name(responding_npc)
            if npc is None:
                results = self.npc_store.search(responding_npc)
                if results:
                    npc = results[0]

            if npc is not None:
                if self.debug:
                    hist_len = len(self.npc_store.get_history(responding_npc))
                    print(f"[DEBUG] NPC Agent 调用: {responding_npc} (历史{hist_len}轮)")

                npc_system = (
                    npc.build_personality_prompt()
                    + "\n\n"
                    + npc.build_state_prompt({
                        "emotion": "calm",
                        "trust": 0.5,
                        "stamina": "fresh",
                    })
                )
                npc_messages = list(self.npc_store.get_history(responding_npc))
                npc_messages.append({
                    "role": "user",
                    "content": f"{self.player.name}对{responding_npc}说：{user_input}",
                })

                if self._llm_available and self.llm:
                    try:
                        npc_reply = self.llm.chat(
                            system=npc_system,
                            messages=npc_messages,
                        )
                    except Exception:
                        npc_reply = "(NPC 模块暂时不可用)"
                else:
                    npc_reply = "(LLM 模块尚未实现)"

                self.npc_store.append_history(
                    responding_npc, "user", f"{self.player.name}: {user_input}",
                )
                self.npc_store.append_history(
                    responding_npc, "assistant", npc_reply,
                )

        # ---- Step 6: 组装输出 ----
        parts: List[str] = []
        if narration:
            parts.append(f"[GM]: {narration}")
        if check_result:
            parts.append(check_result["narrative"])
        if npc_reply:
            npc_label = responding_npc or "NPC"
            parts.append(f"[{npc_label}]: {npc_reply}")

        response = "\n\n".join(parts) if parts else narration

        # ---- Step 7: 状态更新 ----
        combined_parts = [user_input]
        if narration:
            combined_parts.append(narration)
        if npc_reply:
            combined_parts.append(npc_reply)
        combined = " ".join(combined_parts)

        for trigger, keywords in _KEYWORD_TRIGGERS.items():
            for kw in keywords:
                if kw in combined:
                    self.state.apply(trigger)
                    break
        if check_result and check_result.get("stamina_cost"):
            self.state.apply("combat")

        # ---- Step 8: 记录记忆 ----
        if self._llm_available and self.llm:
            try:
                dialogue_for_memory = f"用户：{user_input}"
                if narration:
                    dialogue_for_memory += f"\nGM：{narration}"
                if npc_reply:
                    npc_label = responding_npc or "NPC"
                    dialogue_for_memory += f"\n{npc_label}：{npc_reply}"
                extracted = self.llm.extract_memory(dialogue_for_memory)
            except Exception:
                extracted = ""
        else:
            extracted = ""

        if extracted:
            self.memory.add(
                content=extracted,
                context={"emotion": self.state.get_state()["emotion"]},
            )

        # ---- Step 9: 更新 player_history ----
        self.player_history.append({"role": "user", "content": user_input})
        self.player_history.append({"role": "assistant", "content": response})

        if len(self.player_history) > self.max_history * 2:
            self.player_history = self.player_history[-self.max_history * 2 :]

        return response

    # ------------------------------------------------------------------
    #  State keyword matching (shared helper)
    # ------------------------------------------------------------------

    def _update_state_from_keywords(self, text: str) -> List[str]:
        """在 *text* 中搜索状态触发关键词并更新状态机。

        Parameters
        ----------
        text : str
            待搜索的文本（通常为用户输入）。

        Returns
        -------
        list of str
            被触发的 trigger 名称列表。
        """
        fired: List[str] = []
        for trigger, keywords in _KEYWORD_TRIGGERS.items():
            for kw in keywords:
                if kw in text:
                    self.state.apply(trigger)
                    fired.append(trigger)
                    break
        return fired

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        """单轮对话处理入口 — 三层判定管线。

        流程
        ----
        1. 正则判定：dice / info / event → 直接路由
        2. 正则判定为 dialogue → embedding 判定
        3. embedding 匹配事件 → 事件处理
        4. embedding 未匹配 → LLM (GM Agent) 最终判定
        5. LLM 判定为对话 → NPC 对话管线

        Parameters
        ----------
        user_input : str
            玩家的输入文本。

        Returns
        -------
        str
            GM 的回复文本。
        """
        user_input = user_input.strip()
        if not user_input:
            return "请说点什么吧。"

        intent = self._detect_intent(user_input)

        if self.debug:
            print(f"\n{'─'*50}")
            print(f"[DEBUG] 玩家输入: {user_input}")
            print(f"[DEBUG] 正则意图: {intent}")

        # ---- Tier 1: 正则直接路由 ----
        if intent == "dice":
            response = self._handle_dice(user_input)
            self._update_state_from_keywords(user_input)
        elif intent == "info":
            response = self._handle_info()
            self._update_state_from_keywords(user_input)
        elif intent == "event":
            response = self._handle_event(user_input)
            self._update_state_from_keywords(user_input)
        else:
            # ---- Tier 2: embedding 判定 ----
            event_type = self._classify_event(user_input)
            if self.debug:
                print(f"[DEBUG] embedding 分类: {event_type or '无匹配'}")
            if event_type is not None:
                # embedding 匹配到具体事件类型 → 事件处理
                response = self._handle_event(user_input)
                self._update_state_from_keywords(user_input)
            else:
                # ---- Tier 3: LLM (GM Agent) 最终判定 ----
                response = self._handle_dialogue(user_input)

        if self.debug:
            state = self.state.get_state()
            print(f"[DEBUG] 当前状态: 情绪={state['emotion']} 信任={state['trust']} 体力={state['stamina']}")
            print(f"[DEBUG] 场景NPC: {self.scene_npcs}")
            print(f"{'─'*50}")

        return response
