"""GM 核心调度 — 对话调度中枢，聚合所有子系统。

:class:`GameMaster` 是 TRPG Agent 的顶层入口，负责：

- 意图识别（正则匹配，不调 LLM）
- 骰子投掷（dice 意图）
- 角色信息查询（info 意图）
- 事件判定（event 意图）
- 完整对话管线（dialogue 意图）：检索 -> 组装 prompt -> LLM 生成 -> 状态更新
  -> 记忆记录 -> 历史管理

典型用法::

    gm = GameMaster("configs/character.yaml")
    reply = gm.process("掷骰 d20")
    print(reply)
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

from trpg_agent.character import Character
from trpg_agent.check import difficulty_check, skill_check
from trpg_agent.dice import roll
from trpg_agent.event import resolve_trigger
from trpg_agent.llm import LLM
from trpg_agent.memory import MemoryStore
from trpg_agent.rag import KnowledgeBase
from trpg_agent.state import StateMachine


# ---------------------------------------------------------------------------
#  Intent detection patterns  (regex only — no LLM)
# ---------------------------------------------------------------------------

_PATTERN_DICE = re.compile(r"掷骰|骰子|d\d+", re.IGNORECASE)
_PATTERN_INFO = re.compile(r"查看|属性|状态|角色卡")
_PATTERN_EVENT = re.compile(r"战斗|攻击|触发|陷阱|环境")

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

_EVENT_KEYWORDS: Dict[str, str] = {
    "陷阱": "trap",
    "环境": "environment",
    "发现": "discovery",
    "线索": "discovery",
    "NPC": "npc_reaction",
    "npc": "npc_reaction",
    "反应": "npc_reaction",
}

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
    (r"(射击|射箭|瞄准|拉弓).+", "skill", "弓箭", 0),
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
    ) -> None:
        # -- Character --
        self.character = Character.load(config_path)

        # -- Subsystems --
        self.state = StateMachine()
        # NOTE: MemoryStore 和 KnowledgeBase 各自内部创建了
        # SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2") 。
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

        # -- Conversation history --
        self.max_history: int = 10
        self.history: List[Dict[str, str]] = []

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

    def _handle_info(self) -> str:
        """处理角色信息/状态查询意图。

        返回角色摘要 + 当前情绪/信任度/体力状态。
        """
        state = self.state.get_state()
        return (
            f"{self.character.summary()}\n\n"
            f"【当前状态】\n"
            f"情绪：{state['emotion']}\n"
            f"信任度：{state['trust']}\n"
            f"体力：{state['stamina']}"
        )

    def _handle_event(self, user_input: str) -> str:
        """处理事件触发意图。

        通过关键词匹配确定触发类型，调用 :func:`event.resolve_trigger` 进行判定，
        返回判定结果叙事描述。
        """
        # 默认触发类型
        trigger_type = "environment"
        for keyword, tt in _EVENT_KEYWORDS.items():
            if keyword in user_input:
                trigger_type = tt
                break

        result = resolve_trigger(
            trigger_type=trigger_type,
            character=self.character,
            state=self.state,
        )

        narrative = result.get("narrative", "")
        state_changes = result.get("state_changes", [])
        if state_changes:
            narrative += f"（状态变化：{', '.join(state_changes)}）"

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
            for s in self.character.skills:
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

    def _handle_dialogue(self, user_input: str) -> str:
        """处理对话意图 — 完整对话管线。

        步骤
        ----
        0. 硬编码匹配行动 → 触发检定（无匹配则跳过）
        1. 执行检定（Python，复用 check.py）
        2. 检索记忆 + 知识
        3. 组装 system prompt（人格 + 检定结果 + 状态 + 知识 + 记忆）
        4. 组装 messages（最近历史）
        5. 调用 LLM.chat()（一次完成旁白+角色对话）
        6. 关键词触发状态更新 + 检定失败扣体力
        7. 记录记忆
        8. 更新对话历史
        """
        # ---- Step 0: 硬编码匹配行动 ----
        action = self._match_action(user_input)

        # ---- Step 1: 执行检定（仅当匹配到行动时） ----
        check_result = None
        if action:
            check_result = self._execute_check(action)

        # ---- Step 2: 检索 ----
        memories = self.memory.full_retrieve(user_input)
        knowledge = self.knowledge.query(user_input, self.character.name)

        # ---- Step 3: 组装 system prompt ----
        system_parts: List[str] = [
            self.character.build_personality_prompt(),
            self.character.build_state_prompt(self.state.get_state()),
        ]

        # 检定结果 → GM 叙述指令
        if check_result:
            system_parts.append(
                "## GM 检定结果\n"
                f"{check_result['narrative']}\n\n"
                "请在回复中用第三人称叙述这个行动的过程和结果，空行后以角色身份说话。\n"
                "检定成功 → 描述动作利落完成\n"
                "检定失败 → 描述动作失败或遭遇困难"
            )
        else:
            system_parts.append(
                "## 纯对话模式\n"
                "玩家在进行对话或社交，不需要旁白叙述。直接以角色身份回应。"
            )

        if knowledge:
            system_parts.append("【相关知识】")
            system_parts.extend(knowledge)

        if memories:
            system_parts.append("【相关记忆】")
            for mem in memories[:3]:
                system_parts.append(f"- {mem['content']}")

        system_prompt = "\n\n".join(system_parts)

        # ---- Step 4: 组装 messages ----
        messages: List[Dict[str, str]] = list(self.history)
        messages.append({"role": "user", "content": user_input})

        # ---- Step 5: 调用 LLM ----
        if self._llm_available and self.llm is not None:
            try:
                response = self.llm.chat(system=system_prompt, messages=messages)
            except Exception:
                response = "LLM 模块尚未实现"
        else:
            response = "LLM 模块尚未实现"

        # ---- Step 6: 状态更新 ----
        combined = f"{user_input} {response}"
        for trigger, keywords in _KEYWORD_TRIGGERS.items():
            for kw in keywords:
                if kw in combined:
                    self.state.apply(trigger)
                    break
        # 检定失败 → 消耗体力
        if check_result and check_result.get("stamina_cost"):
            self.state.apply("combat")

        # ---- Step 7: 记录记忆 ----
        if self._llm_available and self.llm is not None:
            try:
                extracted = self.llm.extract_memory(
                    f"用户：{user_input}\n你：{response}"
                )
            except Exception:
                extracted = ""
        else:
            extracted = ""

        if extracted:
            self.memory.add(
                content=extracted,
                context={"emotion": self.state.get_state()["emotion"]},
            )

        # ---- Step 8: 更新对话历史 ----
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        if len(self.history) > self.max_history * 2:
            # TODO: 摘要压缩旧历史，而不是直接丢弃
            self.history = self.history[-self.max_history * 2 :]

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
        """单轮对话处理入口。

        流程
        ----
        1. 意图识别（正则）
        2. 分发到对应的 handler
        3. 返回回复文本

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
            # dialogue handler 内部已包含 Step 5 状态更新
            response = self._handle_dialogue(user_input)

        return response
