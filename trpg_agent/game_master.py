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

    def _handle_dialogue(self, user_input: str) -> str:
        """处理对话意图 — 完整对话管线。

        步骤
        ----
        1. 检索记忆 + 知识
        2. 组装 system prompt（人格 + 状态 + 知识 + 记忆）
        3. 组装 messages（最近历史）
        4. 调用 LLM.chat()（不可用时返回占位字符串）
        5. 关键词触发状态更新
        6. LLM.extract_memory() -> memory.add()（占位，跳过）
        7. 更新对话历史
        """
        # ---- Step 1: 检索 ----
        memories = self.memory.full_retrieve(user_input)
        knowledge = self.knowledge.query(user_input, self.character.name)

        # ---- Step 2: 组装 system prompt ----
        system_parts: List[str] = [
            self.character.build_personality_prompt(),
            self.character.build_state_prompt(self.state.get_state()),
        ]

        if knowledge:
            system_parts.append("【相关知识】")
            system_parts.extend(knowledge)

        if memories:
            system_parts.append("【相关记忆】")
            for mem in memories[:3]:
                system_parts.append(f"- {mem['content']}")

        system_prompt = "\n\n".join(system_parts)

        # ---- Step 3: 组装 messages ----
        messages: List[Dict[str, str]] = list(self.history)
        messages.append({"role": "user", "content": user_input})

        # ---- Step 4: 调用 LLM ----
        if self._llm_available and self.llm is not None:
            try:
                response = self.llm.chat(system=system_prompt, messages=messages)
            except Exception:
                response = "LLM 模块尚未实现"
        else:
            response = "LLM 模块尚未实现"

        # ---- Step 5: 状态更新（关键词触发） ----
        combined = f"{user_input} {response}"
        for trigger, keywords in _KEYWORD_TRIGGERS.items():
            for kw in keywords:
                if kw in combined:
                    self.state.apply(trigger)
                    break

        # ---- Step 6: 记录记忆（占位 — extract_memory 返回空字符串时跳过） ----
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

        # ---- Step 7: 更新对话历史 ----
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
