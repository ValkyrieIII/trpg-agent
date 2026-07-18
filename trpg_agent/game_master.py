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

import asyncio
import json
import os
import re
import time
from collections import deque
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional

import yaml

from agents import Runner

from trpg_agent.event_stream import StatusEvent, make_tool_event, AgentStatusHooks
from trpg_agent.agent_config import (
    GameContext,
    configure_deepseek,
    create_gm_agent,
    build_gm_instructions,
)
from trpg_agent.character import Character
from trpg_agent.dice import roll
from trpg_agent.llm import LLM
from trpg_agent.memory import MemoryStore
from trpg_agent.npc import NPCStore
from trpg_agent.rag import KnowledgeBase
from trpg_agent.state import StateMachine, calc_max_hp

# Dice expression embedded in user text (e.g. "3d6+2", "d20") — used by _fallback_response()
_DICE_EXPR_RE = re.compile(r"(\d*)d(\d+)(?:\s*\+\s*(\d+))?")


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
        self._debug_log: List[str] = []
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
            # Configure OpenAI Agents SDK to use DeepSeek
            configure_deepseek()
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
        self.player_state = StateMachine(
            max_hp=calc_max_hp(self.player.attributes),
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

        # -- World state tracking (set by opening/LGM; defaults for empty state) --
        self._time_of_day: str = "清晨"
        self._weather: str = "薄雾"

        # -- Game over flag (two-phase: pending → confirm → cleanup) --
        self._game_over: bool = False
        self._game_over_pending: bool = False
        self._game_over_cause: str = ""

        # -- Previous turn response (for suggestion continuity) --
        self._last_gm_response: str = ""
        self._last_suggestions: list[str] = []

        # -- World simulation counters (auto NPC actions) --
        self._turn_count: int = 0
        self._last_scene_context: str = ""
        self._world_event_counter: int = 0

        # -- Register backstory NPCs (LLM must be connected first) --
        if self._llm_available:
            self._register_backstory_npcs()
        else:
            self._debug("[DEBUG] LLM 未连接，跳过后设故事 NPC 注册")

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

        # -- OpenAI Agents SDK: GM Agent + conversation history --
        self.history_messages: list[dict[str, str]] = []
        self._npc_agents: dict[str, Any] = {}
        if self._llm_available:
            from trpg_agent.tools import (
                roll_dice, difficulty_check, skill_check, combat_attack,
                get_player_state, get_npc_state,
                create_npc, remove_npc, set_scene, game_over,
                search_knowledge, search_memory, invoke_npc, invoke_npcs,
            )
            self.gm_agent = create_gm_agent(tools=[
                roll_dice, difficulty_check, skill_check, combat_attack,
                get_player_state, get_npc_state,
                create_npc, invoke_npc, invoke_npcs, remove_npc, set_scene, game_over,
                search_knowledge, search_memory,
            ])
        else:
            self.gm_agent = None

        # -- Tool Advisor (zero-tools agent for tool suggestion) --
        if self._llm_available:
            from trpg_agent.agent_config import create_tool_advisor
            self.tool_advisor = create_tool_advisor()
        else:
            self.tool_advisor = None

        # -- Multi-Agent: Judge + Narrator + Orchestrator (new architecture) --
        self.legacy_mode: bool = os.environ.get("TRPG_LEGACY_MODE", "").lower() in ("1", "true", "yes")
        self._judge_agent = None
        self._narrator_agent = None
        self._orchestrator_agent = None
        if self._llm_available and not self.legacy_mode:
            # Judge Agent
            from trpg_agent.judge import create_judge_agent
            self._judge_agent = create_judge_agent()
            # Narrator Agent
            from trpg_agent.narrator import create_narrator_agent
            self._narrator_agent = create_narrator_agent()
            # Orchestrator Agent — needs ALL tools including sub-agent callers
            from trpg_agent.tools import (
                roll_dice, difficulty_check, skill_check, combat_attack,
                get_player_state, get_npc_state,
                create_npc, invoke_npc, invoke_npcs, remove_npc, set_scene, game_over,
                search_knowledge, search_memory,
                invoke_judge, invoke_narrator, broadcast_event,
            )
            from trpg_agent.orchestrator import create_orchestrator_agent
            self._orchestrator_agent = create_orchestrator_agent(tools=[
                invoke_judge, invoke_narrator, broadcast_event,
                invoke_npc, invoke_npcs,
                search_memory, search_knowledge,
                create_npc, remove_npc, set_scene,
                get_player_state, get_npc_state, game_over,
            ])
            self._debug("[Agent] Orchestrator + Judge + Narrator 已就绪")

        # -- Sync NPC Agents for any NPCs already in the store (backstory, etc.) --
        if self._llm_available:
            self._sync_npc_agents()

    def _debug(self, msg: str) -> None:
        """Log a debug message. Prints to stdout and stores in an internal buffer
        that the API server can expose via /api/debug."""
        if self.debug:
            print(msg)
        self._debug_log.append(msg)
        # 防止 CLI 长时间运行时缓冲区无限增长
        if len(self._debug_log) > 500:
            self._debug_log = self._debug_log[-300:]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for Chinese-mixed text.

        Chinese characters: ~1.8 chars/token.  English: ~4 chars/token.
        """
        if not text:
            return 0
        cjk = sum(1 for c in text if '一' <= c <= '鿿' or '　' <= c <= '〿')
        other = len(text) - cjk
        return max(1, int(cjk / 1.8 + other / 4))

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

    def _generate_npc_attributes(
        self, name: str, core: list[str], personality_tone: str
    ) -> dict[str, int]:
        """Generate NPC attributes based on role description via LLM.

        Falls back to balanced defaults (10) if LLM is unavailable.
        """
        if not self._llm_available or not self.llm:
            return {
                "力量": 10, "敏捷": 10, "体质": 10,
                "智力": 10, "感知": 10, "魅力": 10,
            }

        core_text = "\n".join(core)
        try:
            result = self.llm.chat_json(
                system="你是TRPG角色设计师。根据角色描述，为该角色分配合理的属性值（1-20范围）。属性包括：力量、敏捷、体质、智力、感知、魅力。请根据角色特点合理分配，不要所有属性都一样。以JSON格式返回：{\"力量\": 12, \"敏捷\": 10, \"体质\": 11, \"智力\": 14, \"感知\": 13, \"魅力\": 8}",
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

                attributes = self._generate_npc_attributes(name, core, tone)

                self.npc_store.create(
                    name=name,
                    core=core,
                    attributes=attributes,
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

                self._debug(f"[DEBUG] 注册后设故事 NPC: {name} (语调: {tone})")

        except Exception as e:
            self._debug(f"[DEBUG] 后设故事 NPC 提取失败: {e}")

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


    # ------------------------------------------------------------------
    #  Context builder
    # ------------------------------------------------------------------

    def _run_tool_advisor(self, user_input: str) -> str:
        """Query the Tool Advisor for tool suggestions (completely isolated, no tools)."""
        if self.tool_advisor is None:
            return ""
        try:
            advisor_input = (
                f"场景: {self._time_of_day} {self._weather} | "
                f"NPC: {', '.join(self.scene_npcs) if self.scene_npcs else '无'}\n"
                f"玩家: {user_input}"
            )
            result = Runner.run_sync(
                self.tool_advisor,
                input=advisor_input,
                max_turns=1,
            )
            return result.final_output.strip()
        except Exception as e:
            self._debug(f"[Advisor] 调用失败: {e}")
            return ""

    async def _run_tool_advisor_async(self, user_input: str) -> str:
        """Async version — for use inside process_streaming() where event loop is running."""
        if self.tool_advisor is None:
            return ""
        try:
            state = self.player_state.get_state()
            advisor_input = (
                f"场景: {self._time_of_day} {self._weather} | "
                f"NPC: {', '.join(self.scene_npcs) if self.scene_npcs else '无'}\n"
                f"玩家状态: HP {state['hp']}/{state['max_hp']}  情绪 {state['emotion']}  "
                f"信任 {state['trust']}  体力 {state['stamina']}\n"
                f"角色卡:\n{self.player.summary()}\n\n"
                f"玩家: {user_input}"
            )
            result = await Runner.run(
                self.tool_advisor,
                input=advisor_input,
                max_turns=1,
            )
            return result.final_output.strip()
        except Exception as e:
            self._debug(f"[Advisor] 调用失败: {e}")
            return ""

    def _build_input(self, user_input: str) -> str:
        """Build a compact input text for the GM Agent.

        Memory retrieval is NOT done here — the GM calls `search_memory` as a tool
        when it actually needs to recall past events.  This keeps the context window
        lean and avoids injecting the same old memories every turn.
        """
        parts: list[str] = []

        # -- Current state summary (lightweight, always relevant) --
        state = self.player_state.get_state()
        npc_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无"
        parts.append(
            f"## 当前状态\n"
            f"时间: {self._time_of_day}  天气: {self._weather}  |  场景NPC: {npc_text}\n"
            f"HP {state['hp']}/{state['max_hp']}  情绪 {state['emotion']}  "
            f"信任 {state['trust']}  体力 {state['stamina']}"
        )

        # -- Recent history (last 3 exchanges, for multi-turn continuity) --
        history_text = self._format_history()
        if history_text:
            parts.append(history_text)

        # -- Previous turn (suggestion continuity) --
        if self._last_gm_response:
            parts.append(f"## 上一轮 GM 回应\n{self._last_gm_response}")

        # -- Player input --
        parts.append(f"玩家: {user_input}")
        parts.append("请按系统提示词中定义的 JSON 格式回复（只包含 narration 和 suggestions）。")

        return "\n\n".join(parts)

    def _build_game_context(self) -> GameContext:
        """Build the GameContext dataclass with current game state."""
        return GameContext(
            player_state=self.player_state,
            npc_store=self.npc_store,
            memory=self.memory,
            knowledge=self.knowledge,
            scene_npcs=self.scene_npcs,
            time_of_day=self._time_of_day,
            weather=self._weather,
            player_name=self.player.name,
            player_card=self.player.summary(),
            player_skills=self.player.skills,
            player_attributes=self.player.attributes,
            npc_agents=self._npc_agents,
            judge_agent=self._judge_agent,
            narrator_agent=self._narrator_agent,
            history_messages=[],  # don't pass — SDK may leak stale tool_calls to DeepSeek
            llm=self.llm,
            game_over=self._game_over,
            game_over_pending=self._game_over_pending,
            game_over_cause=self._game_over_cause,
            debug=self.debug,
            debug_log=[],
            recent_events=list(self._recent_events),
        )

    def _format_history(self) -> str:
        """Format recent conversation history for context."""
        if not self.history_messages:
            return ""
        lines = ["## 最近对话"]
        for msg in self.history_messages[-6:]:  # last 3 exchanges
            role = "玩家" if msg["role"] == "user" else "GM"
            content = msg["content"][:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines) + "\n\n"

    def _trim_history(self, max_exchanges: int = 5) -> None:
        """Keep only the most recent N exchanges in history."""
        max_messages = max_exchanges * 2  # user + assistant per exchange
        if len(self.history_messages) > max_messages:
            self.history_messages = self.history_messages[-max_messages:]

    def _sync_npc_agents(self) -> None:
        """Ensure every NPC in the store has a corresponding Agent in the pool.

        Called at startup (after backstory NPC registration) and after loading a save,
        so that initial NPCs don't have to wait for a create_npc tool call to get
        their autonomous Agent.
        """
        if not self._llm_available:
            return

        from trpg_agent.npc_agent import create_npc_agent

        for npc_char in self.npc_store.all():
            name = npc_char.name
            if name not in self._npc_agents:
                try:
                    self._npc_agents[name] = create_npc_agent(
                        name=name,
                        personality_prompt=npc_char.build_personality_prompt(),
                        npc_store=self.npc_store,
                    )
                    self._debug(f"[DEBUG] 同步 NPC Agent: {name}")
                except Exception as e:
                    self._debug(f"[DEBUG] NPC Agent 创建失败 ({name}): {e}")

    # ===================================================================
    #  Multi-Agent process flow
    # ===================================================================

    def _process_multi_agent(self, user_input: str) -> str:
        """New architecture: Orchestrator → (Judge + NPCs parallel) → Narrator.

        The Orchestrator analyzes intent and routes to sub-agents via tools.
        It NEVER writes narrative — it collects results and calls Narrator.
        """
        user_input = user_input.strip()
        if not user_input:
            return "请说点什么吧。"

        self._last_user_input = user_input
        from trpg_agent.tools import clear_tool_cache
        clear_tool_cache()

        # ---- Game over confirmation gate ----
        if self._game_over_pending:
            if user_input.lower() in ("/confirm", "确认"):
                return self._confirm_game_over()
            else:
                return self._cancel_game_over()

        # ---- Build context ----
        scene_npcs_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无"
        self._debug(f"\n{'─'*50}\n[Orch] 玩家: {user_input}")
        self._debug(f"[Orch] 场景: {scene_npcs_text}  |  {self._time_of_day}  ·  {self._weather}")

        # Orchestrator input (lean — just state + history, no memory injection)
        state = self.player_state.get_state()
        scene_info = (
            f"时间: {self._time_of_day}  天气: {self._weather}  场景NPC: {scene_npcs_text}\n"
            f"HP {state['hp']}/{state['max_hp']}  情绪 {state['emotion']}  "
            f"信任 {state['trust']}  体力 {state['stamina']}"
        )
        orch_input = (
            f"## 当前状态\n{scene_info}\n\n"
            f"## 玩家 {self.player.name}\n{user_input}"
        )

        # GameContext with sub-agents
        game_ctx = self._build_game_context()

        # Update Orchestrator instructions
        self._orchestrator_agent.instructions = (
            build_gm_instructions(  # reuse existing prompt builder for world/player info
                world_setting=self.world.get("description", ""),
                player_name=self.player.name,
                player_card=self.player.summary(),
                time_of_day=self._time_of_day,
                weather=self._weather,
                scene_npcs=scene_npcs_text,
            )
        )

        # ---- Run Orchestrator ----
        t0 = time.time()
        try:
            result = Runner.run_sync(
                self._orchestrator_agent,
                input=orch_input,
                context=game_ctx,
                max_turns=8,
            )
            response = result.final_output
        except Exception as e:
            self._debug(f"[Orch] Runner 失败: {e}")
            return self._fallback_response(user_input)

        elapsed_ms = int((time.time() - t0) * 1000)

        # ---- Parse JSON output (from Narrator, relayed by Orchestrator) ----
        self._last_suggestions = []
        narration = response
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict):
                narration = parsed.get("narration", response)
                self._last_suggestions = parsed.get("suggestions", [])
        except (json.JSONDecodeError, TypeError):
            # Fallback: treat entire response as narration
            narration = response

        if self.debug:
            self._debug(f"[Orch] 完成  |  耗时 {elapsed_ms}ms  |  输出 {len(response)}字符  |  "
                        f"输入 ~{self._estimate_tokens(orch_input)}tok")
            for entry in game_ctx.debug_log:
                self._debug(entry)

        # ---- History ----
        self.history_messages.append({"role": "user", "content": user_input})
        self.history_messages.append({"role": "assistant", "content": response})
        self._trim_history()

        # ---- State sync ----
        self._time_of_day = game_ctx.time_of_day
        self._weather = game_ctx.weather
        self._game_over = game_ctx.game_over
        self._game_over_pending = game_ctx.game_over_pending
        self._game_over_cause = game_ctx.game_over_cause

        # ---- Memory ----
        if not self._game_over and narration:
            try:
                self._write_memory(user_input, narration, [])
                for npc_name in game_ctx.scene_npcs:
                    combined = (
                        f"以下事件与{npc_name}相关，请从{npc_name}的视角提取记忆:\n"
                        f"玩家: {user_input}\n事件: {narration[:300]}"
                    )
                    self._write_npc_memory(npc_name, combined)
            except Exception as e:
                self._debug(f"[DEBUG] 记忆写入失败: {e}")

        # ---- Tick: NPC autonomous time-passage (every 3 turns) ----
        self._turn_count += 1
        if self._turn_count % 3 == 0 and self._orchestrator_agent is not None:
            self._debug("[Orch] NPC tick...")
            tick_text = self._run_npc_tick()
            if tick_text and tick_text.strip():
                narration += "\n\n" + tick_text

        if self.debug:
            state = self.player_state.get_state()
            self._debug(
                f"[状态] HP {state['hp']}/{state['max_hp']}  |  "
                f"情绪 {state['emotion']}  |  信任 {state['trust']}  |  "
                f"体力 {state['stamina']}  |  在场 {self.scene_npcs or '无'}"
            )

        self._last_gm_response = narration
        return narration

    def _run_npc_tick(self) -> str:
        """Run autonomous tick for all scene NPCs. Returns narrative text or empty."""
        if not self._npc_agents or not self.scene_npcs:
            return ""

        import asyncio as _asyncio
        from trpg_agent.npc_agent import build_npc_tick_input

        async def _tick_one(name: str) -> str | None:
            agent = self._npc_agents.get(name)
            if agent is None:
                return None
            npc_state = self.npc_store.get_state(name)
            state_dict = npc_state.get_state() if npc_state else {
                "emotion": "calm", "stamina": "fresh", "hp": "?", "max_hp": "?"
            }
            scene = f"时间: {self._time_of_day}, 天气: {self._weather}"
            npc_input = build_npc_tick_input(
                npc_name=name,
                npc_state=state_dict,
                scene=scene,
                turns_since_last_action=3,
                memory=self.memory,
            )
            try:
                result = await Runner.run(agent, input=npc_input, max_turns=2)
                response = result.final_output.strip()
                if response and response.lower() != "<silent>":
                    self.npc_store.append_history(name, "user", "[时间流逝]")
                    self.npc_store.append_history(name, "assistant", response)
                    self.npc_store.save_state(name)
                    return f"{name}: {response}"
                return None
            except Exception:
                return None

        async def _run_all():
            tasks = [_tick_one(name) for name in self.scene_npcs]
            results = await _asyncio.gather(*tasks, return_exceptions=True)
            lines = []
            for r in results:
                if isinstance(r, str) and r:
                    lines.append(r)
            return "\n".join(lines)

        try:
            loop = _asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use run_coroutine_threadsafe or just return empty
                self._debug("[Orch] Tick 跳过——异步事件循环已在运行")
                return ""
            return _asyncio.run(_run_all())
        except RuntimeError:
            return _asyncio.run(_run_all())

    # ===================================================================
    #  Public API
    # ===================================================================

    def process(self, user_input: str, console=None) -> str:
        """Single turn entry point — SDK Runner-driven tool calling loop.

        Parameters
        ----------
        user_input : str
            Player's input text.
        console
            Rich Console object for streaming output. Disabled if None.

        Returns
        -------
        str
            GM's reply text.
        """
        # ---- Multi-agent mode: Orchestrator → Judge+NPC → Narrator ----
        if not self.legacy_mode and self._orchestrator_agent is not None:
            return self._process_multi_agent(user_input)

        user_input = user_input.strip()
        if not user_input:
            return "请说点什么吧。"

        self._last_user_input = user_input
        from trpg_agent.tools import clear_tool_cache
        clear_tool_cache()

        # ---- Game over confirmation gate ----
        if self._game_over_pending:
            if user_input.lower() in ("/confirm", "确认"):
                return self._confirm_game_over()
            else:
                return self._cancel_game_over()

        # ---- Command routing: world builder ---- (kept as-is)
        if self.world_builder and user_input.startswith("!"):
            world_result = self._handle_world_builder_command(user_input)
            if world_result:
                return self._narrate_world_builder_result(world_result, user_input)

        # ---- Build context ----
        scene_npcs_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无"
        self._debug(f"\n{'─'*50}\n[GM] 玩家: {user_input}")
        self._debug(f"[GM] 场景: {scene_npcs_text}  |  {self._time_of_day}  ·  {self._weather}")

        input_text = self._build_input(user_input)
        game_ctx = self._build_game_context()

        # -- Tool Advisor: lightweight suggestion before GM acts --
        t0_advice = time.time()
        advice = self._run_tool_advisor(user_input)
        if self.debug and advice:
            ms_advice = int((time.time() - t0_advice) * 1000)
            self._debug(f"[Advisor] {advice}  ({ms_advice}ms)")
        if advice and advice != "无需工具":
            input_text = f"【工具建议】考虑调用 {advice}\n\n" + input_text

        # Update GM instructions with current world state
        world_setting = self.world.get("description", self.world.get("name", "未知世界"))
        self.gm_agent.instructions = build_gm_instructions(
            world_setting=world_setting,
            player_name=self.player.name,
            player_card=self.player.summary(),
            time_of_day=self._time_of_day,
            weather=self._weather,
            scene_npcs=scene_npcs_text,
        )

        # ---- SDK Runner: automatic tool-calling loop ----
        t0 = time.time()
        try:
            result = Runner.run_sync(
                self.gm_agent,
                input=input_text,
                context=game_ctx,
                max_turns=10,
            )
            response = result.final_output
        except Exception as e:
            self._debug(f"[GM] Runner 失败: {e}")
            return self._fallback_response(user_input)

        elapsed_ms = int((time.time() - t0) * 1000)

        # ---- Parse JSON output: narration + suggestions ----
        self._last_suggestions = []
        narration = response  # fallback
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict):
                narration = parsed.get("narration", response)
                self._last_suggestions = parsed.get("suggestions", [])
        except (json.JSONDecodeError, TypeError):
            # Try markdown code block extraction
            stripped = response.strip()
            code_block_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```$', stripped, re.DOTALL)
            if code_block_match:
                try:
                    parsed = json.loads(code_block_match.group(1).strip())
                    if isinstance(parsed, dict):
                        narration = parsed.get("narration", code_block_match.group(1))
                        self._last_suggestions = parsed.get("suggestions", [])
                except (json.JSONDecodeError, TypeError):
                    pass
            # Try to find JSON object in the text
            if narration == response and not self._last_suggestions:
                json_match = re.search(r'\{[^{}]*"narration"[^{}]*\}', stripped, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        if isinstance(parsed, dict):
                            narration = parsed.get("narration", response)
                            self._last_suggestions = parsed.get("suggestions", [])
                    except (json.JSONDecodeError, TypeError):
                        pass
            # Last resort: strip numbered suggestion lines from narration
            if narration == response:
                lines = narration.strip().split('\n')
                clean_lines = []
                suggestion_lines = []
                for line in lines:
                    if re.match(r'^\d+\.\s+', line.strip()):
                        suggestion_lines.append(re.sub(r'^\d+\.\s+', '', line.strip()))
                    else:
                        clean_lines.append(line)
                if suggestion_lines:
                    narration = '\n'.join(clean_lines).strip()
                    self._last_suggestions = suggestion_lines

        # ---- Debug: aggregate tool-level entries ----
        if self.debug:
            self._debug(f"[GM] Runner 完成  |  耗时 {elapsed_ms}ms  |  输出 {len(response)}字符  |  "
                        f"输入 {len(input_text)}字符(~{self._estimate_tokens(input_text)}tok)")
            for entry in game_ctx.debug_log:
                self._debug(entry)

        # ---- Update history ----
        self.history_messages.append({"role": "user", "content": user_input})
        self.history_messages.append({"role": "assistant", "content": response})
        self._trim_history()

        # ---- Sync GameContext state back from potentially-mutated dataclass ----
        self._time_of_day = game_ctx.time_of_day
        self._weather = game_ctx.weather
        self._game_over = game_ctx.game_over
        self._game_over_pending = game_ctx.game_over_pending
        self._game_over_cause = game_ctx.game_over_cause

        # ---- Write memory (always persist; the Agent decides what matters via actions) ----
        if not self._game_over and response:
            try:
                self._write_memory(user_input, response, [])
                # Write NPC memories for NPCs that were invoked during this turn
                for npc_name in game_ctx.scene_npcs:
                    combined = (
                        f"以下事件与{npc_name}相关，请从{npc_name}的视角提取记忆:\n"
                        f"玩家: {user_input}\n事件: {response[:300]}"
                    )
                    self._write_npc_memory(npc_name, combined)
            except Exception as e:
                self._debug(f"[DEBUG] 记忆写入失败: {e}")

        # ---- World simulation (kept external trigger) ----
        world_action = self._maybe_trigger_world_simulation(user_input)
        if world_action:
            narration += "\n\n" + world_action

        if self.debug:
            state = self.player_state.get_state()
            self._debug(
                f"[状态] HP {state['hp']}/{state['max_hp']}  |  "
                f"情绪 {state['emotion']}  |  信任 {state['trust']}  |  "
                f"体力 {state['stamina']}  |  在场 {self.scene_npcs or '无'}"
            )

        self._last_gm_response = narration
        return narration

    async def process_streaming(self, user_input: str) -> AsyncGenerator["str | StatusEvent", None]:
        """Async generator — yields text chunks and StatusEvents in real time.

        Yields:
            str          — narrative text delta for the player
            StatusEvent  — thinking / tool_call status for the UI

        Caller should read `self._last_gm_response`, `self.player_state`,
        and `self.scene_npcs` after the generator is exhausted.
        """
        user_input = user_input.strip()
        if not user_input:
            yield "请说点什么吧。"
            return

        self._last_user_input = user_input

        # ---- Command routing ----
        if self.world_builder and user_input.startswith("!"):
            world_result = self._handle_world_builder_command(user_input)
            if world_result:
                response = self._narrate_world_builder_result(world_result, user_input)
                yield response
                return

        from trpg_agent.tools import clear_tool_cache
        clear_tool_cache()

        self._debug(f"\n{'─'*50}\n[GM] 玩家: {user_input}")

        # ---- Fallback ----
        if not self._llm_available or not self.llm or self.gm_agent is None:
            response = self._fallback_response(user_input)
            yield response
            return

        # ---- Build context ----
        scene_npcs_text = ", ".join(self.scene_npcs) if self.scene_npcs else "无"
        self._debug(f"[GM] 场景: {scene_npcs_text}  |  {self._time_of_day}  ·  {self._weather}")

        input_text = self._build_input(user_input)
        game_ctx = self._build_game_context()

        # -- Tool Advisor --
        t0_advice = time.time()
        advice = await self._run_tool_advisor_async(user_input)
        if self.debug and advice:
            ms_advice = int((time.time() - t0_advice) * 1000)
            self._debug(f"[Advisor] {advice}  ({ms_advice}ms)")
        if advice and advice != "无需工具":
            input_text = f"【工具建议】考虑调用 {advice}\n\n" + input_text

        # Update GM instructions
        world_setting = self.world.get("description", self.world.get("name", "未知世界"))
        self.gm_agent.instructions = build_gm_instructions(
            world_setting=world_setting,
            player_name=self.player.name,
            player_card=self.player.summary(),
            time_of_day=self._time_of_day,
            weather=self._weather,
            scene_npcs=scene_npcs_text,
        )

        # ---- SDK Runner: async with hooks for status events ----
        event_queue: asyncio.Queue = asyncio.Queue()
        hooks = AgentStatusHooks(event_queue)

        full_response = ""
        t0 = time.time()
        try:
            run_task = asyncio.create_task(
                Runner.run(
                    self.gm_agent,
                    input=input_text,
                    context=game_ctx,
                    max_turns=10,
                    hooks=hooks,
                )
            )

            # Yield status events from hooks while runner is working
            while not run_task.done():
                try:
                    event = await asyncio.wait_for(event_queue.get(), timeout=0.1)
                    if isinstance(event, StatusEvent):
                        yield event
                except asyncio.TimeoutError:
                    pass

            # Drain remaining events
            while not event_queue.empty():
                event = event_queue.get_nowait()
                if isinstance(event, StatusEvent):
                    yield event

            result = run_task.result()
            full_response = result.final_output
        except Exception as e:
            self._debug(f"[GM] Runner 失败: {e}")
            fallback = self._fallback_response(user_input)
            full_response = fallback
            yield fallback

        elapsed_ms = int((time.time() - t0) * 1000)

        # ---- Parse JSON output: narration + suggestions ----
        self._last_suggestions = []
        narration = full_response  # fallback: use raw text
        try:
            parsed = json.loads(full_response.strip())
            if isinstance(parsed, dict):
                narration = parsed.get("narration", full_response)
                self._last_suggestions = parsed.get("suggestions", [])
        except (json.JSONDecodeError, TypeError):
            # JSON parse failed — try to extract content from raw LLM output
            # Strip markdown code blocks if present (```json ... ```)
            stripped = full_response.strip()
            code_block_match = re.match(r'^```(?:json)?\s*\n?(.*?)\n?```$', stripped, re.DOTALL)
            if code_block_match:
                try:
                    parsed = json.loads(code_block_match.group(1).strip())
                    if isinstance(parsed, dict):
                        narration = parsed.get("narration", code_block_match.group(1))
                        self._last_suggestions = parsed.get("suggestions", [])
                except (json.JSONDecodeError, TypeError):
                    pass

            # If still not parsed, try to find JSON object in the text
            if narration == full_response and not self._last_suggestions:
                json_match = re.search(r'\{[^{}]*"narration"[^{}]*\}', stripped, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group(0))
                        if isinstance(parsed, dict):
                            narration = parsed.get("narration", full_response)
                            self._last_suggestions = parsed.get("suggestions", [])
                    except (json.JSONDecodeError, TypeError):
                        pass

            # Last resort: strip out numbered suggestion lines from raw text
            # so they don't appear in the chat bubble
            if narration == full_response:
                # Remove lines like "1. xxx" or "2. xxx" from the end of narration
                lines = narration.strip().split('\n')
                clean_lines = []
                suggestion_lines = []
                for line in lines:
                    if re.match(r'^\d+\.\s+', line.strip()):
                        suggestion_lines.append(re.sub(r'^\d+\.\s+', '', line.strip()))
                    else:
                        clean_lines.append(line)
                if suggestion_lines:
                    narration = '\n'.join(clean_lines).strip()
                    self._last_suggestions = suggestion_lines

        if self.debug:
            self._debug(f"[GM] Runner 完成  |  耗时 {elapsed_ms}ms  |  "
                        f"输出 {len(full_response)}字符  |  "
                        f"输入 {len(input_text)}字符(~{self._estimate_tokens(input_text)}tok)")
            for entry in game_ctx.debug_log:
                self._debug(entry)

        # ---- Stream narration to frontend ----
        if narration:
            for sentence in _split_sentences(narration):
                yield sentence
                await asyncio.sleep(0.03)

        # ---- Post-processing (same as process()) ----
        self.history_messages.append({"role": "user", "content": user_input})
        self.history_messages.append({"role": "assistant", "content": full_response})
        self._trim_history()

        self._time_of_day = game_ctx.time_of_day
        self._weather = game_ctx.weather
        self._game_over = game_ctx.game_over
        self._game_over_pending = game_ctx.game_over_pending
        self._game_over_cause = game_ctx.game_over_cause

        if not self._game_over and full_response:
            try:
                self._write_memory(user_input, full_response, [])
                for npc_name in game_ctx.scene_npcs:
                    combined = (
                        f"以下事件与{npc_name}相关，请从{npc_name}的视角提取记忆:\n"
                        f"玩家: {user_input}\n事件: {full_response[:300]}"
                    )
                    self._write_npc_memory(npc_name, combined)
            except Exception as e:
                self._debug(f"[GM] 记忆写入失败: {e}")

        world_action = self._maybe_trigger_world_simulation(user_input)
        if world_action:
            yield "\n\n" + world_action
            full_response += "\n\n" + world_action

        if self.debug:
            state = self.player_state.get_state()
            self._debug(
                f"[状态] HP {state['hp']}/{state['max_hp']}  |  "
                f"情绪 {state['emotion']}  |  信任 {state['trust']}  |  "
                f"体力 {state['stamina']}  |  在场 {self.scene_npcs or '无'}"
            )

        self._last_gm_response = full_response

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
                f"玩家状态: HP {ps['hp']}/{ps['max_hp']}\n"
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
                f"HP: {state['hp']}/{state['max_hp']}\n"
                f"情绪：{state['emotion']}\n"
                f"信任度：{state['trust']}\n"
                f"体力：{state['stamina']}"
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
            self._debug(f"[记忆] NPC「{npc_name}」: {extracted[:100]}")
        except Exception as e:
            self._debug(f"[记忆] NPC「{npc_name}」写入失败: {e}")

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

            link_count = len(similar)
            self._debug(f"[记忆] 主: {extracted[:100]}  (链接 {link_count}条)")

            return mem_id
        except Exception:
            return None

    # ------------------------------------------------------------------
    #  Opening scene generation
    # ------------------------------------------------------------------

    def generate_opening(
        self,
        custom_worldview: str = "",
        custom_npc_setup: str = "",
    ) -> str:
        """Generate the opening scene narration and seed 2-3 starter NPCs.

        Parameters
        ----------
        custom_worldview : str
            自定义世界观描述（自然语言），若为空则使用 config.yaml 中的默认设定。
        custom_npc_setup : str
            自定义初始 NPC 描述（自然语言），若为空则仅注册背景故事中的 NPC。

        The LLM returns JSON with ``narration`` and ``npcs`` so the world
        begins with interactive characters already present.
        """
        if not self._llm_available or not self.llm:
            world_name = self.world.get("name", "未知世界")
            return (
                f"欢迎来到{world_name}。你是{self.player.name}。\n" f"冒险即将开始..."
            )

        # 优先使用自定义世界观，否则用 config.yaml 默认
        if custom_worldview:
            world_desc = custom_worldview.strip()
            # 同步更新 GM 的世界状态
            self.world["description"] = world_desc
        else:
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

        # 自定义 NPC 描述（自然语言）
        npc_instruction = ""
        if custom_npc_setup:
            npc_instruction = (
                f"\n\n玩家自定义了以下初始角色，请解析并创建：\n{custom_npc_setup}\n"
                f"请从中提取有名字、有身份的角色，注册为 NPC 并加入场景。"
            )

        if scenario_text:
            system = (
                f"你是 TRPG 主持人。\n"
                f"世界设定：{world_desc}\n\n"
                f"玩家扮演 {self.player.name}，{player_desc}\n"
                f"途径：{getattr(self.player, 'pathway', '未知')}，"
                f"序列{getattr(self.player, 'sequence', 9)}\n"
                f"以下剧本已加载，请严格按照剧本节点0生成开场：\n{scenario_text}\n\n"
                f'以 JSON 格式回复：\n'
                f'{{"narration": "开场叙述（不超过200字）", '
                f'"npcs": [{{"name": "NPC称呼", "core": ["角色描述"], '
                f'"relations": {{"{self.player.name}": "关系"}}}}], '
                f'"suggestions": ["建议1", "建议2", "建议3"]}}\n\n'
                f"要求：\n"
                f"- 严格按照剧本节点0的核心事件生成开场\n"
                f"- 只注册当前物理在场的角色。任务目标、传闻人物只在 narration 提及即可\n"
                f"{existing_npc_text}"
                f"{npc_instruction}"
            )
        else:
            system = (
                f"你是 TRPG 主持人。\n"
                f"世界设定：{world_desc}\n\n"
                f"玩家扮演 {self.player.name}，{player_desc}\n"
                f"途径：{getattr(self.player, 'pathway', '未知')}，"
                f"序列{getattr(self.player, 'sequence', 9)}\n\n"
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
                f"{npc_instruction}"
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

                # Determine personality_tone from core description
                tone = "正常"
                if core:
                    first_core = str(core[0]) if core else ""
                    if any(kw in first_core for kw in ["温柔", "活泼", "开朗"]):
                        tone = "温和友好"
                    elif any(kw in first_core for kw in ["冷酷", "严肃", "沉默"]):
                        tone = "冷淡严肃"
                    elif any(kw in first_core for kw in ["狡猾", "精明", "圆滑"]):
                        tone = "精明圆滑"

                attributes = self._generate_npc_attributes(name, core, tone)

                self.npc_store.create(
                    name=name,
                    core=core,
                    attributes=attributes,
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
                if name not in self.scene_npcs:
                    self.scene_npcs.append(name)

            if self.debug and npcs:
                print(f"[DEBUG] 开局创建 NPC: {[n.get('name') for n in npcs]}")

            # Treat opening as first GM output — write memory + set context
            self._last_suggestions = suggestions
            self._last_gm_response = narration
            self._recent_events.append(narration)
            self._write_memory("游戏开始", narration, [])

            return narration
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
    #  Game over confirmation (called from process() / API, not from SDK tools)
    # ------------------------------------------------------------------

    def _confirm_game_over(self) -> str:
        """Execute actual game-over cleanup after player confirmation."""
        import glob as _glob

        if not self._game_over_pending:
            return "没有待确认的游戏结束。"

        # Delete save file
        if os.path.exists("data/save.json"):
            os.remove("data/save.json")

        # Clear NPC state files
        for f in _glob.glob("data/chroma/npcs/*_state.json"):
            try:
                os.remove(f)
            except Exception:
                pass

        # Clear run-time state
        self.scene_npcs.clear()
        self._time_of_day = "黄昏"
        self._weather = "阴"

        # Clear all memory collections
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

        self._game_over = True
        self._game_over_pending = False
        cause = self._game_over_cause
        self._game_over_cause = ""
        return f"游戏结束: {cause}。冒险终结。"

    def _cancel_game_over(self) -> str:
        """Cancel a pending game-over, resuming normal gameplay."""
        self._game_over_pending = False
        self._game_over_cause = ""
        return "已取消游戏结束。冒险继续。"

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

        self._debug(
            f"[DEBUG] 存档加载成功 ({path})\n"
            f"  场景: {self.scene_npcs}\n"
            f"  时间: {self._time_of_day} 天气: {self._weather}"
        )

        # Ensure NPC Agents exist for restored NPCs
        self._sync_npc_agents()

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


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming effect."""
    parts = re.split(r'([。！？；\n]+)', text)
    sentences = []
    for i in range(0, len(parts), 2):
        sentence = parts[i]
        if i + 1 < len(parts):
            sentence += parts[i + 1]
        if sentence.strip():
            sentences.append(sentence)
    if len(parts) % 2 == 1 and parts[-1].strip():
        sentences.append(parts[-1])
    if not sentences:
        sentences = [text]
    return sentences
