"""世界构建 Agent — 自然语言创建 NPC 和世界观。

WorldBuilder 允许玩家/GM 用自然语言描述角色或世界设定，
自动解析为结构化数据并写入系统。
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from trpg_agent.llm import LLM
    from trpg_agent.npc import NPCStore
    from trpg_agent.rag import KnowledgeBase


class WorldBuilder:
    """世界构建 Agent — 自然语言驱动 NPC/世界观创建。

    Parameters
    ----------
    llm : LLM
        LLM 实例，用于解析自然语言。
    npc_store : NPCStore
        NPC 存储实例。
    knowledge : KnowledgeBase
        知识库实例。
    """

    def __init__(
        self,
        llm: "LLM",
        npc_store: "NPCStore",
        knowledge: "KnowledgeBase",
        debug: bool = False,
    ) -> None:
        self.llm = llm
        self.npc_store = npc_store
        self.knowledge = knowledge
        self.debug = debug

    # ------------------------------------------------------------------
    #  P1: 自然语言创建 NPC
    # ------------------------------------------------------------------

    def create_npc_from_description(
        self,
        description: str,
        world_context: str = "",
        player_name: str = "",
    ) -> str:
        """用自然语言描述创建 NPC。

        Parameters
        ----------
        description : str
            自然语言描述，如："一个在码头工作的老渔民，性格豪爽，喜欢喝酒，
            对贝克兰德的黑帮很了解"
        world_context : str, optional
            当前世界观/场景上下文。
        player_name : str, optional
            玩家名称，用于建立关系。

        Returns
        -------
        str
            创建结果消息。
        """
        system = (
            "你是一个 TRPG 世界构建助手。将玩家的自然语言描述转换为结构化的 NPC 数据。\n"
            "请根据描述推断合理的属性、性格、语言特征等。\n"
            "以 JSON 格式返回，结构如下：\n"
            '{\n'
            '  "name": "NPC名称",\n'
            '  "core": ["角色背景描述1", "描述2"],\n'
            '  "attributes": {\n'
            '    "力量": 10, "敏捷": 10, "体质": 10,\n'
            '    "智力": 10, "感知": 10, "魅力": 10, "灵性": 10\n'
            '  },\n'
            '  "personality": {\n'
            '    "tone": "说话风格，如豪爽、阴险、温柔等",\n'
            '    "verbal_tics": "口头禅或语言习惯，没有则填无",\n'
            '    "emotion_map": {\n'
            '      "calm": "平静时的表现",\n'
            '      "wary": "警惕时的表现",\n'
            '      "hostile": "敌对时的表现"\n'
            '    }\n'
            '  },\n'
            '  "relations": {},\n'
            '  "catchphrases": []\n'
            '}\n'
            "注意：\n"
            "- 属性值范围 1-20，根据角色类型合理分配\n"
            "- core 至少包含一条背景描述\n"
            "- personality 要贴合描述，不要千篇一律\n"
            "- 如果描述中提到与玩家的关系，写入 relations 字段\n"
        )

        if world_context:
            system += f"\n当前世界观/场景：\n{world_context}\n"

        messages = [{"role": "user", "content": f"请用自然语言创建 NPC：{description}"}]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
        except Exception as e:
            return f"NPC 创建失败：{e}"

        name = (result.get("name") or "").strip()
        if not name:
            return "NPC 创建失败：未解析到名称"

        # Check for duplicate
        existing = self.npc_store.find_by_name(name)
        if existing is not None:
            return f"NPC「{name}」已存在。如需更新请删除后重新创建。"

        core = result.get("core", [name])
        if not isinstance(core, list):
            core = [str(core)]
        if not core:
            core = [name]

        attributes = result.get("attributes", {})
        default_attrs = {
            "力量": 10, "敏捷": 10, "体质": 10,
            "智力": 10, "感知": 10, "魅力": 10, "灵性": 10,
        }
        for k, v in default_attrs.items():
            if k not in attributes:
                attributes[k] = v

        personality = result.get("personality", {})
        if not isinstance(personality, dict):
            personality = {"tone": "正常", "verbal_tics": "无", "emotion_map": {}}
        personality.setdefault("tone", "正常")
        personality.setdefault("verbal_tics", "无特殊语言习惯")
        personality.setdefault("emotion_map", {
            "calm": f"以{personality.get('tone', '正常')}的态度说话",
            "wary": "警惕地观察",
            "hostile": "表现出敌意",
        })

        relations = result.get("relations", {})
        if not isinstance(relations, dict):
            relations = {}

        self.npc_store.create(
            name=name,
            core=core,
            attributes=attributes,
            personality=personality,
            relations=relations,
        )

        if self.debug:
            print(f"[DEBUG] 自然语言创建 NPC: {name} (语调: {personality.get('tone')})")

        core_summary = "；".join(core[:2])
        return f"已创建 NPC「{name}」: {core_summary}\n语调: {personality.get('tone')}"

    # ------------------------------------------------------------------
    #  P2: 自然语言构建世界观
    # ------------------------------------------------------------------

    def add_world_knowledge(
        self,
        description: str,
        known_by: str = "所有人",
        category: str = "",
    ) -> str:
        """用自然语言添加世界观知识。

        Parameters
        ----------
        description : str
            自然语言描述的世界设定。
        known_by : str, optional
            哪些角色可以知道这条知识。
        category : str, optional
            知识分类。

        Returns
        -------
        str
            添加结果消息。
        """
        # LLM 帮助结构化知识内容，提取关键信息
        system = (
            "你是一个 TRPG 世界构建助手。将玩家的自然语言描述提炼为精炼的世界观知识条目。\n"
            "保持描述简洁（不超过200字），保留关键信息（地点、人物、组织、历史事件等）。\n"
            "如果描述中包含多个独立知识点，请分别提炼为独立的条目。\n"
            "以 JSON 数组格式返回，每个条目一个字符串：\n"
            '["知识条目1", "知识条目2", ...]\n'
            "如果只有一个条目，也返回数组格式。"
        )
        messages = [{"role": "user", "content": f"请提炼以下世界观描述：\n{description}"}]

        try:
            result = self.llm.chat_json(system=system, messages=messages)
        except Exception as e:
            return f"世界观添加失败：{e}"

        # Parse result - could be a list or a dict with "knowledge" key
        entries = []
        if isinstance(result, list):
            entries = result
        elif isinstance(result, dict):
            # Try various keys
            for key in ["knowledge", "entries", "items", "facts"]:
                val = result.get(key)
                if isinstance(val, list):
                    entries = val
                    break
            if not entries and result.get("narration"):
                entries = [result["narration"]]

        if not entries:
            # Fallback: use original description
            entries = [description]

        count = 0
        for entry in entries:
            if isinstance(entry, str) and entry.strip():
                self.knowledge.add_knowledge(
                    content=entry.strip(),
                    known_by=known_by,
                    category=category,
                )
                count += 1

        return f"已添加 {count} 条世界观知识"

    # ------------------------------------------------------------------
    #  P3: 世界模拟 — NPC 自主行动
    # ------------------------------------------------------------------

    def simulate_npc_actions(
        self,
        npc_name: str,
        scene_context: str = "",
        time_of_day: str = "",
        weather: str = "",
        other_npc_actions: List[str] = None,
    ) -> Optional[str]:
        """模拟 NPC 的自主行动（让世界活起来）。

        Parameters
        ----------
        npc_name : str
            NPC 名称。
        scene_context : str, optional
            当前场景描述。
        time_of_day : str, optional
            当前时间。
        weather : str, optional
            当前天气。
        other_npc_actions : list of str, optional
            其他 NPC 已发生的行动，用于避免矛盾。

        Returns
        -------
        str or None
            NPC 的行动描述，如果无需行动则返回 None。
        """
        npc = self.npc_store.find_by_name(npc_name)
        if npc is None:
            return None

        npc_state = self.npc_store.get_state(npc_name)
        state_dict = (
            npc_state.get_state()
            if npc_state
            else {"emotion": "calm", "trust": 0.5, "stamina": "fresh", "hp": 10, "max_hp": 10}
        )

        npc_system = (
            f"你是 NPC「{npc_name}」。\n"
            f"{npc.build_personality_prompt()}\n"
            f"\n当前状态：\n"
            f"- 情绪：{state_dict.get('emotion', 'calm')}\n"
            f"- 体力：{state_dict.get('stamina', 'fresh')}\n"
            f"- HP：{state_dict.get('hp', '?')}/{state_dict.get('max_hp', '?')}\n"
        )

        if scene_context:
            npc_system += f"\n当前场景：{scene_context}\n"
        if time_of_day:
            npc_system += f"\n时间：{time_of_day}\n"
        if weather:
            npc_system += f"\n天气：{weather}\n"

        # Inject other NPC actions to avoid contradictions
        if other_npc_actions:
            npc_system += "\n此刻场景中发生的事：\n"
            for a in other_npc_actions:
                npc_system += f"- {a}\n"

        # Inject NPC history
        history = self.npc_store.get_history(npc_name)
        if history:
            npc_system += "\n最近对话记录：\n"
            for h in list(history)[-5:]:
                role = h.get("role", "")
                content = h.get("content", "")
                npc_system += f"- {role}: {content}\n"

        npc_system += (
            "\n\n请决定你现在要做什么。你可以：\n"
            "- 在场景中走动/探索\n"
            "- 自言自语或与看不见的第三方对话\n"
            "- 执行与你的角色身份相符的日常行为\n"
            "- 对其他 NPC 的行动做出反应\n"
            "- 对环境做出反应\n"
            "如果你觉得此刻不需要行动，返回空字符串。\n"
            "用 JSON 格式返回：\n"
            '{\n'
            '  "action": "行动描述（1-2句话，不超过80字）",\n'
            '  "persist_memory": true/false\n'
            '}'
        )

        try:
            result = self.llm.chat_json(system=npc_system, messages=[
                {"role": "user", "content": "请决定你现在要做什么。"}
            ])

            action = result.get("action", "")
            persist = result.get("persist_memory", True)

            if action and action.strip():
                action = action.strip()
                if persist:
                    self.npc_store.append_history(npc_name, "assistant", f"[自主行动] {action}")
                self.npc_store.save_state(npc_name)
                return f"{npc_name}：{action}"
            return None
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] NPC 模拟行动失败 ({npc_name}): {e}")
            return None
