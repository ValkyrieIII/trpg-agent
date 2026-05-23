"""Character data model — YAML loading, validation, and prompt construction.

The :class:`Character` dataclass represents a TRPG character card loaded from
a YAML configuration file.  It provides methods for constructing the system
prompt fragments injected by the GM core into each LLM call.

Typical usage::

    from trpg_agent.character import Character

    char = Character.load("config.yaml")
    personality_prompt = char.build_personality_prompt()
    state_prompt = char.build_state_prompt({"emotion": "calm", ...})
    summary = char.summary()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


# ---------------------------------------------------------------------------
#  Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Character:
    """Immutable (by convention) character card.

    Parameters
    ----------
    name : str
        Character name.
    core : list of str
        Background / role description lines (fully injected each turn).
    personality : dict
        Must contain keys ``tone``, ``verbal_tics``, ``emotion_map``,
        ``catchphrases``.
    few_shot : list of dict
        Dialogue examples, each with ``input`` and ``output`` keys.
    attributes : dict of str → int
        Numeric attributes such as strength, agility, etc.
    skills : list of dict
        Each entry has ``name`` (str) and ``value`` (int) keys.
    """

    name: str
    core: List[str]
    personality: Dict[str, Any]
    few_shot: List[Dict[str, str]] = field(default_factory=list)
    attributes: Dict[str, int] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    #  Prompt builders
    # ------------------------------------------------------------------

    def build_personality_prompt(self) -> str:
        """Assemble the full personality system prompt.

        Includes: core background, tone, verbal tics, catchphrases,
        and few-shot dialogue examples.
        """
        parts: List[str] = []

        # --- core background ---
        parts.append("【角色背景】")
        parts.extend(self.core)

        # --- tone & verbal tics ---
        parts.append("")
        parts.append("【说话方式】")
        parts.append(f"语调：{self.personality.get('tone', '正常')}")
        parts.append(f"语言习惯：{self.personality.get('verbal_tics', '无')}")

        # --- catchphrases ---
        catchphrases = self.personality.get("catchphrases", [])
        if catchphrases:
            parts.append("")
            parts.append("【口头禅】")
            for cp in catchphrases:
                parts.append(f"- {cp}")

        # --- few-shot examples ---
        if self.few_shot:
            parts.append("")
            parts.append("【对话示例】")
            for example in self.few_shot:
                parts.append(f"玩家：{example['input']}")
                parts.append(f"你：{example['output']}")
                parts.append("")

        return "\n".join(parts)

    def build_state_prompt(self, state: dict) -> str:
        """Build a state block describing the character's current condition.

        Parameters
        ----------
        state : dict
            Expected keys: ``emotion`` (str), ``trust`` (float),
            ``stamina`` (str).

        Returns
        -------
        str
            Formatted state prompt with behaviour description resolved
            from the character's ``emotion_map``.
        """
        emotion = state.get("emotion", "calm")
        trust = state.get("trust", 0.5)
        stamina = state.get("stamina", "fresh")

        emotion_map = self.personality.get("emotion_map", {})
        behaviour = emotion_map.get(emotion, "正常反应")

        lines = [
            "【当前状态】",
            f"情绪：{emotion}",
            f"信任度：{trust}",
            f"体力：{stamina}",
            f"行为表现：{behaviour}",
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Return a concise summary of character attributes and skills."""
        lines: List[str] = [f"角色：{self.name}"]

        if self.attributes:
            lines.append("")
            lines.append("【属性】")
            for k, v in self.attributes.items():
                lines.append(f"  {k}: {v}")

        if self.skills:
            lines.append("")
            lines.append("【技能】")
            for skill in self.skills:
                lines.append(f"  {skill['name']}: {skill['value']}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Class-method alias
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, config_path: str) -> Character:
        """Load a character from *config_path*.

        Equivalent to :func:`load_character`.
        """
        return load_character(config_path)


# ---------------------------------------------------------------------------
#  Module-level loader
# ---------------------------------------------------------------------------

def load_character(config_path: str) -> Character:
    """Load and validate a :class:`Character` from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Character
        A fully populated character instance.

    Raises
    ------
    SystemExit
        If the file cannot be read, the YAML is malformed, or required
        fields are missing (the message is a friendly description rather
        than a raw traceback).
    """
    # --- file I/O ---
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：找不到配置文件 {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"错误：配置文件格式有误 — {e}")
        sys.exit(1)

    # --- top-level key ---
    if not isinstance(data, dict) or "character" not in data:
        print("错误：配置文件中缺少 'character' 字段")
        sys.exit(1)

    char_data = data["character"]

    # --- required field validation ---
    # Each required field is checked individually so the user gets a clear,
    # specific message instead of a generic error.
    _require_field(char_data, "name", "角色名称")
    _require_field(char_data, "core", "角色背景")
    _require_field(char_data, "personality", "人格设置")
    _require_field(char_data, "attributes", "属性")

    return Character(
        name=char_data["name"],
        core=char_data["core"],
        personality=char_data["personality"],
        few_shot=char_data.get("few_shot", []),
        attributes=char_data["attributes"],
        skills=char_data.get("skills", []),
    )


def _require_field(data: dict, key: str, cn_name: str) -> None:
    """Check that *key* exists in *data* and is truthy; exit otherwise."""
    if key not in data or not data[key]:
        print(f"错误：角色配置缺少必填字段「{cn_name}」({key})")
        sys.exit(1)
