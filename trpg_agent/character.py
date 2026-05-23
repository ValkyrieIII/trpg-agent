"""Character data model — YAML loading and validation.

The :class:`Character` dataclass represents a player character card loaded from
a YAML configuration file.  It holds player attributes and skills without any
NPC personality/roleplay data (those belong to :class:`NPCCharacter`).

Typical usage::

    from trpg_agent.character import Character

    char = Character.load("config.yaml")
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
    """Immutable (by convention) player character card.

    Parameters
    ----------
    name : str
        Character name.
    core : list of str
        Background / role description lines.
    attributes : dict of str → int
        Numeric attributes such as strength, agility, etc.
    skills : list of dict
        Each entry has ``name`` (str) and ``value`` (int) keys.
    """

    name: str
    core: List[str]
    attributes: Dict[str, int] = field(default_factory=dict)
    skills: List[Dict[str, Any]] = field(default_factory=list)

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
    if not isinstance(data, dict) or "player" not in data:
        print("错误：配置文件中缺少 'player' 字段")
        sys.exit(1)

    char_data = data["player"]

    # --- required field validation ---
    # Each required field is checked individually so the user gets a clear,
    # specific message instead of a generic error.
    _require_field(char_data, "name", "角色名称")
    _require_field(char_data, "core", "角色背景")
    _require_field(char_data, "attributes", "属性")

    return Character(
        name=char_data["name"],
        core=char_data["core"],
        attributes=char_data["attributes"],
        skills=char_data.get("skills", []),
    )


def _require_field(data: dict, key: str, cn_name: str) -> None:
    """Check that *key* exists in *data* and is truthy; exit otherwise."""
    if key not in data or not data[key]:
        print(f"错误：角色配置缺少必填字段「{cn_name}」({key})")
        sys.exit(1)
