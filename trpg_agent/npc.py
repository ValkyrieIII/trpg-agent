"""NPC character card + ChromaDB persistence.

The :class:`NPCCharacter` dataclass extends :class:`Character` with a
personality layer (tone, verbal tics, catchphrases, emotion map) and
few-shot dialogue examples inherited from the v1 Character design.

The :class:`NPCStore` provides ChromaDB-backed persistence and semantic
search for NPCs, plus per-NPC conversation history management.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
import yaml
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from trpg_agent.character import Character


# ---------------------------------------------------------------------------
#  NPCCharacter
# ---------------------------------------------------------------------------

@dataclass
class NPCCharacter(Character):
    """NPC character card — extends Character with personality layer.

    Parameters
    ----------
    name : str
        NPC name.
    core : list of str
        Background / role description lines.
    attributes : dict of str → int
        Numeric attributes such as strength, agility, etc.
    skills : list of dict
        Each entry has ``name`` (str) and ``value`` (int) keys.
    personality : dict
        Must contain keys ``tone``, ``verbal_tics``, ``emotion_map``,
        ``catchphrases`` (optional list of str).
    few_shot : list of dict
        Dialogue examples, each with ``input`` and ``output`` keys.
    """

    personality: Dict[str, Any] = field(default_factory=dict)
    few_shot: List[Dict[str, str]] = field(default_factory=list)

    # ------------------------------------------------------------------
    #  Prompt builders
    # ------------------------------------------------------------------

    def build_personality_prompt(self) -> str:
        """Assemble the NPC personality system prompt.

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
        """Build a state block describing the NPC's current condition.

        Parameters
        ----------
        state : dict
            Expected keys: ``emotion`` (str), ``trust`` (float),
            ``stamina`` (str).

        Returns
        -------
        str
            Formatted state prompt with behaviour description resolved
            from the NPC's ``emotion_map``.
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

    # ------------------------------------------------------------------
    #  Class-method load
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str) -> NPCCharacter:
        """Load NPC from a standalone YAML file.

        The YAML must have a top-level ``npc`` key containing the
        NPC definition.

        Parameters
        ----------
        path : str
            Path to the YAML configuration file.

        Returns
        -------
        NPCCharacter
            A fully populated NPC character instance.

        Raises
        ------
        SystemExit
            If the file cannot be read, the YAML is malformed, or required
            fields are missing.
        """
        return load_npc(path)


# ---------------------------------------------------------------------------
#  Module-level loader
# ---------------------------------------------------------------------------

def load_npc(config_path: str) -> NPCCharacter:
    """Load and validate an :class:`NPCCharacter` from a YAML file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    NPCCharacter
        A fully populated NPC character instance.

    Raises
    ------
    SystemExit
        If the file cannot be read, the YAML is malformed, or required
        fields are missing.
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
    if not isinstance(data, dict) or "npc" not in data:
        print("错误：配置文件中缺少 'npc' 字段")
        sys.exit(1)

    npc_data = data["npc"]

    # --- required field validation ---
    _require_npc_field(npc_data, "name", "角色名称")
    _require_npc_field(npc_data, "core", "角色背景")
    _require_npc_field(npc_data, "personality", "人格设置")
    _require_npc_field(npc_data, "attributes", "属性")

    return NPCCharacter(
        name=npc_data["name"],
        core=npc_data["core"],
        personality=npc_data["personality"],
        attributes=npc_data["attributes"],
        skills=npc_data.get("skills", []),
        few_shot=npc_data.get("few_shot", []),
    )


def _require_npc_field(data: dict, key: str, cn_name: str) -> None:
    """Check that *key* exists in *data* and is truthy; exit otherwise."""
    if key not in data or not data[key]:
        print(f"错误：NPC 配置缺少必填字段「{cn_name}」({key})")
        sys.exit(1)


# ---------------------------------------------------------------------------
#  NPCStore — ChromaDB persistence
# ---------------------------------------------------------------------------

class NPCStore:
    """ChromaDB persistence + per-NPC conversation history.

    NPCs are indexed in ChromaDB for semantic search and loaded into an
    in-memory cache on startup.  Per-NPC conversation histories are kept
    in memory with a configurable sliding window.

    Parameters
    ----------
    persist_dir : str
        Directory path for ChromaDB persistence.
        Defaults to ``"data/chroma/npcs"``.
    """

    def __init__(self, persist_dir: str = "data/chroma/npcs") -> None:
        self._persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # -- ChromaDB --
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
        )
        self._collection = self._client.get_or_create_collection(
            name="npcs",
            embedding_function=self._embedding_fn,
        )

        # -- In-memory caches --
        self._npcs: Dict[str, NPCCharacter] = {}  # name -> NPCCharacter
        self._histories: Dict[str, List[Dict[str, str]]] = {}  # name -> history
        self._max_history: int = 10

        # -- Restore from ChromaDB --
        self._load_all()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def save(self, npc: NPCCharacter) -> None:
        """Persist NPC to ChromaDB (upsert by name).

        The NPC's personality, attributes, skills, core, and few_shot are
        serialised as JSON metadata.  The name and core lines form the
        document text for semantic search.

        Parameters
        ----------
        npc : NPCCharacter
            The NPC character to persist.
        """
        # Remove existing entry if present (ChromaDB does not support
        # true upsert across different metadata shapes).
        try:
            existing = self._collection.get(ids=[npc.name])
            if existing and existing.get("ids"):
                self._collection.delete(ids=[npc.name])
        except Exception:
            pass

        # Serialise fields as JSON metadata
        metadata = {
            "personality": json.dumps(npc.personality, ensure_ascii=False),
            "core": json.dumps(npc.core, ensure_ascii=False),
            "attributes": json.dumps(npc.attributes, ensure_ascii=False),
            "skills": json.dumps(npc.skills, ensure_ascii=False),
            "few_shot": json.dumps(npc.few_shot, ensure_ascii=False),
        }

        # NPC name + core as document text for semantic search
        doc = f"{npc.name}: {' '.join(npc.core)}"

        self._collection.add(
            ids=[npc.name],
            documents=[doc],
            metadatas=[metadata],
        )

        # Update in-memory cache
        self._npcs[npc.name] = npc

    def find_by_name(self, name: str) -> Optional[NPCCharacter]:
        """Look up NPC by exact name (from in-memory cache).

        Parameters
        ----------
        name : str
            The NPC name to look up.

        Returns
        -------
        NPCCharacter or None
            The NPC if found, or ``None``.
        """
        return self._npcs.get(name)

    def search(self, query: str, n: int = 5) -> List[NPCCharacter]:
        """Semantic search for NPCs via ChromaDB.

        Parameters
        ----------
        query : str
            The search query text.
        n : int, optional
            Maximum number of results to return (default ``5``).

        Returns
        -------
        list of NPCCharacter
            Matching NPCs.
        """
        try:
            raw = self._collection.query(query_texts=[query], n_results=n)
        except Exception:
            return []

        ids_list = raw.get("ids", [[]])
        if not ids_list or not ids_list[0]:
            return []

        results: List[NPCCharacter] = []
        for i in range(len(ids_list[0])):
            name = ids_list[0][i]
            # Return from cache if available
            if name in self._npcs:
                results.append(self._npcs[name])
            else:
                # Reconstruct from metadata
                meta = raw["metadatas"][0][i] if raw.get("metadatas") else {}
                npc = NPCCharacter(
                    name=name,
                    core=json.loads(meta.get("core", "[]")),
                    attributes=json.loads(meta.get("attributes", "{}")),
                    skills=json.loads(meta.get("skills", "[]")),
                    personality=json.loads(meta.get("personality", "{}")),
                    few_shot=json.loads(meta.get("few_shot", "[]")),
                )
                self._npcs[name] = npc
                results.append(npc)

        return results

    def all(self) -> List[NPCCharacter]:
        """Return all known NPCs.

        Returns
        -------
        list of NPCCharacter
            All NPCs currently in the in-memory cache (which is populated
            from ChromaDB on startup and kept in sync via ``save``).
        """
        return list(self._npcs.values())

    def create(
        self,
        name: str,
        core: List[str],
        attributes: Dict[str, int],
        skills: Optional[List[Dict[str, Any]]] = None,
        personality: Optional[Dict[str, Any]] = None,
        few_shot: Optional[List[Dict[str, str]]] = None,
    ) -> NPCCharacter:
        """Dynamically create and persist a new NPC.

        The created NPC is saved to ChromaDB before being returned.

        Parameters
        ----------
        name : str
            NPC name.
        core : list of str
            Background / role description lines.
        attributes : dict of str → int
            Numeric attributes.
        skills : list of dict, optional
            Skill definitions (default empty list).
        personality : dict, optional
            Personality layer (default empty dict).
        few_shot : list of dict, optional
            Few-shot dialogue examples (default empty list).

        Returns
        -------
        NPCCharacter
            The newly created and persisted NPC.
        """
        npc = NPCCharacter(
            name=name,
            core=core,
            attributes=attributes,
            skills=skills or [],
            personality=personality or {},
            few_shot=few_shot or [],
        )
        self.save(npc)
        return npc

    # ------------------------------------------------------------------
    #  Per-NPC conversation history
    # ------------------------------------------------------------------

    def get_history(self, name: str) -> List[Dict[str, str]]:
        """Get per-NPC conversation history (sliding window).

        Parameters
        ----------
        name : str
            The NPC name.

        Returns
        -------
        list of dict
            Each entry has ``role`` and ``content`` keys (or empty list
            if no history exists).
        """
        return self._histories.get(name, [])

    def append_history(self, name: str, role: str, content: str) -> None:
        """Append a turn to NPC conversation history with sliding window.

        Parameters
        ----------
        name : str
            The NPC name.
        role : str
            ``"user"`` or ``"assistant"``.
        content : str
            The message content.
        """
        if name not in self._histories:
            self._histories[name] = []

        self._histories[name].append({"role": role, "content": content})

        # Sliding window — keep only the most recent turns
        max_turns = self._max_history * 2
        if len(self._histories[name]) > max_turns:
            self._histories[name] = self._histories[name][-max_turns:]

    def clear_history(self, name: str | None = None) -> int:
        """Clear NPC conversation history.

        Parameters
        ----------
        name : str or None
            NPC name to clear, or ``None`` to clear all.

        Returns
        -------
        int
            Number of NPCs whose history was cleared.
        """
        if name is None:
            count = len(self._histories)
            self._histories.clear()
            return count
        if name in self._histories:
            del self._histories[name]
            return 1
        return 0

    # ------------------------------------------------------------------
    #  Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Load all NPCs from ChromaDB into memory cache on startup."""
        try:
            all_data = self._collection.get()
        except Exception:
            return

        ids_list = all_data.get("ids", [])
        if not ids_list:
            return

        metadatas_list = all_data.get("metadatas", [])

        for i, name in enumerate(ids_list):
            meta = metadatas_list[i] if metadatas_list and i < len(metadatas_list) else {}

            npc = NPCCharacter(
                name=name,
                core=json.loads(meta.get("core", "[]")),
                attributes=json.loads(meta.get("attributes", "{}")),
                skills=json.loads(meta.get("skills", "[]")),
                personality=json.loads(meta.get("personality", "{}")),
                few_shot=json.loads(meta.get("few_shot", "[]")),
            )
            self._npcs[name] = npc
