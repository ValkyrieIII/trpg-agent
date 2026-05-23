"""RAG 知识库 — ChromaDB 语义检索 + 角色身份权限过滤。

知识源为 Markdown 文件，通过 YAML front matter 标注可见角色（known_by）。
KnowledgeBase 被 GameMaster 核心用于世界观知识检索。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---------------------------------------------------------------------------
#  Front-matter parser
# ---------------------------------------------------------------------------

def _parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse YAML front matter from a Markdown file.

    Returns
    -------
    tuple of (dict, str)
        ``(metadata_dict, content_string)``.
        If no front matter is found, metadata is an empty dict and content
        is the original text unchanged.
    """
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, text

    # Find closing delimiter
    end_idx: Optional[int] = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx is None:
        return {}, text

    metadata: Dict[str, Any] = {}
    for line in lines[1:end_idx]:
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if value.startswith("[") and value.endswith("]"):
            # List syntax: [item1, item2]
            items = [
                x.strip().strip('"').strip("'")
                for x in value[1:-1].split(",")
                if x.strip()
            ]
            metadata[key] = items
        else:
            # Scalar value
            metadata[key] = value.strip('"').strip("'")

    content = "\n".join(lines[end_idx + 1 :]).strip()
    return metadata, content


# ---------------------------------------------------------------------------
#  KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """RAG 知识库 — ChromaDB 语义检索 + 角色身份权限过滤。

    Parameters
    ----------
    persist_dir : str
        ChromaDB 持久化目录（默认 ``"data/chroma/knowledge"``，
        与记忆系统的 ``data/chroma/memories/`` 分离）。
    """

    def __init__(self, persist_dir: str = "data/chroma/knowledge") -> None:
        self._persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-small-zh-v1.5",
        )
        self._collection = self._client.get_or_create_collection(
            name="knowledge",
            embedding_function=self._embedding_fn,
        )

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def add_knowledge(
        self,
        content: str,
        known_by: Union[str, List[str]],
        category: str = "",
    ) -> None:
        """Add a knowledge entry to the vector store.

        Parameters
        ----------
        content : str
            The knowledge text.
        known_by : str or list of str
            Role(s) allowed to see this knowledge.
            ``"所有人"`` means public (any character can retrieve it).
        category : str, optional
            Optional category label (e.g. ``"location"``, ``"npc"``).
        """
        if isinstance(known_by, str):
            known_by_list = [known_by]
        else:
            known_by_list = list(known_by)

        known_by_str = ",".join(known_by_list)

        self._collection.add(
            ids=[uuid4().hex[:8]],
            documents=[content],
            metadatas=[
                {
                    "known_by": known_by_str,
                    "category": category,
                }
            ],
        )

    def query(
        self,
        query: str,
        character: str,
        n: int = 3,
        threshold: float = 0.5,
    ) -> List[str]:
        """Semantic search with character permission filtering.

        Retrieves knowledge semantically similar to *query*, then filters
        out entries the given *character* is not permitted to see.

        ChromaDB returns cosine distance — smaller values indicate higher
        similarity.  Entries with ``distance > threshold`` are discarded.

        Parameters
        ----------
        query : str
            The search query text.
        character : str
            The character requesting knowledge.
        n : int, optional
            Maximum number of results to return (default ``3``).
        threshold : float, optional
            Maximum cosine distance to accept (default ``0.5``).

        Returns
        -------
        list of str
            Content strings of matching knowledge entries (up to *n*).
        """
        # Fetch extra candidates to compensate for permission filtering
        fetch_n = max(n * 5, 20)

        try:
            raw = self._collection.query(
                query_texts=[query], n_results=fetch_n
            )
        except Exception:
            return []

        ids_list = raw.get("ids", [[]])
        if not ids_list or not ids_list[0]:
            return []

        distances = raw.get("distances", [[]])
        metadatas = raw.get("metadatas", [[]])
        documents = raw.get("documents", [[]])

        results: List[str] = []
        for i in range(len(ids_list[0])):
            # Distance filter
            dist = distances[0][i] if distances and distances[0] else 1.0
            if dist > threshold:
                continue

            meta = metadatas[0][i] if metadatas and metadatas[0] else {}
            known_by_str = meta.get("known_by", "")

            # Permission check
            allowed_roles = [
                r.strip() for r in known_by_str.split(",") if r.strip()
            ]
            if character in allowed_roles or "所有人" in allowed_roles:
                doc = documents[0][i] if documents and documents[0] else ""
                if doc:
                    results.append(doc)
                    if len(results) >= n:
                        break

        return results

    def load_from_dir(self, dir_path: str) -> None:
        """Load all ``.md`` files from a directory.

        Each file may contain a YAML front matter block (``---`` delimited)
        specifying ``known_by`` and ``category``.  The body text after the
        front matter is split into paragraphs (separated by blank lines);
        each paragraph becomes a separate knowledge entry sharing the same
        metadata.

        Entries with identical content are not added twice (deduplication).

        Parameters
        ----------
        dir_path : str
            Path to the directory containing ``.md`` knowledge files.
        """
        if not os.path.isdir(dir_path):
            return

        seen_contents: set = set()

        for filename in sorted(os.listdir(dir_path)):
            if not filename.endswith(".md"):
                continue

            filepath = os.path.join(dir_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()

            if not text.strip():
                continue

            metadata, content = _parse_front_matter(text)
            if not content:
                continue

            known_by = metadata.get("known_by", "所有人")
            category = metadata.get("category", "")

            # Split body into paragraphs (separated by one or more blank lines)
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()
            ]

            if not paragraphs:
                if content not in seen_contents:
                    seen_contents.add(content)
                    self.add_knowledge(content, known_by=known_by, category=category)
            else:
                for para in paragraphs:
                    if para not in seen_contents:
                        seen_contents.add(para)
                        self.add_knowledge(para, known_by=known_by, category=category)
