"""Hybrid memory store — ChromaDB for semantic search, NetworkX for relationship graphs.

The :class:`MemoryStore` combines vector similarity search with a directed
graph to represent relationships between memories, providing both semantic
retrieval and relational traversal for the GM core.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

import chromadb
import networkx as nx
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

# Memory types
TYPE_EVENT: str = "event"
TYPE_FACT: str = "fact"
TYPE_EMOTION_PEAK: str = "emotion_peak"

# Relationship type constants
REL_CAUSES: str = "导致了"
REL_RELATES_TO: str = "关联到"
REL_CONTRADICTS: str = "反驳了"
REL_AFTER: str = "发生在...之后"


# ---------------------------------------------------------------------------
#  MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """Hybrid memory store: ChromaDB (semantic search) + NetworkX (relational graph).

    Memories are indexed in ChromaDB for vector similarity search, while
    their relationships are maintained in a NetworkX directed graph for
    multi-hop traversal.

    Parameters
    ----------
    persist_dir : str
        Directory path for ChromaDB persistence and the graph JSON file.
        Defaults to ``"data/chroma/memories"``.
    """

    def __init__(self, persist_dir: str = "data/chroma/memories") -> None:
        self._persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)

        # -- ChromaDB --
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="memories",
            embedding_function=self._embedding_fn,
        )

        # -- NetworkX directed graph --
        self._graph: nx.DiGraph = nx.DiGraph()
        self._graph_path = os.path.join(persist_dir, "memory_graph.json")
        self._load_graph()

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        context: dict,
        importance: float = 0.5,
        mem_type: str = TYPE_EVENT,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Add a new memory to both ChromaDB and the graph.

        The memory is written to the ChromaDB collection for semantic
        search and simultaneously recorded as a node in the NetworkX
        graph.  The graph is automatically persisted to JSON after the
        operation.

        Parameters
        ----------
        content : str
            The memory content text.
        context : dict
            Contextual information (expected keys: ``location``, ``npcs``,
            ``emotion``).
        importance : float, optional
            Importance score between 0.0 and 1.0 (default ``0.5``).
        mem_type : str, optional
            Memory type — ``"event"``, ``"fact"``, or ``"emotion_peak"``
            (default ``"event"``).
        tags : list of str, optional
            Optional tags for categorisation.

        Returns
        -------
        str
            The generated 8-character short memory ID.
        """
        if tags is None:
            tags = []

        mem_id = uuid4().hex[:8]
        timestamp = datetime.now(timezone.utc).isoformat()

        # -- ChromaDB: serialise context dict as JSON string in metadata --
        chroma_meta = {
            "type": mem_type,
            "timestamp": timestamp,
            "importance": importance,
            "tags": json.dumps(tags, ensure_ascii=False),
            "context": json.dumps(context, ensure_ascii=False),
        }

        self._collection.add(
            ids=[mem_id],
            documents=[content],
            metadatas=[chroma_meta],
        )

        # -- NetworkX graph node --
        self._graph.add_node(
            mem_id,
            content=content,
            type=mem_type,
            timestamp=timestamp,
            importance=importance,
            tags=tags,
            context=context,
        )

        self._save_graph()
        return mem_id

    def search(self, query: str, n: int = 3) -> List[Dict[str, Any]]:
        """Semantically search memories via ChromaDB.

        Parameters
        ----------
        query : str
            The search query text.
        n : int, optional
            Maximum number of results to return (default ``3``).

        Returns
        -------
        list of dict
            Each result dict contains keys ``id``, ``content``,
            ``importance``, ``type``, ``timestamp``, and ``context``.
        """
        raw = self._collection.query(query_texts=[query], n_results=n)

        output: List[Dict[str, Any]] = []

        # Guard against query on an empty collection
        ids_list = raw.get("ids", [[]])
        if not ids_list or not ids_list[0]:
            return output

        for i in range(len(ids_list[0])):
            meta = raw["metadatas"][0][i]
            output.append(
                {
                    "id": ids_list[0][i],
                    "content": raw["documents"][0][i],
                    "importance": meta.get("importance", 0.5),
                    "type": meta.get("type", TYPE_EVENT),
                    "timestamp": meta.get("timestamp", ""),
                    "context": json.loads(meta.get("context", "{}")),
                }
            )
        return output

    def link(self, from_id: str, to_id: str, relation: str) -> None:
        """Add a directed edge between two memories in the graph.

        If either endpoint does not exist in the graph the call is
        silently ignored.

        Parameters
        ----------
        from_id : str
            Source memory ID.
        to_id : str
            Target memory ID.
        relation : str
            Relationship label (e.g. ``"导致了"``, ``"关联到"``).
        """
        if not self._graph.has_node(from_id) or not self._graph.has_node(to_id):
            return
        self._graph.add_edge(from_id, to_id, relation=relation)
        self._save_graph()

    def get_related(
        self, mem_id: str, hops: int = 2
    ) -> List[Dict[str, Any]]:
        """Traverse the graph to find memories related to *mem_id*.

        The traversal is bidirectional (treats the directed graph as
        undirected for navigational purposes).  Returned memories have
        their ``importance`` boosted by **+0.1** (capped at 1.0).

        Parameters
        ----------
        mem_id : str
            The starting memory ID.
        hops : int, optional
            Number of traversal hops (default ``2``).  A value of ``0``
            is treated as ``1`` so that direct neighbours are always
            included.

        Returns
        -------
        list of dict
            Related memories, each with the same fields as
            :meth:`search` results.  The starting node is excluded.
        """
        if not self._graph.has_node(mem_id):
            return []

        # hops=0 should still return direct neighbours
        radius = hops if hops > 0 else 1
        ego = nx.ego_graph(self._graph, mem_id, radius=radius, undirected=True)

        results: List[Dict[str, Any]] = []
        for node in ego.nodes():
            if node == mem_id:
                continue
            node_data = self._graph.nodes[node]
            importance = min(1.0, node_data.get("importance", 0.5) + 0.1)
            results.append(
                {
                    "id": node,
                    "content": node_data.get("content", ""),
                    "importance": importance,
                    "type": node_data.get("type", TYPE_EVENT),
                    "timestamp": node_data.get("timestamp", ""),
                    "context": node_data.get("context", {}),
                }
            )
        return results

    def full_retrieve(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Combined semantic + graph retrieval.

        1. Performs a semantic search via :meth:`search`.
        2. Expands each result via :meth:`get_related` (default 2 hops).
        3. Deduplicates by memory ID.
        4. Sorts by ``importance`` descending.

        Parameters
        ----------
        query : str
            The search query text.
        n : int, optional
            Number of top semantic results to retrieve (default ``5``).

        Returns
        -------
        list of dict
            Merged and sorted results.
        """
        semantic = self.search(query, n=n)

        seen: set = set()
        merged: List[Dict[str, Any]] = []

        for entry in semantic:
            if entry["id"] not in seen:
                seen.add(entry["id"])
                merged.append(entry)

            related = self.get_related(entry["id"])
            for rel in related:
                if rel["id"] not in seen:
                    seen.add(rel["id"])
                    merged.append(rel)

        merged.sort(key=lambda x: x["importance"], reverse=True)
        return merged

    # ------------------------------------------------------------------
    #  Persistence — JSON graph
    # ------------------------------------------------------------------

    def _save_graph(self) -> None:
        """Serialise the NetworkX graph to ``memory_graph.json``."""
        nodes = []
        for n, attr in self._graph.nodes(data=True):
            entry = {"id": n}
            entry.update(attr)
            nodes.append(entry)

        edges = []
        for u, v, attr in self._graph.edges(data=True):
            entry = {"source": u, "target": v}
            entry.update(attr)
            edges.append(entry)

        data: Dict[str, Any] = {"nodes": nodes, "edges": edges}
        with open(self._graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_graph(self) -> None:
        """Deserialise the NetworkX graph from ``memory_graph.json``.

        If the file does not exist, an empty graph is kept.
        """
        if not os.path.exists(self._graph_path):
            return

        with open(self._graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._graph = nx.DiGraph()
        for node_entry in data.get("nodes", []):
            nid = node_entry["id"]
            attrs = {k: v for k, v in node_entry.items() if k != "id"}
            self._graph.add_node(nid, **attrs)
        for edge_entry in data.get("edges", []):
            source = edge_entry["source"]
            target = edge_entry["target"]
            attrs = {
                k: v
                for k, v in edge_entry.items()
                if k not in ("source", "target")
            }
            self._graph.add_edge(source, target, **attrs)
