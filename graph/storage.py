"""
Graph storage layer: load/save KnowledgeGraph JSON + build vector index.

The vector index is rebuilt deterministically from graph.json on every load,
so no separate index file is needed. For large graphs, swap InMemoryVectorStore
with a persistent store (Chroma, FAISS, etc.).

Indexed text fields (for retrieval coverage):
  SOP          → trigger_samples, sub_scenario, name
  SOPStep      → goal
  SOPRule      → condition (+ reference_script snippet)
  SOPSubRule   → condition
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from graph.models import GraphEdge, GraphNode, KnowledgeGraph

load_dotenv()

DEFAULT_GRAPH_PATH = "data/graph.json"


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def _create_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------

@dataclass
class GraphStore:
    """
    In-memory NetworkX graph + vector index over searchable text fields.

    Attributes
    ----------
    nx_graph   : directed graph; nodes have 'node_type' + all data fields as attrs
    vector_store : InMemoryVectorStore; each doc has metadata {node_id, node_type, sop_id}
    node_map   : node_id → GraphNode (fast attribute lookup)
    """
    nx_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    vector_store: Optional[InMemoryVectorStore] = None
    node_map: dict[str, GraphNode] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str = DEFAULT_GRAPH_PATH) -> "GraphStore":
        """Load graph from JSON and rebuild vector index."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Graph file not found: {path}")

        with open(p, encoding="utf-8") as f:
            raw = json.load(f)

        kg = KnowledgeGraph.model_validate(raw)
        store = cls()
        store._build_nx(kg)
        store._build_vector_index(kg)
        return store

    @classmethod
    def from_kg(cls, kg: KnowledgeGraph) -> "GraphStore":
        """Build store directly from an in-memory KnowledgeGraph."""
        store = cls()
        store._build_nx(kg)
        store._build_vector_index(kg)
        return store

    def save(self, path: str = DEFAULT_GRAPH_PATH) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        kg = self._to_kg()
        with open(path, "w", encoding="utf-8") as f:
            f.write(kg.model_dump_json(indent=2))
        print(f"[storage] saved {self.nx_graph.number_of_nodes()} nodes, "
              f"{self.nx_graph.number_of_edges()} edges → {path}")

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_nx(self, kg: KnowledgeGraph) -> None:
        G = nx.DiGraph()
        for node in kg.nodes:
            self.node_map[node.id] = node
            G.add_node(node.id, node_type=node.node_type, **node.data)
        for edge in kg.edges:
            G.add_edge(edge.source, edge.target, edge_type=edge.edge_type, **edge.metadata)
        self.nx_graph = G

    def _build_vector_index(self, kg: KnowledgeGraph) -> None:
        """Build InMemoryVectorStore from searchable text across all node types."""
        embeddings = _create_embeddings()
        docs: list[Document] = []

        for node in kg.nodes:
            d = node.data
            ntype = node.node_type
            sop_id = d.get("sop_id", node.id if ntype == "SOP" else "")

            if ntype == "SOP":
                # index each trigger sample as a separate doc for better recall
                for sample in d.get("trigger_samples", []):
                    if sample.strip():
                        docs.append(Document(
                            page_content=sample,
                            metadata={"node_id": node.id, "node_type": ntype, "sop_id": node.id},
                        ))
                # index name and sub_scenario together
                text_parts = [d.get("name", ""), d.get("sub_scenario", "")]
                combined = " ".join(t for t in text_parts if t)
                if combined.strip():
                    docs.append(Document(
                        page_content=combined,
                        metadata={"node_id": node.id, "node_type": ntype, "sop_id": node.id},
                    ))

            elif ntype == "SOPStep":
                goal = d.get("goal", "")
                if goal.strip():
                    docs.append(Document(
                        page_content=goal,
                        metadata={"node_id": node.id, "node_type": ntype, "sop_id": sop_id},
                    ))

            elif ntype == "SOPRule":
                condition = d.get("condition", "")
                if condition.strip():
                    docs.append(Document(
                        page_content=condition,
                        metadata={"node_id": node.id, "node_type": ntype, "sop_id": sop_id},
                    ))
                # also index a snippet of the reference script for keyword matching
                script = d.get("reference_script") or ""
                if len(script) > 20:
                    docs.append(Document(
                        page_content=script[:300],
                        metadata={"node_id": node.id, "node_type": "SOPRule_script",
                                  "sop_id": sop_id},
                    ))

            elif ntype == "SOPSubRule":
                condition = d.get("condition", "")
                if condition.strip():
                    docs.append(Document(
                        page_content=condition,
                        metadata={"node_id": node.id, "node_type": ntype, "sop_id": sop_id},
                    ))

        if docs:
            self.vector_store = InMemoryVectorStore.from_documents(docs, embeddings)
        else:
            self.vector_store = InMemoryVectorStore(embeddings)

    def _to_kg(self) -> KnowledgeGraph:
        nodes = [self.node_map[nid] for nid in self.nx_graph.nodes]
        edges = [
            GraphEdge(
                source=u, target=v,
                edge_type=data.get("edge_type", ""),
                metadata={k: v_ for k, v_ in data.items() if k != "edge_type"},
            )
            for u, v, data in self.nx_graph.edges(data=True)
        ]
        return KnowledgeGraph(nodes=nodes, edges=edges)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Return top-k semantically similar nodes.

        Returns list of dicts: {node_id, node_type, sop_id, score, text}
        """
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        hits = []
        for doc, score in results:
            hits.append({
                "node_id": doc.metadata["node_id"],
                "node_type": doc.metadata["node_type"],
                "sop_id": doc.metadata.get("sop_id", ""),
                "score": float(score),
                "text": doc.page_content,
            })
        return hits

    def get_node(self, node_id: str) -> Optional[dict[str, Any]]:
        """Return node data dict (includes node_type)."""
        node = self.node_map.get(node_id)
        if node is None:
            return None
        return {"id": node.id, "node_type": node.node_type, **node.data}

    def get_ancestors(self, node_id: str) -> list[dict[str, Any]]:
        """Walk edges in reverse to collect [SOP → Step → Rule] chain."""
        ancestors = []
        current = node_id
        visited = set()
        while current:
            visited.add(current)
            preds = list(self.nx_graph.predecessors(current))
            if not preds:
                break
            parent = preds[0]
            if parent in visited:
                break
            node = self.get_node(parent)
            if node:
                ancestors.insert(0, node)
            current = parent
        return ancestors

    def get_sop_context(self, sop_id: str) -> dict[str, Any]:
        """
        Return the full SOP tree sorted by step_index / rule_index.

        Returns:
        {
          "sop": {...},
          "steps": [
            {
              "step": {...},
              "rules": [
                {
                  "rule": {...},
                  "sub_rules": [...],
                  "tools": [...]
                }
              ]
            }
          ]
        }
        """
        sop_node = self.get_node(sop_id)
        if not sop_node:
            return {}

        result: dict = {"sop": sop_node, "steps": []}

        step_edges = [
            (v, d)
            for _, v, d in self.nx_graph.out_edges(sop_id, data=True)
            if d.get("edge_type") == "HAS_STEP"
        ]
        step_edges.sort(key=lambda x: x[1].get("step_index", 0))

        for step_id, _ in step_edges:
            step_node = self.get_node(step_id)
            if not step_node:
                continue

            rule_edges = [
                (v, d)
                for _, v, d in self.nx_graph.out_edges(step_id, data=True)
                if d.get("edge_type") == "HAS_RULE"
            ]
            rule_edges.sort(key=lambda x: x[1].get("rule_index", 0))

            step_rules = []
            for rule_id, _ in rule_edges:
                rule_node = self.get_node(rule_id)
                if not rule_node:
                    continue

                sub_rules = [
                    self.get_node(v)
                    for _, v, d in self.nx_graph.out_edges(rule_id, data=True)
                    if d.get("edge_type") == "HAS_SUB_RULE"
                ]
                tools = [
                    self.get_node(v)
                    for _, v, d in self.nx_graph.out_edges(rule_id, data=True)
                    if d.get("edge_type") == "USES_TOOL"
                ]
                step_rules.append({
                    "rule": rule_node,
                    "sub_rules": [s for s in sub_rules if s],
                    "tools": [t for t in tools if t],
                })

            result["steps"].append({"step": step_node, "rules": step_rules})

        return result

    def get_connected_subgraph(self, node_id: str) -> nx.DiGraph:
        """
        Return a copy of the induced subgraph containing node_id and all nodes
        reachable from it by following directed edges.

        Returns an empty DiGraph when node_id is not in the graph.
        """
        if node_id not in self.nx_graph:
            return nx.DiGraph()
        reachable = nx.descendants(self.nx_graph, node_id)
        reachable.add(node_id)
        return self.nx_graph.subgraph(reachable).copy()

    def get_rule_context(self, node_id: str, node_type: str) -> dict[str, Any]:
        """
        Return focused context for a specific rule or sub-rule node.
        Includes the node itself + its parent chain + siblings at same level.
        """
        node = self.get_node(node_id)
        if not node:
            return {}

        ancestors = self.get_ancestors(node_id)

        # siblings: other rules in the same step
        preds = list(self.nx_graph.predecessors(node_id))
        siblings = []
        if preds:
            parent_id = preds[0]
            parent_type = self.nx_graph.nodes[parent_id].get("node_type", "")
            edge_type = "HAS_RULE" if parent_type == "SOPStep" else "HAS_SUB_RULE"
            siblings = [
                self.get_node(v)
                for _, v, d in self.nx_graph.out_edges(parent_id, data=True)
                if d.get("edge_type") == edge_type and v != node_id
            ]

        # children of this node
        children = []
        if node_type in ("SOPRule",):
            children = [
                self.get_node(v)
                for _, v, d in self.nx_graph.out_edges(node_id, data=True)
                if d.get("edge_type") == "HAS_SUB_RULE"
            ]

        return {
            "node": node,
            "ancestors": ancestors,
            "siblings": [s for s in siblings if s],
            "children": [c for c in children if c],
        }
