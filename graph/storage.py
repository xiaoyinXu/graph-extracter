"""
Graph storage layer: load/save KnowledgeGraph JSON + build vector index.

The vector index is rebuilt from graph.json on every load using Elasticsearch
as the backing store (ES_HOSTS / ES_INDEX from .env).  The index is dropped
and recreated on each build so the store always reflects the current graph.

Indexed text fields are declared in schema/customer_service.yaml via
``x_index: true`` on individual attributes — no hardcoding in this file.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch.vectorstores import DistanceStrategy
from langchain_openai import OpenAIEmbeddings

from graph.models import GraphEdge, GraphNode, KnowledgeGraph
from graph.schema_loader import FieldSpec, get_class_field_specs, get_relation_specs, get_root_class
from graph.utils import subgraph_to_mermaid

load_dotenv()

logger = logging.getLogger(__name__)

# Path to the LinkML schema — resolved relative to this file so it works
# regardless of the caller's working directory.
_SCHEMA_PATH = str(Path(__file__).resolve().parent.parent / "schema" / "customer_service.yaml")
# Override via env var if needed (e.g. in tests or alternative deployments)
SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", _SCHEMA_PATH)

# Only string/str fields carry free text worth embedding.
_STRING_RANGES: frozenset[str] = frozenset({"string", "str"})
# Hard cap on characters embedded per field value to control token cost.
_INDEX_MAX_CHARS: int = 300

DEFAULT_GRAPH_PATH = "data/graph.json"

# Root class and its ancestry-id key derived from the schema at module load.
_ROOT_CLASS: str = get_root_class(SCHEMA_PATH)
# Key name stored in child node.data to reference their root ancestor,
# e.g. "SOP" → "sop_id"
_ROOT_ID_KEY: str = f"{_ROOT_CLASS.lower()}_id"


def _field_to_output_key(field_name: str) -> str:
    """Map schema field name to context-dict output key.

    ``used_tools`` → ``tools`` (strip the ``used_`` prefix).
    Everything else keeps its field name unchanged.
    """
    if field_name.startswith("used_"):
        return field_name[5:]
    return field_name


@dataclass
class SubgraphResult:
    """Result of a vector search + subgraph traversal.

    Attributes
    ----------
    hits : list[dict]
        Raw score-ordered vector-search results (multiple docs per node possible).
    start_node_ids : list[str]
        Deduplicated node IDs used as traversal entry points.  When
        ``traverse_from_root=True`` these will be root entity IDs; when
        ``traverse_from_root=False`` they are the matched node IDs.
    subgraph : nx.DiGraph
        Merged connected subgraph of all traversal roots.  **Not JSON-serializable.**
    mermaid : str
        Mermaid ``graph TD`` diagram of the subgraph.
    """

    hits: list[dict[str, Any]]
    start_node_ids: list[str]
    subgraph: nx.DiGraph
    mermaid: str


# ---------------------------------------------------------------------------
# Embeddings factory
# ---------------------------------------------------------------------------

def _create_embeddings() -> Embeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
        api_key=os.getenv("EMBEDDING_API_KEY"),
        base_url=os.getenv("EMBEDDING_BASE_URL"),
    )


def _create_vector_store(
    docs: list[Document],
    embeddings: Embeddings,
) -> VectorStore:
    """Build an ElasticsearchStore from *docs*, dropping any existing index first.

    Index configuration (ES 8.x dense_vector):
      - field type : dense_vector, 1536 dims, cosine similarity
      - strategy   : ApproxRetrievalStrategy (kNN, default)

    Reads ES_HOSTS and ES_INDEX from the environment (defaults to
    ``http://localhost:9200`` and ``kg_index`` respectively).  The index is
    always rebuilt from scratch so the store exactly mirrors the current graph.

    This function is intentionally kept as a module-level callable so unit
    tests can patch it with an in-memory store to avoid hitting ES.
    """
    es_url = os.getenv("ES_HOSTS", "http://localhost:9200")
    index_name = os.getenv("ES_INDEX", "kg_index")

    # Always drop and recreate so the index mirrors the current graph exactly.
    es_client = Elasticsearch(es_url)
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)

    store_kwargs: dict[str, Any] = dict(
        index_name=index_name,
        embedding=embeddings,
        es_url=es_url,
        distance_strategy=DistanceStrategy.COSINE,
        num_dimensions=1536,
        # schema is stored as a nested object but not full-text indexed
        metadata_mappings={"schema": {"type": "object", "enabled": False}},
    )

    if not docs:
        # Empty graph: initialise the store (creates the index with correct mapping).
        return ElasticsearchStore(**store_kwargs)

    return ElasticsearchStore.from_documents(documents=docs, **store_kwargs)


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
    vector_store : InMemoryVectorStore; each doc has metadata {node_id, node_type, root_id}
    node_map   : node_id → GraphNode (fast attribute lookup)
    schema_path: LinkML YAML path used for this graph; empty = module-level default
    """
    nx_graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    vector_store: Optional[VectorStore] = None
    node_map: dict[str, GraphNode] = field(default_factory=dict)
    schema_path: str = ""

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
        # Restore the schema that was used to build this graph (persisted in JSON)
        store = cls(schema_path=kg.schema_path or "")
        store._build_nx(kg)
        store._build_vector_index(kg)
        return store

    @classmethod
    def from_kg(cls, kg: KnowledgeGraph, schema_path: Optional[str] = None) -> "GraphStore":
        """Build store directly from an in-memory KnowledgeGraph."""
        effective_schema = schema_path or kg.schema_path or ""
        store = cls(schema_path=effective_schema)
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
        """Build Elasticsearch vector index driven entirely by schema x_index annotations.

        For each node type, ``get_class_field_specs`` is called once to discover
        which fields carry ``x_index: true``.  Multivalued fields produce one
        Document per list item; scalar fields produce a single Document.
        No class names or field names are hardcoded here.
        """
        _sp = self.schema_path or SCHEMA_PATH
        _root_class = get_root_class(_sp)
        _root_id_key = f"{_root_class.lower()}_id"

        embeddings = _create_embeddings()
        docs: list[Document] = []

        # Cache indexed field specs per node_type (one schema parse per type).
        specs_cache: dict[str, list[FieldSpec]] = {}

        for node in kg.nodes:
            d = node.data
            ntype = node.node_type

            # root_id: use the ancestor-id field when present (child nodes);
            # for root nodes that own the entity hierarchy, fall back to node.id.
            root_id: str = d.get(_root_id_key) or node.id

            if ntype not in specs_cache:
                try:
                    specs_cache[ntype] = [
                        s for s in get_class_field_specs(_sp, ntype)
                        if s.indexed and s.range.lower() in _STRING_RANGES
                    ]
                except Exception as exc:
                    logger.warning(
                        "[storage] could not load field specs for %s from %s: %s",
                        ntype, _sp, exc,
                    )
                    specs_cache[ntype] = []

            indexed_specs = specs_cache[ntype]
            if not indexed_specs:
                continue

            def _meta(node_id: str = node.id, node_type: str = ntype,
                      _root_id: str = root_id, _data: dict = d) -> dict[str, Any]:
                return {
                    "node_id": node_id,
                    "node_type": node_type,
                    "root_id": _root_id,
                    "schema": _data,
                }

            for spec in indexed_specs:
                value = d.get(spec.name)
                if value is None:
                    continue

                if spec.multivalued and isinstance(value, list):
                    # Each list item becomes its own Document for better recall.
                    for item in value:
                        text = str(item).strip()
                        if len(text) > spec.min_index_chars:
                            docs.append(Document(
                                page_content=text[:_INDEX_MAX_CHARS],
                                metadata=_meta(),
                            ))
                else:
                    text = str(value).strip()
                    if len(text) > spec.min_index_chars:
                        docs.append(Document(
                            page_content=text[:_INDEX_MAX_CHARS],
                            metadata=_meta(),
                        ))

        logger.info("[storage] indexing %d documents for %d nodes", len(docs), len(kg.nodes))
        self.vector_store = _create_vector_store(docs, embeddings)

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
        return KnowledgeGraph(nodes=nodes, edges=edges, schema_path=self.schema_path or None)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def similarity_search(
        self, query: str, k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Return top-k semantically similar nodes.

        Returns list of dicts: {node_id, node_type, root_id, score, text}
        """
        if self.vector_store is None:
            return []
        results = self.vector_store.similarity_search_with_score(query, k=k)
        hits = []
        for doc, score in results:
            hits.append({
                "node_id": doc.metadata["node_id"],
                "node_type": doc.metadata["node_type"],
                "root_id": doc.metadata.get("root_id", ""),
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
        """Walk edges in reverse to collect ancestor chain."""
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

    def get_root_context(self, root_id: str) -> dict[str, Any]:
        """Return the full root-node tree sorted by index fields (schema-driven).

        Output structure mirrors the schema relation hierarchy:
        {
          "<root_lower>": {...},           # e.g. "sop": {...}
          "<field_key>": [                 # e.g. "steps": [
            {
              "<child_singular>": {...},   # e.g. "step": {...}
              "<grandchild_key>": [        # e.g. "rules": [
                {
                  "<gchild_singular>": {...},
                  "<leaf_field>": [node, ...],   # flat leaf lists
                }
              ]
            }
          ]
        }
        """
        root_node = self.get_node(root_id)
        if not root_node:
            return {}

        result: dict[str, Any] = {_ROOT_CLASS.lower(): root_node}
        root_rel_specs = get_relation_specs(SCHEMA_PATH, _ROOT_CLASS)
        for spec in root_rel_specs:
            output_key = _field_to_output_key(spec.field_name)
            result[output_key] = self._build_relation_list(root_id, _ROOT_CLASS, spec)

        return result

    def _build_relation_list(
        self,
        node_id: str,
        class_name: str,
        spec: Any,
    ) -> list[Any]:
        """Build the child list for one relation spec (schema-driven, recursive)."""
        edges = [
            (v, d)
            for _, v, d in self.nx_graph.out_edges(node_id, data=True)
            if d.get("edge_type") == spec.edge_type
        ]

        # Sort by index field from edge metadata if present
        try:
            child_field_specs = get_class_field_specs(SCHEMA_PATH, spec.target_class)
            index_field = next(
                (s.name for s in child_field_specs if s.name.endswith("_index")),
                None,
            )
        except Exception:
            index_field = None

        if index_field:
            edges.sort(key=lambda x: x[1].get(index_field, 0))

        output_key = _field_to_output_key(spec.field_name)
        singular_key = output_key.rstrip("s")  # "steps" → "step", "rules" → "rule"

        child_rel_specs = get_relation_specs(SCHEMA_PATH, spec.target_class)

        items: list[Any] = []
        for child_id, _ in edges:
            child_node = self.get_node(child_id)
            if not child_node:
                continue

            if child_rel_specs:
                # Non-leaf: wrap in dict and recurse
                entry: dict[str, Any] = {singular_key: child_node}
                for child_spec in child_rel_specs:
                    child_output_key = _field_to_output_key(child_spec.field_name)
                    entry[child_output_key] = self._build_relation_list(
                        child_id, spec.target_class, child_spec
                    )
                items.append(entry)
            else:
                # Leaf node: flat list
                items.append(child_node)

        return items

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
        Return focused context for a specific node.
        Includes the node itself + its ancestor chain + siblings at same level
        + non-tool children (e.g. sub-rules).
        """
        node = self.get_node(node_id)
        if not node:
            return {}

        ancestors = self.get_ancestors(node_id)

        # Siblings: other nodes connected from the same parent with the same edge type
        preds = list(self.nx_graph.predecessors(node_id))
        siblings: list[dict] = []
        if preds:
            parent_id = preds[0]
            parent_type = self.nx_graph.nodes[parent_id].get("node_type", "")
            # Derive the edge type from the schema (parent → this node)
            parent_rel_specs = get_relation_specs(SCHEMA_PATH, parent_type)
            edge_type = next(
                (spec.edge_type for spec in parent_rel_specs if spec.target_class == node_type),
                None,
            )
            if edge_type:
                siblings = [
                    self.get_node(v)
                    for _, v, d in self.nx_graph.out_edges(parent_id, data=True)
                    if d.get("edge_type") == edge_type and v != node_id
                ]

        # Children: non-tool relation children (e.g. sub-rules)
        children: list[dict] = []
        node_rel_specs = get_relation_specs(SCHEMA_PATH, node_type)
        for spec in node_rel_specs:
            if spec.edge_type.startswith("USES_"):
                continue  # skip tool/service relations
            for _, v, d in self.nx_graph.out_edges(node_id, data=True):
                if d.get("edge_type") == spec.edge_type:
                    child = self.get_node(v)
                    if child:
                        children.append(child)

        return {
            "node": node,
            "ancestors": ancestors,
            "siblings": [s for s in siblings if s],
            "children": children,
        }

    # ------------------------------------------------------------------
    # Vector search + subgraph traversal (combined pipeline)
    # ------------------------------------------------------------------

    def _find_root(self, node_id: str) -> str:
        """Walk upward through predecessors to find the topmost ancestor node.

        Used only as a fallback when a hit has no ``root_id`` (e.g. Tool nodes).
        Prefer resolving via ``root_id`` for hierarchy nodes.
        """
        ancestors = self.get_ancestors(node_id)
        return ancestors[0]["id"] if ancestors else node_id

    def search_and_traverse(
        self,
        query: str,
        k: int = 3,
        traverse_from_root: bool = False,
    ) -> SubgraphResult:
        """
        Vector-search for nodes matching *query*, then traverse each matched
        node's connected subgraph and return a merged result.

        Parameters
        ----------
        query : str
            Natural-language text to search for.
        k : int
            Number of top vector-search document hits to consider (default 3).
            Note: the vector index stores multiple docs per node (one per indexed
            text field), so the actual number of unique nodes may be lower than k.
        traverse_from_root : bool
            If ``True``, resolve each hit's root entity node (via the indexed
            ``root_id``) and traverse the **full entity subtree** from there.
            If ``False`` (default), traverse the subgraph downward from the
            matched node itself for a focused result.

        Returns
        -------
        SubgraphResult
            ``hits``           — raw score-ordered vector-search results
            ``start_node_ids`` — deduplicated IDs used as traversal entry points
            ``subgraph``       — merged ``nx.DiGraph`` of all traversed nodes
            ``mermaid``        — Mermaid ``graph TD`` diagram of the subgraph
        """
        hits = self.similarity_search(query, k=k)

        if not hits:
            return SubgraphResult(
                hits=[],
                start_node_ids=[],
                subgraph=nx.DiGraph(),
                mermaid="graph TD",
            )

        # Deduplicate hits: keep best score per node_id (vector index stores
        # multiple docs per node for different text fields, e.g. trigger samples)
        seen_hits: dict[str, dict[str, Any]] = {}
        for hit in hits:
            nid = hit["node_id"]
            if nid not in seen_hits or hit["score"] > seen_hits[nid]["score"]:
                seen_hits[nid] = hit
        deduped_hits = list(seen_hits.values())

        # Determine traversal entry point for each unique matched node.
        # When traverse_from_root=True, use the indexed root_id as the canonical
        # root — this avoids ambiguous predecessor walks on multi-parent nodes.
        start_ids: list[str] = []
        seen_starts: set[str] = set()
        for hit in deduped_hits:
            if traverse_from_root:
                root_id = hit.get("root_id", "")
                start = root_id if root_id and root_id in self.nx_graph else self._find_root(hit["node_id"])
            else:
                start = hit["node_id"]
            if start not in seen_starts:
                seen_starts.add(start)
                start_ids.append(start)

        # Merge all connected subgraphs (safe to compose: all share the same
        # base graph so node attribute conflicts cannot occur)
        subgraphs = [self.get_connected_subgraph(sid) for sid in start_ids]
        merged: nx.DiGraph = nx.compose_all(subgraphs) if subgraphs else nx.DiGraph()

        mermaid = subgraph_to_mermaid(merged, self.node_map)

        return SubgraphResult(
            hits=hits,
            start_node_ids=start_ids,
            subgraph=merged,
            mermaid=mermaid,
        )
