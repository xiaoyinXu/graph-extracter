"""
Unit tests for GraphStore.search_and_traverse() and
KnowledgeGraphRetriever.search_subgraph().

Tests cover:
  - Empty store / no-hit edge cases
  - Deduplication of hits that share a node_id
  - traverse_from_root=False: traversal starts at the matched node
  - traverse_from_root=True:  traversal starts at the topmost ancestor
  - Multiple hits → subgraphs are merged
  - Mermaid output structure
  - KnowledgeGraphRetriever.search_subgraph end-to-end
"""
from __future__ import annotations

from unittest.mock import patch

import networkx as nx
import pytest
from langchain_core.vectorstores import InMemoryVectorStore

from graph.models import KnowledgeGraph
from graph.storage import GraphStore, SubgraphResult
from tests.conftest import FakeEmbeddings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_store(kg: KnowledgeGraph) -> GraphStore:
    """Build a GraphStore without real embeddings or Elasticsearch."""
    def fake_vs(docs, emb):
        return InMemoryVectorStore.from_documents(docs, emb) if docs else InMemoryVectorStore(emb)

    with patch("graph.storage._create_embeddings", return_value=FakeEmbeddings()), \
         patch("graph.storage._create_vector_store", side_effect=fake_vs):
        return GraphStore.from_kg(kg)


def _mock_hits(*node_ids: str, base_score: float = 0.9) -> list[dict]:
    """Build a fake similarity_search return value for the given node IDs."""
    return [
        {
            "node_id": nid,
            "node_type": "SOPRule",
            "root_id": "sop1",
            "score": base_score - i * 0.05,
            "text": f"text for {nid}",
        }
        for i, nid in enumerate(node_ids)
    ]


# ---------------------------------------------------------------------------
# SubgraphResult structure
# ---------------------------------------------------------------------------

class TestSubgraphResultStructure:
    def test_has_required_keys(self, minimal_kg):
        store = make_store(minimal_kg)
        result = store.search_and_traverse("配送超时", k=3)
        assert hasattr(result, "hits")
        assert hasattr(result, "start_node_ids")
        assert hasattr(result, "subgraph")
        assert hasattr(result, "mermaid")

    def test_subgraph_is_nx_digraph(self, minimal_kg):
        store = make_store(minimal_kg)
        result = store.search_and_traverse("配送超时", k=3)
        assert isinstance(result.subgraph, nx.DiGraph)

    def test_mermaid_starts_with_graph_td(self, minimal_kg):
        store = make_store(minimal_kg)
        result = store.search_and_traverse("配送超时", k=3)
        assert result.mermaid.startswith("graph TD")

    def test_hits_is_list(self, minimal_kg):
        store = make_store(minimal_kg)
        result = store.search_and_traverse("配送超时", k=3)
        assert isinstance(result.hits, list)

    def test_start_node_ids_is_list(self, minimal_kg):
        store = make_store(minimal_kg)
        result = store.search_and_traverse("配送超时", k=3)
        assert isinstance(result.start_node_ids, list)


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

class TestEmptyStore:
    def test_empty_store_returns_empty_subgraph(self):
        empty_store = make_store(KnowledgeGraph())
        result = empty_store.search_and_traverse("anything", k=3)
        assert result.hits == []
        assert result.start_node_ids == []
        assert result.subgraph.number_of_nodes() == 0
        assert result.mermaid == "graph TD"


# ---------------------------------------------------------------------------
# traverse_from_root=False (default) — stay at matched node
# ---------------------------------------------------------------------------

class TestTraverseFromMatchedNode:
    def test_matched_leaf_node_returns_single_node(self, minimal_kg):
        """tool1 is a leaf (no outgoing edges) → subgraph contains only tool1."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("tool1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        sg = result.subgraph
        assert "tool1" in sg.nodes
        assert sg.number_of_nodes() == 1

    def test_matched_rule_node_includes_children(self, minimal_kg):
        """rule1 has subrule1 and tool1 as children → both appear in subgraph."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        sg = result.subgraph
        assert "rule1" in sg.nodes
        assert "subrule1" in sg.nodes
        assert "tool1" in sg.nodes

    def test_matched_rule_node_excludes_ancestors(self, minimal_kg):
        """Ancestors of rule1 (sop1, step1) must NOT appear when traverse_from_root=False."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        sg = result.subgraph
        assert "sop1" not in sg.nodes
        assert "step1" not in sg.nodes

    def test_start_node_ids_equals_matched_node(self, minimal_kg):
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        assert result.start_node_ids == ["rule1"]


# ---------------------------------------------------------------------------
# traverse_from_root=True — walk up first, then traverse full subtree
# ---------------------------------------------------------------------------

class TestTraverseFromRoot:
    def test_matched_rule_returns_full_sop_tree(self, minimal_kg):
        """When rule1 is matched but traverse_from_root=True, the root SOP
        is found first, so the full SOP subtree is returned."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=True)
        sg = result.subgraph
        # All nodes must be present because we traversed from the root SOP
        for node in minimal_kg.nodes:
            assert node.id in sg.nodes, f"{node.id} missing from subgraph"

    def test_root_node_id_is_sop_not_rule(self, minimal_kg):
        """Root traversal must begin at sop1, not rule1."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=True)
        assert "sop1" in result.start_node_ids
        assert "rule1" not in result.start_node_ids

    def test_matched_root_node_unchanged_by_traverse_from_root(self, minimal_kg):
        """sop1 has no ancestors → traverse_from_root has no effect on the root."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("sop1")):
            result_default = store.search_and_traverse("anything", traverse_from_root=False)
            result_root = store.search_and_traverse("anything", traverse_from_root=True)
        assert set(result_default.subgraph.nodes) == set(result_root.subgraph.nodes)

    def test_subrule_traverse_from_root_reaches_all_nodes(self, minimal_kg):
        """subrule1 → ancestors: sop1, step1, rule1 → root is sop1 → all nodes."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("subrule1")):
            result = store.search_and_traverse("anything", traverse_from_root=True)
        sg = result.subgraph
        assert len(sg.nodes) == len(minimal_kg.nodes)


# ---------------------------------------------------------------------------
# Deduplication of hits sharing a node_id
# ---------------------------------------------------------------------------

class TestHitDeduplication:
    def test_duplicate_node_ids_deduplicated_in_root_ids(self, minimal_kg):
        """Vector index may return multiple docs for the same node_id.
        start_node_ids must contain each ID at most once."""
        store = make_store(minimal_kg)
        dup_hits = [
            {"node_id": "rule1", "node_type": "SOPRule", "root_id": "sop1", "score": 0.9, "text": "t1"},
            {"node_id": "rule1", "node_type": "SOPRule", "root_id": "sop1", "score": 0.8, "text": "t2"},
        ]
        with patch.object(store, "similarity_search", return_value=dup_hits):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        assert result.start_node_ids.count("rule1") == 1

    def test_best_score_kept_when_deduplicating(self, minimal_kg):
        """When two hits share a node_id, the one with the higher score is kept."""
        store = make_store(minimal_kg)
        dup_hits = [
            {"node_id": "rule1", "node_type": "SOPRule", "root_id": "sop1", "score": 0.7, "text": "low"},
            {"node_id": "rule1", "node_type": "SOPRule", "root_id": "sop1", "score": 0.95, "text": "high"},
        ]
        with patch.object(store, "similarity_search", return_value=dup_hits):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        # Both hits are returned in `hits` (raw), but start_node_ids is deduplicated
        assert len(result.hits) == 2
        assert len(result.start_node_ids) == 1


# ---------------------------------------------------------------------------
# Multiple hits → merged subgraph
# ---------------------------------------------------------------------------

class TestMultipleHitsMerge:
    def test_two_independent_hits_merge_their_subgraphs(self, minimal_kg):
        """rule1 and step2 are in different parts of the tree.
        The merged subgraph should contain nodes from both traversals."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search",
                          return_value=_mock_hits("rule1", "step2")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        sg = result.subgraph
        # rule1 subtree: rule1, subrule1, tool1
        assert "rule1" in sg.nodes
        assert "subrule1" in sg.nodes
        assert "tool1" in sg.nodes
        # step2 subtree: step2 (leaf)
        assert "step2" in sg.nodes

    def test_two_hits_both_listed_in_start_node_ids(self, minimal_kg):
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search",
                          return_value=_mock_hits("rule1", "step2")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        assert "rule1" in result.start_node_ids
        assert "step2" in result.start_node_ids

    def test_overlapping_hits_with_root_traversal_deduplicated(self, minimal_kg):
        """rule1 and rule2 both trace back to sop1.
        With traverse_from_root=True, start_node_ids should contain sop1 only once."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search",
                          return_value=_mock_hits("rule1", "rule2")):
            result = store.search_and_traverse("anything", traverse_from_root=True)
        assert result.start_node_ids.count("sop1") == 1


# ---------------------------------------------------------------------------
# Mermaid output for search_and_traverse
# ---------------------------------------------------------------------------

class TestSearchTraverseMermaid:
    def test_mermaid_contains_matched_node(self, minimal_kg):
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        assert "rule1" in result.mermaid

    def test_mermaid_contains_child_nodes(self, minimal_kg):
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=False)
        assert "subrule1" in result.mermaid
        assert "tool1" in result.mermaid

    def test_full_tree_mermaid_via_root_traversal(self, minimal_kg, capsys):
        """Print Mermaid for root traversal — useful for visual inspection."""
        store = make_store(minimal_kg)
        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = store.search_and_traverse("anything", traverse_from_root=True)
        print(f"\n--- search_and_traverse mermaid (traverse_from_root=True) ---\n"
              f"{result.mermaid}\n")
        captured = capsys.readouterr()
        assert "graph TD" in captured.out
        assert "sop1" in captured.out


# ---------------------------------------------------------------------------
# _find_root helper
# ---------------------------------------------------------------------------

class TestFindRoot:
    def test_root_node_returns_itself(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store._find_root("sop1") == "sop1"

    def test_child_node_returns_sop(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store._find_root("rule1") == "sop1"

    def test_deep_node_returns_sop(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store._find_root("subrule1") == "sop1"

    def test_leaf_tool_node_returns_itself(self, minimal_kg):
        """tool1 has no incoming SOP hierarchy edges (predecessors only from rule1),
        so its root is rule1 (the first predecessor without further predecessors
        going up is sop1 via rule1 → step1 → sop1)."""
        store = make_store(minimal_kg)
        # tool1 predecessors: rule1 → step1 → sop1
        root = store._find_root("tool1")
        assert root == "sop1"


# ---------------------------------------------------------------------------
# KnowledgeGraphRetriever.search_subgraph end-to-end
# ---------------------------------------------------------------------------

class TestRetrieverSearchSubgraph:
    def test_search_subgraph_returns_subgraph_result(self, minimal_kg):
        from graph.retriever import KnowledgeGraphRetriever
        store = make_store(minimal_kg)
        retriever = KnowledgeGraphRetriever.__new__(KnowledgeGraphRetriever)
        retriever.store = store

        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = retriever.search_subgraph("用户询问配送时间")
        from graph.storage import SubgraphResult
        assert isinstance(result, SubgraphResult)
        assert isinstance(result.subgraph, nx.DiGraph)
        assert isinstance(result.mermaid, str)

    def test_search_subgraph_traverse_from_root_flag(self, minimal_kg):
        from graph.retriever import KnowledgeGraphRetriever
        store = make_store(minimal_kg)
        retriever = KnowledgeGraphRetriever.__new__(KnowledgeGraphRetriever)
        retriever.store = store

        with patch.object(store, "similarity_search", return_value=_mock_hits("rule1")):
            result = retriever.search_subgraph("用户询问配送时间", traverse_from_root=True)
        assert "sop1" in result.subgraph.nodes

    def test_search_subgraph_k_parameter(self, minimal_kg):
        """k is forwarded to similarity_search."""
        from graph.retriever import KnowledgeGraphRetriever
        store = make_store(minimal_kg)
        retriever = KnowledgeGraphRetriever.__new__(KnowledgeGraphRetriever)
        retriever.store = store

        with patch.object(store, "similarity_search", return_value=[]) as mock_search:
            retriever.search_subgraph("test", k=7)
        mock_search.assert_called_once_with("test", k=7)

    def test_search_subgraph_integration_no_mock(self, minimal_kg):
        """Integration test: real FakeEmbeddings, no mocking of similarity_search."""
        from graph.retriever import KnowledgeGraphRetriever
        store = make_store(minimal_kg)
        retriever = KnowledgeGraphRetriever.__new__(KnowledgeGraphRetriever)
        retriever.store = store

        result = retriever.search_subgraph("配送超时", k=3)
        assert result.mermaid.startswith("graph TD")
        assert len(result.hits) <= 3
