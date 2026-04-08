"""
Unit tests for graph/storage.py — GraphStore (no real API calls).

All tests use FakeEmbeddings from conftest.py, patched into graph.storage
so that _build_vector_index() never calls OpenAI or Elasticsearch.
"""
from __future__ import annotations

import pytest
from unittest.mock import patch
from langchain_core.vectorstores import InMemoryVectorStore

from graph.models import GraphEdge, GraphNode, KnowledgeGraph
from graph.storage import GraphStore
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


# ---------------------------------------------------------------------------
# Build: nx graph structure
# ---------------------------------------------------------------------------

class TestBuildNx:
    def test_node_count(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store.nx_graph.number_of_nodes() == len(minimal_kg.nodes)

    def test_edge_count(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store.nx_graph.number_of_edges() == len(minimal_kg.edges)

    def test_node_type_attribute(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store.nx_graph.nodes["sop1"]["node_type"] == "SOP"
        assert store.nx_graph.nodes["step1"]["node_type"] == "SOPStep"
        assert store.nx_graph.nodes["rule1"]["node_type"] == "SOPRule"
        assert store.nx_graph.nodes["tool1"]["node_type"] == "Tool"

    def test_edge_types(self, minimal_kg):
        store = make_store(minimal_kg)
        G = store.nx_graph
        assert G.edges["sop1", "step1"]["edge_type"] == "HAS_STEP"
        assert G.edges["step1", "step2"]["edge_type"] == "NEXT_STEP"
        assert G.edges["rule1", "subrule1"]["edge_type"] == "HAS_SUB_RULE"
        assert G.edges["rule1", "tool1"]["edge_type"] == "USES_TOOL"

    def test_node_map_populated(self, minimal_kg):
        store = make_store(minimal_kg)
        for node in minimal_kg.nodes:
            assert node.id in store.node_map


# ---------------------------------------------------------------------------
# get_node
# ---------------------------------------------------------------------------

class TestGetNode:
    def test_existing_node(self, minimal_kg):
        store = make_store(minimal_kg)
        node = store.get_node("sop1")
        assert node is not None
        assert node["id"] == "sop1"
        assert node["node_type"] == "SOP"
        assert node["name"] == "催派送SOP"

    def test_missing_node_returns_none(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store.get_node("nonexistent") is None

    def test_tool_node_fields(self, minimal_kg):
        store = make_store(minimal_kg)
        tool = store.get_node("tool1")
        assert tool["name"] == "查路由"
        assert tool["tool_type"] == "QUERY"


# ---------------------------------------------------------------------------
# get_ancestors
# ---------------------------------------------------------------------------

class TestGetAncestors:
    def test_rule_ancestors(self, minimal_kg):
        """rule1's ancestors should be [sop1, step1] (root → parent)."""
        store = make_store(minimal_kg)
        ancestors = store.get_ancestors("rule1")
        ancestor_ids = [a["id"] for a in ancestors]
        assert "step1" in ancestor_ids
        assert "sop1" in ancestor_ids

    def test_subrule_ancestors(self, minimal_kg):
        """subrule1's ancestors: sop1, step1, rule1."""
        store = make_store(minimal_kg)
        ancestors = store.get_ancestors("subrule1")
        ancestor_ids = [a["id"] for a in ancestors]
        assert "rule1" in ancestor_ids
        assert "step1" in ancestor_ids
        assert "sop1" in ancestor_ids

    def test_root_node_has_no_ancestors(self, minimal_kg):
        store = make_store(minimal_kg)
        assert store.get_ancestors("sop1") == []

    def test_ancestors_root_first(self, minimal_kg):
        """Ancestors list should be ordered from root to direct parent."""
        store = make_store(minimal_kg)
        ancestors = store.get_ancestors("rule1")
        node_types = [a["node_type"] for a in ancestors]
        # SOP must appear before SOPStep
        assert node_types.index("SOP") < node_types.index("SOPStep")


# ---------------------------------------------------------------------------
# get_sop_context
# ---------------------------------------------------------------------------

class TestGetSopContext:
    def test_sop_node_present(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("sop1")
        assert ctx["sop"]["id"] == "sop1"

    def test_steps_sorted_by_index(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("sop1")
        steps = ctx["steps"]
        assert len(steps) == 2
        indices = [s["step"]["step_index"] for s in steps]
        assert indices == sorted(indices)

    def test_rules_sorted_by_index(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("sop1")
        step1_rules = ctx["steps"][0]["rules"]
        indices = [r["rule"]["rule_index"] for r in step1_rules]
        assert indices == sorted(indices)

    def test_subrules_present(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("sop1")
        rule1_ctx = ctx["steps"][0]["rules"][0]
        assert len(rule1_ctx["sub_rules"]) == 1
        assert rule1_ctx["sub_rules"][0]["id"] == "subrule1"

    def test_tools_present(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("sop1")
        rule1_ctx = ctx["steps"][0]["rules"][0]
        assert len(rule1_ctx["tools"]) == 1
        assert rule1_ctx["tools"][0]["id"] == "tool1"

    def test_nonexistent_sop_returns_empty(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_sop_context("no_such_sop")
        assert ctx == {}


# ---------------------------------------------------------------------------
# get_rule_context
# ---------------------------------------------------------------------------

class TestGetRuleContext:
    def test_node_present(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_rule_context("rule1", "SOPRule")
        assert ctx["node"]["id"] == "rule1"

    def test_ancestors_present(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_rule_context("rule1", "SOPRule")
        ancestor_ids = [a["id"] for a in ctx["ancestors"]]
        assert "step1" in ancestor_ids

    def test_siblings_present(self, minimal_kg):
        """rule1 and rule2 share step1; each should see the other as sibling."""
        store = make_store(minimal_kg)
        ctx = store.get_rule_context("rule1", "SOPRule")
        sibling_ids = [s["id"] for s in ctx["siblings"]]
        assert "rule2" in sibling_ids

    def test_children_are_subrules(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_rule_context("rule1", "SOPRule")
        child_ids = [c["id"] for c in ctx["children"]]
        assert "subrule1" in child_ids

    def test_nonexistent_node_returns_empty(self, minimal_kg):
        store = make_store(minimal_kg)
        ctx = store.get_rule_context("ghost", "SOPRule")
        assert ctx == {}


# ---------------------------------------------------------------------------
# similarity_search
# ---------------------------------------------------------------------------

class TestSimilaritySearch:
    def test_returns_list(self, minimal_kg):
        store = make_store(minimal_kg)
        results = store.similarity_search("配送超时", k=3)
        assert isinstance(results, list)

    def test_result_has_required_keys(self, minimal_kg):
        store = make_store(minimal_kg)
        results = store.similarity_search("包裹没到", k=2)
        for hit in results:
            assert "node_id" in hit
            assert "node_type" in hit
            assert "sop_id" in hit
            assert "score" in hit
            assert "text" in hit

    def test_k_limits_results(self, minimal_kg):
        store = make_store(minimal_kg)
        results = store.similarity_search("物流", k=2)
        assert len(results) <= 2

    def test_empty_store_returns_empty(self):
        empty_kg = KnowledgeGraph()
        store = make_store(empty_kg)
        results = store.similarity_search("查询", k=3)
        assert results == []


# ---------------------------------------------------------------------------
# save / load round-trip (filesystem)
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_roundtrip(self, minimal_kg, tmp_path):
        path = str(tmp_path / "test_graph.json")

        def fake_vs(docs, emb):
            return InMemoryVectorStore.from_documents(docs, emb) if docs else InMemoryVectorStore(emb)

        with patch("graph.storage._create_embeddings", return_value=FakeEmbeddings()), \
             patch("graph.storage._create_vector_store", side_effect=fake_vs):
            store = GraphStore.from_kg(minimal_kg)
            store.save(path)
            loaded = GraphStore.load(path)

        assert loaded.nx_graph.number_of_nodes() == store.nx_graph.number_of_nodes()
        assert loaded.nx_graph.number_of_edges() == store.nx_graph.number_of_edges()

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            GraphStore.load("/tmp/definitely_not_there_12345.json")
