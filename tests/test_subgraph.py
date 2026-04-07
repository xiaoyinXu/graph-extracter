"""
Unit tests for GraphStore.get_connected_subgraph() and subgraph_to_mermaid().

Each test class focuses on a single concern.  Several tests also *print* the
rendered Mermaid diagram so it is visible during ``pytest -s`` runs for visual
inspection.
"""
from __future__ import annotations

import networkx as nx
import pytest
from unittest.mock import patch

from graph.models import KnowledgeGraph
from graph.storage import GraphStore
from graph.utils import subgraph_to_mermaid
from tests.conftest import FakeEmbeddings


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_store(kg: KnowledgeGraph) -> GraphStore:
    """Build a GraphStore without calling the real OpenAI embeddings API."""
    with patch("graph.storage._create_embeddings", return_value=FakeEmbeddings()):
        return GraphStore.from_kg(kg)


# ---------------------------------------------------------------------------
# get_connected_subgraph
# ---------------------------------------------------------------------------

class TestGetConnectedSubgraph:
    def test_from_root_includes_all_nodes(self, minimal_kg):
        """Starting from sop1 every node in the graph is reachable."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        assert sg.number_of_nodes() == len(minimal_kg.nodes)

    def test_from_root_includes_all_edges(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        assert sg.number_of_edges() == len(minimal_kg.edges)

    def test_from_step_excludes_parent_sop(self, minimal_kg):
        """Directed traversal: the SOP node is *upstream* of step1, not downstream."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("step1")
        assert "sop1" not in sg.nodes
        assert "step1" in sg.nodes

    def test_from_step1_includes_rules_subrules_tools(self, minimal_kg):
        """step1 reaches rule1, rule2, subrule1, tool1, and step2 via NEXT_STEP."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("step1")
        for expected in ("rule1", "rule2", "subrule1", "tool1", "step2"):
            assert expected in sg.nodes

    def test_from_rule_excludes_ancestors(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("rule1")
        assert "sop1" not in sg.nodes
        assert "step1" not in sg.nodes
        assert "rule1" in sg.nodes
        assert "subrule1" in sg.nodes
        assert "tool1" in sg.nodes

    def test_leaf_node_returns_single_node(self, minimal_kg):
        """tool1 has no outgoing edges → subgraph contains only itself."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("tool1")
        assert sg.number_of_nodes() == 1
        assert "tool1" in sg.nodes
        assert sg.number_of_edges() == 0

    def test_nonexistent_node_returns_empty_graph(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("no_such_node")
        assert sg.number_of_nodes() == 0

    def test_returns_independent_copy(self, minimal_kg):
        """Mutating the returned subgraph must not affect the original graph."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        sg.remove_node("step1")
        assert "step1" in store.nx_graph


# ---------------------------------------------------------------------------
# subgraph_to_mermaid – structure
# ---------------------------------------------------------------------------

class TestSubgraphToMermaid:
    def test_starts_with_graph_td(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        assert mermaid.startswith("graph TD")

    def test_all_node_ids_present(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        for node in minimal_kg.nodes:
            assert node.id in mermaid

    def test_edge_types_present(self, minimal_kg):
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        for edge_type in ("HAS_STEP", "HAS_RULE", "HAS_SUB_RULE", "USES_TOOL", "NEXT_STEP"):
            assert edge_type in mermaid

    def test_sop_label_contains_name(self, minimal_kg):
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        assert "催派送SOP" in mermaid

    def test_step_label_contains_goal(self, minimal_kg):
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        assert "安抚用户情绪" in mermaid

    def test_rule_label_contains_condition(self, minimal_kg):
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        assert "用户询问配送时间" in mermaid

    def test_tool_label_contains_name(self, minimal_kg):
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        assert "查路由" in mermaid

    def test_partial_subgraph_excludes_upstream_nodes(self, minimal_kg):
        """Mermaid for rule1 subgraph must not mention sop1 or step1."""
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("rule1"), store.node_map
        )
        assert "sop1" not in mermaid
        assert "step1" not in mermaid
        assert "subrule1" in mermaid
        assert "tool1" in mermaid

    def test_empty_subgraph_returns_graph_td_only(self):
        mermaid = subgraph_to_mermaid(nx.DiGraph(), {})
        assert mermaid == "graph TD"

    def test_node_bracket_syntax(self, minimal_kg):
        """Every node must be declared with Mermaid bracket notation."""
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        # Each node id should appear in a line like:  sop1["..."]
        for node in minimal_kg.nodes:
            assert f'{node.id}["' in mermaid

    def test_arrow_syntax(self, minimal_kg):
        """Edges must use Mermaid arrow syntax with edge-type label."""
        store = make_store(minimal_kg)
        mermaid = subgraph_to_mermaid(
            store.get_connected_subgraph("sop1"), store.node_map
        )
        assert "-->|" in mermaid


# ---------------------------------------------------------------------------
# Mermaid output – visual / integration snapshots (printed for -s inspection)
# ---------------------------------------------------------------------------

class TestMermaidOutput:
    def test_full_sop_mermaid_output(self, minimal_kg, capsys):
        """Print the full SOP subgraph as Mermaid for visual inspection."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("sop1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        print(f"\n--- Mermaid: full SOP subgraph (root=sop1) ---\n{mermaid}\n")
        captured = capsys.readouterr()
        assert "graph TD" in captured.out
        assert "sop1" in captured.out

    def test_step_subgraph_mermaid_output(self, minimal_kg, capsys):
        """Print the step1 subgraph (excludes SOP node) as Mermaid."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("step1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        print(f"\n--- Mermaid: step1 subgraph ---\n{mermaid}\n")
        captured = capsys.readouterr()
        assert "step1" in captured.out
        assert "sop1" not in captured.out

    def test_rule_subgraph_mermaid_output(self, minimal_kg, capsys):
        """Print the rule1 subgraph (subrules + tools only) as Mermaid."""
        store = make_store(minimal_kg)
        sg = store.get_connected_subgraph("rule1")
        mermaid = subgraph_to_mermaid(sg, store.node_map)
        print(f"\n--- Mermaid: rule1 subgraph ---\n{mermaid}\n")
        captured = capsys.readouterr()
        assert "rule1" in captured.out
