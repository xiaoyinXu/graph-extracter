"""
Unit tests for graph/utils.py — print_graph_topology.
"""
from __future__ import annotations

import pytest
from io import StringIO
from typing import Any
from unittest.mock import patch

from graph.extractor import build_extraction_graph
from graph.utils import print_graph_topology


def capture_topology(compiled_graph: Any, name: str) -> str:
    """Helper: return print_graph_topology output as a string."""
    buf = StringIO()
    with patch("builtins.print", side_effect=lambda *args, **kwargs: buf.write(" ".join(str(a) for a in args) + "\n")):
        print_graph_topology(compiled_graph, name)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Extraction graph topology
# ---------------------------------------------------------------------------

class TestExtractionGraphTopology:
    @pytest.fixture(scope="class")
    def output(self):
        graph = build_extraction_graph()
        return capture_topology(graph, "Extraction Pipeline")

    def test_header_contains_name(self, output):
        assert "Extraction Pipeline" in output

    def test_all_user_nodes_listed(self, output):
        for node in ("extract_sops", "build_graph", "save_graph"):
            assert node in output

    def test_start_and_end_listed(self, output):
        assert "__start__" in output
        assert "__end__" in output

    def test_fixed_edge_start_to_extract(self, output):
        assert "__start__" in output
        assert "extract_sops" in output

    def test_build_to_save_edge(self, output):
        assert "build_graph" in output
        assert "save_graph" in output

    def test_conditional_edge_shown(self, output):
        assert "conditional" in output
        assert "should_retry" in output

    def test_fixed_label_present(self, output):
        assert "[fixed]" in output


# ---------------------------------------------------------------------------
# Retrieval graph topology (linear, no conditional edges)
# ---------------------------------------------------------------------------

class TestRetrievalGraphTopology:
    @pytest.fixture(scope="class")
    def output(self):
        from unittest.mock import MagicMock
        from graph.retriever import build_retrieval_graph
        from graph.storage import GraphStore

        store = MagicMock(spec=GraphStore)
        store.nx_graph = MagicMock()
        store.nx_graph.number_of_nodes.return_value = 0
        store.nx_graph.number_of_edges.return_value = 0

        rg = build_retrieval_graph(store)
        return capture_topology(rg, "Retrieval Pipeline")

    def test_header_contains_name(self, output):
        assert "Retrieval Pipeline" in output

    def test_all_retrieval_nodes_listed(self, output):
        for node in ("search_nodes", "expand_context", "generate_answer"):
            assert node in output

    def test_no_conditional_edges(self, output):
        assert "conditional" not in output

    def test_node_count_in_header(self, output):
        # 5 nodes: __start__ + 3 user + __end__
        assert "Nodes (5)" in output


# ---------------------------------------------------------------------------
# Edge-case: graph with no conditional edges
# ---------------------------------------------------------------------------

class TestEmptyConditionalBranches:
    def test_no_conditional_section_when_all_fixed(self):
        from unittest.mock import MagicMock
        from graph.retriever import build_retrieval_graph
        from graph.storage import GraphStore

        store = MagicMock(spec=GraphStore)
        store.nx_graph = MagicMock()
        store.nx_graph.number_of_nodes.return_value = 0
        store.nx_graph.number_of_edges.return_value = 0

        rg = build_retrieval_graph(store)
        output = capture_topology(rg, "Test")
        # All edges are [fixed]; no "(dynamic)" should appear
        assert "(dynamic)" not in output
