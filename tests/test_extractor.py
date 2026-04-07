"""
Unit tests for graph/extractor.py — build_graph_node, should_retry.

LLM-dependent nodes (extract_sops_node) are tested via mocking;
pure logic nodes (build_graph_node, should_retry) are tested directly.
"""
from __future__ import annotations

import pytest

from graph.extractor import build_graph_node, should_retry
from graph.models import ExtractionOutput, KnowledgeGraph


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------

class TestShouldRetry:
    def test_no_output_low_retry_count_retries(self):
        state = {"extracted_output": None, "retry_count": 0, "errors": []}
        assert should_retry(state) == "extract_sops"

    def test_no_output_retry_count_2_still_retries(self):
        state = {"extracted_output": None, "retry_count": 2, "errors": []}
        assert should_retry(state) == "extract_sops"

    def test_no_output_retry_count_3_stops(self):
        state = {"extracted_output": None, "retry_count": 3, "errors": []}
        assert should_retry(state) == "build_graph"

    def test_no_output_retry_count_above_3_stops(self):
        state = {"extracted_output": None, "retry_count": 99, "errors": []}
        assert should_retry(state) == "build_graph"

    def test_with_output_always_proceeds(self, minimal_extraction_output):
        state = {
            "extracted_output": minimal_extraction_output,
            "retry_count": 0,
            "errors": [],
        }
        assert should_retry(state) == "build_graph"

    def test_with_output_high_retry_count_still_proceeds(self, minimal_extraction_output):
        state = {
            "extracted_output": minimal_extraction_output,
            "retry_count": 10,
            "errors": [],
        }
        assert should_retry(state) == "build_graph"


# ---------------------------------------------------------------------------
# build_graph_node — no extracted_output
# ---------------------------------------------------------------------------

class TestBuildGraphNodeEmpty:
    def test_no_output_returns_empty_graph(self):
        state = {"extracted_output": None, "errors": []}
        result = build_graph_node(state)
        assert isinstance(result["graph"], KnowledgeGraph)
        assert result["graph"].nodes == []
        assert result["graph"].edges == []

    def test_no_output_appends_error(self):
        state = {"extracted_output": None, "errors": []}
        result = build_graph_node(state)
        assert any("no extracted_output" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# build_graph_node — happy path
# ---------------------------------------------------------------------------

class TestBuildGraphNode:
    @pytest.fixture
    def result(self, minimal_extraction_output):
        state = {
            "extracted_output": minimal_extraction_output,
            "errors": [],
        }
        return build_graph_node(state)

    def test_no_errors(self, result):
        assert result["errors"] == []

    def test_returns_knowledge_graph(self, result):
        assert isinstance(result["graph"], KnowledgeGraph)

    def test_sop_node_present(self, result):
        node_ids = {n.id for n in result["graph"].nodes}
        assert "sop1" in node_ids

    def test_step_nodes_present(self, result):
        node_ids = {n.id for n in result["graph"].nodes}
        assert "step1" in node_ids
        assert "step2" in node_ids

    def test_rule_nodes_present(self, result):
        node_ids = {n.id for n in result["graph"].nodes}
        assert "rule1" in node_ids
        assert "rule2" in node_ids

    def test_subrule_node_present(self, result):
        node_ids = {n.id for n in result["graph"].nodes}
        assert "subrule1" in node_ids

    def test_tool_node_present(self, result):
        node_types = {n.node_type for n in result["graph"].nodes}
        assert "Tool" in node_types

    def test_has_step_edges(self, result):
        edge_types = {e.edge_type for e in result["graph"].edges}
        assert "HAS_STEP" in edge_types

    def test_has_rule_edges(self, result):
        edge_types = {e.edge_type for e in result["graph"].edges}
        assert "HAS_RULE" in edge_types

    def test_has_sub_rule_edges(self, result):
        edge_types = {e.edge_type for e in result["graph"].edges}
        assert "HAS_SUB_RULE" in edge_types

    def test_uses_tool_edges(self, result):
        edge_types = {e.edge_type for e in result["graph"].edges}
        assert "USES_TOOL" in edge_types

    def test_next_step_edge_between_steps(self, result):
        next_step_edges = [
            e for e in result["graph"].edges if e.edge_type == "NEXT_STEP"
        ]
        assert len(next_step_edges) == 1
        # step1 → step2
        assert next_step_edges[0].source == "step1"
        assert next_step_edges[0].target == "step2"

    def test_next_step_edge_has_rejected_metadata(self, result):
        next_step_edge = next(
            e for e in result["graph"].edges if e.edge_type == "NEXT_STEP"
        )
        assert next_step_edge.metadata.get("when") == "REJECTED"


# ---------------------------------------------------------------------------
# build_graph_node — tool deduplication
# ---------------------------------------------------------------------------

class TestToolDeduplication:
    def test_same_tool_name_creates_one_node(self):
        """Two rules using a tool with the same name → only one Tool node."""
        from graph.models import (
            ExtractedSOP, ExtractedSOPStep, ExtractedSOPRule, ExtractedTool,
            IssueType, ToolType,
        )
        shared_tool = ExtractedTool(
            id="tool_dup",
            name="工单新",
            tool_type=ToolType.TICKET,
        )
        output = ExtractionOutput(sops=[
            ExtractedSOP(
                id="sop1", name="Test SOP", issue_type=IssueType.REFUND,
                steps=[
                    ExtractedSOPStep(
                        id="step1", step_index=1, goal="g1",
                        rules=[
                            ExtractedSOPRule(
                                id="rule1", rule_index=1,
                                condition="c1", execution_approach="e1",
                                used_tools=[shared_tool],
                            ),
                            ExtractedSOPRule(
                                id="rule2", rule_index=2,
                                condition="c2", execution_approach="e2",
                                used_tools=[shared_tool],
                            ),
                        ],
                    )
                ],
            )
        ])
        result = build_graph_node({"extracted_output": output, "errors": []})
        tool_nodes = [n for n in result["graph"].nodes if n.node_type == "Tool"]
        assert len(tool_nodes) == 1  # deduplicated by name


# ---------------------------------------------------------------------------
# build_graph_node — preserves errors from prior state
# ---------------------------------------------------------------------------

class TestBuildGraphNodePreservesErrors:
    def test_prior_errors_are_preserved(self, minimal_extraction_output):
        state = {
            "extracted_output": minimal_extraction_output,
            "errors": ["prior_error"],
        }
        result = build_graph_node(state)
        assert "prior_error" in result["errors"]
