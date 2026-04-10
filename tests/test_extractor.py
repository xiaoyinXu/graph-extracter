"""
Unit tests for graph/extractor.py — build_graph_node, should_retry,
validate_graph_node, should_retry_after_validation.

LLM-dependent nodes (extract_sops_node) are tested via mocking;
pure logic nodes (build_graph_node, should_retry, validate_graph_node)
are tested directly.
"""
from __future__ import annotations

import pytest

from graph.extractor import (
    build_graph_node,
    should_retry,
    should_retry_after_validation,
    validate_graph_node,
)
from graph.models import ExtractionOutput, GraphEdge, GraphNode, KnowledgeGraph


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------

class TestShouldRetry:
    def test_no_output_low_retry_count_retries(self):
        state = {"extracted_output": None, "retry_count": 0, "errors": []}
        assert should_retry(state) == "extract_entities"

    def test_no_output_retry_count_2_still_retries(self):
        state = {"extracted_output": None, "retry_count": 2, "errors": []}
        assert should_retry(state) == "extract_entities"

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


# ---------------------------------------------------------------------------
# Helpers for validation tests
# ---------------------------------------------------------------------------

def _base_state(kg: KnowledgeGraph, retry_count: int = 0) -> dict:
    return {
        "raw_text": "", "extracted_output": None,
        "graph": kg, "errors": [], "retry_count": retry_count,
        "validation_issues": [],
    }


def _make_sop_node(**overrides) -> GraphNode:
    data = {
        "name": "催派送SOP",
        "issue_type": "DELIVERY_URGE",
        "sub_scenario": "未到站-常规催派送",
        "trigger_samples": ["怎么还没到"],
    }
    data.update(overrides)
    return GraphNode(id="sop1", node_type="SOP", data=data)


def _make_step_node(**overrides) -> GraphNode:
    data = {"step_index": 1, "goal": "安抚用户", "sop_id": "sop1"}
    data.update(overrides)
    return GraphNode(id="step1", node_type="SOPStep", data=data)


def _make_rule_node(**overrides) -> GraphNode:
    data = {
        "rule_index": 1,
        "condition": "用户询问时间",
        "execution_approach": "查询路由后告知",
        "sop_id": "sop1",
    }
    data.update(overrides)
    return GraphNode(id="rule1", node_type="SOPRule", data=data)


def _make_tool_node(**overrides) -> GraphNode:
    data = {"name": "查路由", "tool_type": "QUERY"}
    data.update(overrides)
    return GraphNode(id="tool1", node_type="Tool", data=data)


def _issues(result: dict, severity: str | None = None) -> list:
    issues = result["validation_issues"]
    if severity:
        return [i for i in issues if i["severity"] == severity]
    return issues


# ---------------------------------------------------------------------------
# validate_graph_node — basic structure
# ---------------------------------------------------------------------------

class TestValidateGraphNodeBasic:
    def test_returns_validation_issues_key(self):
        kg = KnowledgeGraph(nodes=[_make_sop_node()], edges=[])
        result = validate_graph_node(_base_state(kg))
        assert "validation_issues" in result

    def test_valid_kg_has_no_errors(self, minimal_kg):
        result = validate_graph_node(_base_state(minimal_kg))
        assert _issues(result, "ERROR") == []

    def test_none_graph_adds_error_message(self):
        state = _base_state(KnowledgeGraph())
        state["graph"] = None
        result = validate_graph_node(state)
        assert any("no graph" in e for e in result["errors"])

    def test_empty_kg_passes(self):
        result = validate_graph_node(_base_state(KnowledgeGraph()))
        assert _issues(result, "ERROR") == []


# ---------------------------------------------------------------------------
# validate_graph_node — required field checks
# ---------------------------------------------------------------------------

class TestValidateRequiredFields:
    def test_missing_sop_name_is_error(self):
        node = _make_sop_node(name=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "name" and i["node_id"] == "sop1" for i in errors)

    def test_empty_sop_name_is_error(self):
        node = _make_sop_node(name="   ")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "name" for i in errors)

    def test_missing_step_goal_is_error(self):
        node = _make_step_node(goal=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "goal" for i in errors)

    def test_missing_rule_condition_is_error(self):
        node = _make_rule_node(condition=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "condition" for i in errors)

    def test_missing_rule_execution_approach_is_error(self):
        node = _make_rule_node(execution_approach=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "execution_approach" for i in errors)

    def test_missing_tool_name_is_error(self):
        node = _make_tool_node(name=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "name" for i in errors)

    def test_optional_field_none_is_not_error(self):
        """reference_script is optional — None value should not be an error."""
        node = _make_rule_node(reference_script=None)
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert not any(i["field"] == "reference_script" for i in errors)


# ---------------------------------------------------------------------------
# validate_graph_node — enum checks
# ---------------------------------------------------------------------------

class TestValidateEnumFields:
    def test_valid_issue_type_passes(self):
        node = _make_sop_node(issue_type="DELIVERY_URGE")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        assert not any(i["field"] == "issue_type" for i in _issues(result, "ERROR"))

    def test_invalid_issue_type_is_error(self):
        node = _make_sop_node(issue_type="INVALID_TYPE")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "issue_type" for i in errors)

    def test_invalid_tool_type_is_error(self):
        node = _make_tool_node(tool_type="MAGIC")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "tool_type" for i in errors)

    def test_valid_tool_type_passes(self):
        node = _make_tool_node(tool_type="QUERY")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        assert not any(i["field"] == "tool_type" for i in _issues(result, "ERROR"))


# ---------------------------------------------------------------------------
# validate_graph_node — structural checks
# ---------------------------------------------------------------------------

class TestValidateStructural:
    def test_duplicate_node_ids_is_error(self):
        nodes = [_make_sop_node(), _make_sop_node()]  # same id "sop1"
        kg = KnowledgeGraph(nodes=nodes, edges=[])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "id" and "uplicate" in i["message"] for i in errors)

    def test_edge_missing_source_is_error(self):
        node = _make_sop_node()
        edge = GraphEdge(source="ghost", target="sop1", edge_type="HAS_STEP", metadata={})
        kg = KnowledgeGraph(nodes=[node], edges=[edge])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "source" for i in errors)

    def test_edge_missing_target_is_error(self):
        node = _make_sop_node()
        edge = GraphEdge(source="sop1", target="ghost", edge_type="HAS_STEP", metadata={})
        kg = KnowledgeGraph(nodes=[node], edges=[edge])
        result = validate_graph_node(_base_state(kg))
        errors = _issues(result, "ERROR")
        assert any(i["field"] == "target" for i in errors)

    def test_valid_edge_passes(self):
        sop = _make_sop_node()
        step = _make_step_node()
        edge = GraphEdge(source="sop1", target="step1", edge_type="HAS_STEP", metadata={})
        kg = KnowledgeGraph(nodes=[sop, step], edges=[edge])
        result = validate_graph_node(_base_state(kg))
        assert not any(i["field"] in ("source", "target") for i in _issues(result, "ERROR"))


# ---------------------------------------------------------------------------
# validate_graph_node — optional field warnings
# ---------------------------------------------------------------------------

class TestValidateWarnings:
    def test_optional_empty_string_is_warning(self):
        """acceptance_check is optional; empty string → WARNING."""
        node = _make_step_node(acceptance_check="")
        kg = KnowledgeGraph(nodes=[node], edges=[])
        result = validate_graph_node(_base_state(kg))
        warnings = _issues(result, "WARNING")
        assert any(i["field"] == "acceptance_check" for i in warnings)


# ---------------------------------------------------------------------------
# should_retry_after_validation
# ---------------------------------------------------------------------------

class TestShouldRetryAfterValidation:
    def _state(self, issues: list, retry_count: int = 0) -> dict:
        return {"validation_issues": issues, "retry_count": retry_count, "errors": []}

    def test_no_issues_goes_to_save(self):
        assert should_retry_after_validation(self._state([])) == "save_graph"

    def test_warnings_only_goes_to_save(self):
        issues = [{"severity": "WARNING", "field": "x", "node_id": "n", "node_type": "SOP", "message": "w"}]
        assert should_retry_after_validation(self._state(issues)) == "save_graph"

    def test_error_low_retry_retries(self):
        issues = [{"severity": "ERROR", "field": "name", "node_id": "n", "node_type": "SOP", "message": "e"}]
        assert should_retry_after_validation(self._state(issues, retry_count=0)) == "extract_entities"

    def test_error_retry_count_2_still_retries(self):
        issues = [{"severity": "ERROR", "field": "name", "node_id": "n", "node_type": "SOP", "message": "e"}]
        assert should_retry_after_validation(self._state(issues, retry_count=2)) == "extract_entities"

    def test_error_retry_count_3_goes_to_save(self):
        issues = [{"severity": "ERROR", "field": "name", "node_id": "n", "node_type": "SOP", "message": "e"}]
        assert should_retry_after_validation(self._state(issues, retry_count=3)) == "save_graph"
