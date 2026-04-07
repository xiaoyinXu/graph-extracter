"""
Unit tests for graph/models.py — Pydantic extraction models.
"""
import pytest
from pydantic import ValidationError

from graph.models import (
    ExtractionOutput,
    ExtractedSOP,
    ExtractedSOPRule,
    ExtractedSOPStep,
    ExtractedSOPSubRule,
    ExtractedTool,
    GraphEdge,
    GraphNode,
    IssueType,
    KnowledgeGraph,
    ToolType,
)


# ---------------------------------------------------------------------------
# Enum tests
# ---------------------------------------------------------------------------

class TestEnums:
    def test_issue_type_values(self):
        assert IssueType.DELIVERY_URGE == "DELIVERY_URGE"
        assert IssueType.REFUND == "REFUND"

    def test_tool_type_values(self):
        assert ToolType.TICKET == "TICKET"
        assert ToolType.QUERY == "QUERY"

    def test_issue_type_invalid(self):
        with pytest.raises(ValidationError):
            ExtractedSOP(
                id="s1", name="x",
                issue_type="INVALID_TYPE",  # type: ignore[arg-type]
            )


# ---------------------------------------------------------------------------
# ExtractedTool
# ---------------------------------------------------------------------------

class TestExtractedTool:
    def test_valid_tool(self):
        t = ExtractedTool(id="t1", name="工单新", tool_type=ToolType.TICKET)
        assert t.id == "t1"
        assert t.description is None

    def test_tool_with_description(self):
        t = ExtractedTool(id="t1", name="拉群", tool_type=ToolType.GROUP_CHAT,
                          description="创建群聊")
        assert t.description == "创建群聊"


# ---------------------------------------------------------------------------
# ExtractedSOPSubRule
# ---------------------------------------------------------------------------

class TestExtractedSOPSubRule:
    def test_valid_subrule(self):
        sr = ExtractedSOPSubRule(
            id="sr1",
            condition="如果追问",
            execution_approach="按政策回复",
        )
        assert sr.reference_script is None

    def test_subrule_missing_required(self):
        with pytest.raises(ValidationError):
            ExtractedSOPSubRule(id="sr1")  # missing condition, execution_approach


# ---------------------------------------------------------------------------
# ExtractedSOPRule
# ---------------------------------------------------------------------------

class TestExtractedSOPRule:
    def test_rule_index_must_be_ge_1(self):
        with pytest.raises(ValidationError):
            ExtractedSOPRule(
                id="r1", rule_index=0,   # ge=1 violated
                condition="c", execution_approach="e",
            )

    def test_defaults_empty_lists(self):
        r = ExtractedSOPRule(
            id="r1", rule_index=1,
            condition="c", execution_approach="e",
        )
        assert r.sub_rules == []
        assert r.used_tools == []


# ---------------------------------------------------------------------------
# ExtractedSOPStep
# ---------------------------------------------------------------------------

class TestExtractedSOPStep:
    def test_step_index_must_be_ge_1(self):
        with pytest.raises(ValidationError):
            ExtractedSOPStep(id="s1", step_index=0, goal="g")

    def test_optional_acceptance_check(self):
        s = ExtractedSOPStep(id="s1", step_index=1, goal="目标")
        assert s.acceptance_check is None
        assert s.rules == []


# ---------------------------------------------------------------------------
# ExtractionOutput
# ---------------------------------------------------------------------------

class TestExtractionOutput:
    def test_empty_sops(self):
        out = ExtractionOutput(sops=[])
        assert out.sops == []

    def test_roundtrip_json(self, minimal_extraction_output):
        json_str = minimal_extraction_output.model_dump_json()
        restored = ExtractionOutput.model_validate_json(json_str)
        assert len(restored.sops) == len(minimal_extraction_output.sops)
        assert restored.sops[0].id == minimal_extraction_output.sops[0].id
        assert restored.sops[0].issue_type == IssueType.REFUND

    def test_validate_from_dict(self):
        data = {
            "sops": [{
                "id": "sop1",
                "name": "催派送SOP",
                "issue_type": "DELIVERY_URGE",
                "trigger_samples": ["包裹没到"],
                "steps": [],
            }]
        }
        out = ExtractionOutput.model_validate(data)
        assert out.sops[0].issue_type == IssueType.DELIVERY_URGE


# ---------------------------------------------------------------------------
# GraphNode / GraphEdge / KnowledgeGraph
# ---------------------------------------------------------------------------

class TestGraphModels:
    def test_graph_node_creation(self):
        n = GraphNode(id="n1", node_type="SOP", data={"name": "test"})
        assert n.id == "n1"
        assert n.data["name"] == "test"

    def test_graph_edge_defaults(self):
        e = GraphEdge(source="a", target="b", edge_type="HAS_STEP")
        assert e.metadata == {}

    def test_knowledge_graph_empty_defaults(self):
        kg = KnowledgeGraph()
        assert kg.nodes == []
        assert kg.edges == []

    def test_knowledge_graph_roundtrip(self, minimal_kg):
        json_str = minimal_kg.model_dump_json()
        restored = KnowledgeGraph.model_validate_json(json_str)
        assert len(restored.nodes) == len(minimal_kg.nodes)
        assert len(restored.edges) == len(minimal_kg.edges)

    def test_node_ids_preserved(self, minimal_kg):
        ids = {n.id for n in minimal_kg.nodes}
        assert "sop1" in ids
        assert "step1" in ids
        assert "tool1" in ids

    def test_edge_types_preserved(self, minimal_kg):
        edge_types = {e.edge_type for e in minimal_kg.edges}
        assert "HAS_STEP" in edge_types
        assert "NEXT_STEP" in edge_types
        assert "HAS_RULE" in edge_types
        assert "USES_TOOL" in edge_types
