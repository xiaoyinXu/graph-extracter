"""Shared fixtures and helpers for all test modules."""
from __future__ import annotations

import random
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

from graph.models import (
    ExtractionOutput,
    ExtractedSOP,
    ExtractedSOPStep,
    ExtractedSOPRule,
    ExtractedSOPSubRule,
    ExtractedTool,
    GraphEdge,
    GraphNode,
    IssueType,
    KnowledgeGraph,
    ToolType,
)


# ---------------------------------------------------------------------------
# Fake embeddings (deterministic, no API calls)
# ---------------------------------------------------------------------------

class FakeEmbeddings(Embeddings):
    """Returns reproducible 8-dim pseudo-random vectors based on text hash."""

    DIM = 8

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)

    @staticmethod
    def _vec(text: str) -> List[float]:
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        v = [rng.gauss(0, 1) for _ in range(FakeEmbeddings.DIM)]
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]


# ---------------------------------------------------------------------------
# Minimal KnowledgeGraph fixture (no LLM, no API)
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_kg() -> KnowledgeGraph:
    """
    One SOP → two Steps → Step-1 has two Rules → Rule-1 has one SubRule and one Tool.

    Graph shape:
      sop1 ──HAS_STEP──► step1 ──HAS_RULE──► rule1 ──HAS_SUB_RULE──► subrule1
                                                     └─USES_TOOL──► tool1
                       └─HAS_STEP──► step2
      step1 ──NEXT_STEP──► step2
    """
    nodes = [
        GraphNode(id="sop1", node_type="SOP", data={
            "name": "催派送SOP",
            "issue_type": "DELIVERY_URGE",
            "sub_scenario": "配送超时",
            "trigger_samples": ["我的包裹还没到", "配送怎么这么慢"],
        }),
        GraphNode(id="step1", node_type="SOPStep", data={
            "step_index": 1,
            "goal": "安抚用户情绪",
            "acceptance_check": "用户仍不接受",
            "sop_id": "sop1",
        }),
        GraphNode(id="step2", node_type="SOPStep", data={
            "step_index": 2,
            "goal": "查询物流状态并告知",
            "acceptance_check": None,
            "sop_id": "sop1",
        }),
        GraphNode(id="rule1", node_type="SOPRule", data={
            "rule_index": 1,
            "condition": "用户询问配送时间",
            "execution_approach": "查询路由后安抚",
            "reference_script": "您好，我这边帮您查询一下配送情况",
            "step_id": "step1",
            "sop_id": "sop1",
        }),
        GraphNode(id="rule2", node_type="SOPRule", data={
            "rule_index": 2,
            "condition": "用户要求赔偿",
            "execution_approach": "按政策说明",
            "reference_script": None,
            "step_id": "step1",
            "sop_id": "sop1",
        }),
        GraphNode(id="subrule1", node_type="SOPSubRule", data={
            "condition": "如果用户追问具体到达时间",
            "execution_approach": "承诺时效",
            "reference_script": "预计明天到达",
            "rule_id": "rule1",
            "step_id": "step1",
            "sop_id": "sop1",
        }),
        GraphNode(id="tool1", node_type="Tool", data={
            "name": "查路由",
            "tool_type": "QUERY",
            "description": "查询包裹路由信息",
        }),
    ]
    edges = [
        GraphEdge(source="sop1", target="step1", edge_type="HAS_STEP", metadata={"step_index": 1}),
        GraphEdge(source="sop1", target="step2", edge_type="HAS_STEP", metadata={"step_index": 2}),
        GraphEdge(source="step1", target="step2", edge_type="NEXT_STEP", metadata={"when": "REJECTED"}),
        GraphEdge(source="step1", target="rule1", edge_type="HAS_RULE", metadata={"rule_index": 1}),
        GraphEdge(source="step1", target="rule2", edge_type="HAS_RULE", metadata={"rule_index": 2}),
        GraphEdge(source="rule1", target="subrule1", edge_type="HAS_SUB_RULE", metadata={"index": 1}),
        GraphEdge(source="rule1", target="tool1", edge_type="USES_TOOL", metadata={}),
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


@pytest.fixture
def minimal_extraction_output() -> ExtractionOutput:
    """One complete SOP suitable for build_graph_node tests."""
    return ExtractionOutput(sops=[
        ExtractedSOP(
            id="sop1",
            name="退款SOP",
            issue_type=IssueType.REFUND,
            trigger_samples=["我要退款", "订单取消"],
            steps=[
                ExtractedSOPStep(
                    id="step1",
                    step_index=1,
                    goal="核实退款条件",
                    acceptance_check="用户不接受",
                    rules=[
                        ExtractedSOPRule(
                            id="rule1",
                            rule_index=1,
                            condition="用户申请退款",
                            execution_approach="按政策处理",
                            reference_script="好的，我来帮您处理",
                            used_tools=[
                                ExtractedTool(
                                    id="tool1",
                                    name="工单新",
                                    tool_type=ToolType.TICKET,
                                    description="创建退款工单",
                                )
                            ],
                            sub_rules=[
                                ExtractedSOPSubRule(
                                    id="subrule1",
                                    condition="如果用户追问进度",
                                    execution_approach="告知工单号",
                                    reference_script="工单号为{ticket_id}",
                                )
                            ],
                        ),
                        ExtractedSOPRule(
                            id="rule2",
                            rule_index=2,
                            condition="用户催促退款",
                            execution_approach="升级处理",
                        ),
                    ],
                ),
                ExtractedSOPStep(
                    id="step2",
                    step_index=2,
                    goal="升级退款投诉",
                ),
            ],
        )
    ])
