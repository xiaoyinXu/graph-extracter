"""
Pydantic extraction models mirroring the customer_service.yaml LinkML schema.
Used as structured output targets for LLM-based knowledge graph extraction.

Enumerations are generated dynamically from the schema YAML so that adding
or renaming enum values in customer_service.yaml is automatically reflected
here without touching this file.
"""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from graph.schema_loader import get_enum_values

_SCHEMA_PATH = str(Path(__file__).resolve().parent.parent / "schema" / "customer_service.yaml")
SCHEMA_PATH: str = os.getenv("SCHEMA_PATH", _SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Enumerations — generated dynamically from schema/customer_service.yaml
# ---------------------------------------------------------------------------

def _make_str_enum(name: str, schema_path: str) -> type:
    """Build a str-Enum type from a LinkML enum defined in *schema_path*."""
    values = get_enum_values(schema_path, name)
    if not values:
        raise ValueError(f"Enum '{name}' not found or empty in {schema_path}")
    return Enum(name, {v: v for v in values}, type=str)  # type: ignore[return-value]


IssueType = _make_str_enum("IssueType", SCHEMA_PATH)
ToolType = _make_str_enum("ToolType", SCHEMA_PATH)


# ---------------------------------------------------------------------------
# Entity extraction models (nested tree mirroring SOP → Step → Rule → SubRule)
# ---------------------------------------------------------------------------

class ExtractedTool(BaseModel):
    """客服系统内部工具（如：工单新、拉群、查路由）"""
    id: str = Field(description="工具唯一ID，如 tool_ticket_001")
    name: str = Field(description="工具名称，如：工单新、拉群、查路由")
    tool_type: ToolType = Field(description="工具类型枚举值")
    description: Optional[str] = Field(default=None, description="工具功能说明及使用时机")


class ExtractedSOPSubRule(BaseModel):
    """规则内的追问分支（如果用户追问XXX）"""
    id: str = Field(description="子规则唯一ID，如 subrule_step1_rule1_001")
    condition: str = Field(description="触发此分支的追问条件（如：如果用户追问配送路线停滞）")
    execution_approach: str = Field(description="执行思路：告诉客服怎么做")
    reference_script: Optional[str] = Field(default=None, description="参考话术：告诉客服说什么")


class ExtractedSOPRule(BaseModel):
    """SOP步骤内的一条规则（条件 → 执行思路 → 参考话术）"""
    id: str = Field(description="规则唯一ID，如 rule_step1_001")
    rule_index: int = Field(ge=1, description="规则在步骤内的顺序编号，从1开始")
    condition: str = Field(description="触发此规则的用户意图描述")
    execution_approach: str = Field(description="执行思路：告诉客服应该怎么处理")
    reference_script: Optional[str] = Field(default=None, description="参考话术：告诉客服说什么，含变量占位符")
    sub_rules: list[ExtractedSOPSubRule] = Field(
        default_factory=list,
        description="该规则下的追问分支列表（如果用户追问...）"
    )
    used_tools: list[ExtractedTool] = Field(
        default_factory=list,
        description="执行此规则时需要调用的内部工具"
    )


class ExtractedSOPStep(BaseModel):
    """SOP中的一个升级步骤（第一步、第二步...）"""
    id: str = Field(description="步骤唯一ID，如 step_001")
    step_index: int = Field(ge=1, description="步骤顺序编号，从1开始")
    goal: str = Field(description="本步骤的处理目标（对应'目标：'字段）")
    acceptance_check: Optional[str] = Field(
        default=None,
        description="用户接受检验点：用户不认可时升级到下一步的条件描述"
    )
    rules: list[ExtractedSOPRule] = Field(
        default_factory=list,
        description="本步骤下的规则列表（有序）"
    )


class ExtractedSOP(BaseModel):
    """标准作业流程（SOP）完整提取结果"""
    id: str = Field(description="SOP唯一ID，如 sop_delivery_urge_001")
    name: str = Field(description="SOP名称，如：催派送SOP")
    issue_type: IssueType = Field(description="适用的问题类型（用于Agent意图路由）")
    sub_scenario: Optional[str] = Field(default=None, description="细分场景名称")
    trigger_samples: list[str] = Field(
        default_factory=list,
        description="触发此SOP的客户原声样本，用于向量检索时的意图匹配"
    )
    steps: list[ExtractedSOPStep] = Field(
        default_factory=list,
        description="SOP步骤列表（有序，step_index决定执行顺序）"
    )


class ExtractionOutput(BaseModel):
    """LLM结构化抽取的完整输出"""
    sops: list[ExtractedSOP] = Field(description="从文本中提取的SOP列表")


# ---------------------------------------------------------------------------
# Graph node / edge representations (used by storage layer)
# ---------------------------------------------------------------------------

class GraphNode(BaseModel):
    """知识图谱中的节点"""
    id: str
    node_type: str  # class name from schema, e.g. SOP | SOPStep | SOPRule | Tool
    data: dict      # serialized entity fields


class GraphEdge(BaseModel):
    """知识图谱中的有向边"""
    source: str
    target: str
    edge_type: str  # derived from schema relation, e.g. HAS_STEP | USES_TOOL | NEXT_STEP
    metadata: dict = Field(default_factory=dict)


class KnowledgeGraph(BaseModel):
    """可序列化的完整知识图谱"""
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)
    schema_path: Optional[str] = Field(
        default=None,
        description="LinkML schema YAML path used to build this graph (persisted for reload)"
    )
