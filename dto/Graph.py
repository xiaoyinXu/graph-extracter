"""
DTO layer for knowledge graph.

``NodeDTO`` / ``EdgeDTO`` / ``KnowledgeGraphDTO`` are generic and schema-agnostic.
The legacy SOP-specific classes (SOPDTO, SOPStepDTO, …) are kept for
backward compatibility but should not be used in new code.
"""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Generic (schema-agnostic)
# ---------------------------------------------------------------------------

class NodeDTO(BaseModel):
    """A single knowledge-graph node with typed metadata."""
    id: str
    node_type: str
    data: dict[str, Any] = Field(default_factory=dict)


class EdgeDTO(BaseModel):
    """A directed edge in the knowledge graph."""
    source: str
    target: str
    edge_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphDTO(BaseModel):
    """Schema-agnostic flat representation of the full knowledge graph."""
    nodes: list[NodeDTO] = Field(default_factory=list)
    edges: list[EdgeDTO] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Legacy backward-compatible classes (kept for existing callers)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    pass


class Relationship(BaseModel):
    pass


class Graph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


class ToolDTO(BaseModel):
    id: str
    name: str
    tool_type: str
    description: Optional[str] = None
    mcp_endpoint: Optional[str] = None


class SOPSubRuleDTO(BaseModel):
    id: str
    condition: str
    execution_approach: str
    reference_script: Optional[str] = None


class SOPRuleDTO(BaseModel):
    id: str
    rule_index: int
    condition: str
    execution_approach: str
    reference_script: Optional[str] = None
    sub_rule_ids: list[str] = []
    tool_ids: list[str] = []


class SOPStepDTO(BaseModel):
    id: str
    step_index: int
    goal: str
    acceptance_check: Optional[str] = None
    rule_ids: list[str] = []
    next_step_id: Optional[str] = None


class SOPDTO(BaseModel):
    id: str
    name: str
    issue_type: str
    sub_scenario: Optional[str] = None
    trigger_samples: list[str] = []
    step_ids: list[str] = []
    quality_score: Optional[float] = None
