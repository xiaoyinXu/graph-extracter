"""
DTO layer for knowledge graph. Typed entities mirror the customer_service.yaml schema.
The generic Entity/Relationship/Graph are kept for backward compatibility.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Generic (backward-compatible)
# ---------------------------------------------------------------------------

class Entity(BaseModel):
    pass


class Relationship(BaseModel):
    pass


class Graph(BaseModel):
    entities: list[Entity]
    relationships: list[Relationship]


# ---------------------------------------------------------------------------
# Typed entities (mirror customer_service.yaml LinkML schema)
# ---------------------------------------------------------------------------

class ToolDTO(BaseModel):
    id: str
    name: str
    tool_type: str  # ToolType enum value
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
    issue_type: str         # IssueType enum value
    sub_scenario: Optional[str] = None
    trigger_samples: list[str] = []
    step_ids: list[str] = []
    quality_score: Optional[float] = None


class KnowledgeGraphDTO(BaseModel):
    """Flat DTO representation of the full knowledge graph."""
    sops: list[SOPDTO] = []
    steps: list[SOPStepDTO] = []
    rules: list[SOPRuleDTO] = []
    sub_rules: list[SOPSubRuleDTO] = []
    tools: list[ToolDTO] = []
