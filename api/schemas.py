"""Pydantic request / response schemas for the graph-extracter API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared
# ---------------------------------------------------------------------------

class NodeOut(BaseModel):
    id: str
    node_type: str
    data: dict[str, Any]


class EdgeOut(BaseModel):
    source: str
    target: str
    edge_type: str
    metadata: dict[str, Any]


class HitOut(BaseModel):
    node_id: str
    node_type: str
    root_id: str
    score: float
    text: str


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class HealthOut(BaseModel):
    status: str
    graph_path: str
    node_count: int
    edge_count: int


# ---------------------------------------------------------------------------
# /nodes
# ---------------------------------------------------------------------------

class NodeListOut(BaseModel):
    total: int
    nodes: list[NodeOut]


# ---------------------------------------------------------------------------
# /search  (vector search only)
# ---------------------------------------------------------------------------

class SearchIn(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言检索文本")
    k: int = Field(5, ge=1, le=50, description="返回命中数量")


class SearchOut(BaseModel):
    query: str
    hits: list[HitOut]


# ---------------------------------------------------------------------------
# /search/subgraph  (vector search + subgraph traversal)
# ---------------------------------------------------------------------------

class SubgraphSearchIn(BaseModel):
    query: str = Field(..., min_length=1, description="自然语言检索文本")
    k: int = Field(3, ge=1, le=20, description="向量检索 top-k")
    traverse_from_root: bool = Field(
        False, description="True=从根SOP节点开始遍历完整子图；False=从命中节点向下遍历"
    )


class SubgraphSearchOut(BaseModel):
    query: str
    hits: list[HitOut]
    start_node_ids: list[str]
    nodes: list[NodeOut]
    edges: list[EdgeOut]
    mermaid: str


# ---------------------------------------------------------------------------
# /build  (extract + build knowledge graph from doc + schema)
# ---------------------------------------------------------------------------

class BuildIn(BaseModel):
    schema_name: str = Field(
        ...,
        description="Schema 文件名（不含 .yaml 后缀），如 customer_service 或 technology",
        pattern=r"^[A-Za-z0-9_-]+$",
    )
    doc_name: str = Field(
        ...,
        description="文档文件名（含扩展名），如 流程数据.txt",
    )
    graph_name: Optional[str] = Field(
        default=None,
        description="输出图文件名（不含 .json），默认与 schema_name 相同",
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    )


class BuildOut(BaseModel):
    schema_name: str
    doc_name: str
    graph_path: str
    node_count: int
    edge_count: int
    node_type_summary: dict[str, int]


# ---------------------------------------------------------------------------
# /query  (full RAG pipeline)
# ---------------------------------------------------------------------------

class QueryIn(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class QueryOut(BaseModel):
    question: str
    answer: str
    hits: list[HitOut]
    context: str
