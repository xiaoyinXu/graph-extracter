"""
FastAPI application for graph-extracter.

Endpoints:
  GET  /health                     — 服务健康检查 + 图规模
  GET  /nodes                      — 列出所有节点（可按 node_type 过滤）
  GET  /nodes/{node_id}            — 获取单个节点
  GET  /nodes/{node_id}/subgraph   — 获取节点的联通子图（含 mermaid）
  POST /search                     — 向量检索（返回命中节点列表）
  POST /search/subgraph            — 向量检索 + 子图遍历（返回 mermaid）
  POST /query                      — 完整 RAG 问答（向量检索 → 上下文扩展 → LLM 回答）
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Query

from api.schemas import (
    EdgeOut,
    HealthOut,
    HitOut,
    NodeListOut,
    NodeOut,
    QueryIn,
    QueryOut,
    SearchIn,
    SearchOut,
    SubgraphSearchIn,
    SubgraphSearchOut,
)
from graph.retriever import KnowledgeGraphRetriever
from graph.storage import DEFAULT_GRAPH_PATH

GRAPH_PATH = os.getenv("GRAPH_PATH", DEFAULT_GRAPH_PATH)

# ---------------------------------------------------------------------------
# App-lifetime state
# ---------------------------------------------------------------------------

_retriever: Optional[KnowledgeGraphRetriever] = None


def _get_retriever() -> KnowledgeGraphRetriever:
    if _retriever is None:
        raise HTTPException(status_code=503, detail="Graph not loaded yet")
    return _retriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _retriever
    print(f"[api] Loading graph from {GRAPH_PATH}...")
    _retriever = KnowledgeGraphRetriever(GRAPH_PATH)
    print("[api] Ready.")
    yield
    _retriever = None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Graph Extracter API",
    description="SOP 知识图谱检索与问答接口",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthOut, summary="服务健康检查")
def health() -> HealthOut:
    r = _get_retriever()
    return HealthOut(
        status="ok",
        graph_path=GRAPH_PATH,
        node_count=r.store.nx_graph.number_of_nodes(),
        edge_count=r.store.nx_graph.number_of_edges(),
    )


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

@app.get("/nodes", response_model=NodeListOut, summary="列出所有节点")
def list_nodes(
    node_type: Optional[str] = Query(None, description="按 node_type 过滤，如 SOP / SOPRule"),
) -> NodeListOut:
    r = _get_retriever()
    nodes = list(r.store.node_map.values())
    if node_type:
        nodes = [n for n in nodes if n.node_type == node_type]
    return NodeListOut(
        total=len(nodes),
        nodes=[NodeOut(id=n.id, node_type=n.node_type, data=n.data) for n in nodes],
    )


@app.get("/nodes/{node_id}", response_model=NodeOut, summary="获取单个节点")
def get_node(node_id: str) -> NodeOut:
    r = _get_retriever()
    node = r.store.node_map.get(node_id)
    if not node:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    return NodeOut(id=node.id, node_type=node.node_type, data=node.data)


@app.get(
    "/nodes/{node_id}/subgraph",
    response_model=SubgraphSearchOut,
    summary="获取节点联通子图（含 mermaid）",
)
def get_node_subgraph(node_id: str) -> SubgraphSearchOut:
    r = _get_retriever()
    if node_id not in r.store.nx_graph:
        raise HTTPException(status_code=404, detail=f"Node '{node_id}' not found")
    subgraph = r.store.get_connected_subgraph(node_id)
    nodes = [
        NodeOut(
            id=nid,
            node_type=r.store.nx_graph.nodes[nid].get("node_type", ""),
            data={k: v for k, v in r.store.nx_graph.nodes[nid].items() if k != "node_type"},
        )
        for nid in subgraph.nodes
    ]
    edges = [
        EdgeOut(
            source=u,
            target=v,
            edge_type=data.get("edge_type", ""),
            metadata={k: v_ for k, v_ in data.items() if k != "edge_type"},
        )
        for u, v, data in subgraph.edges(data=True)
    ]
    from graph.utils import subgraph_to_mermaid
    mermaid = subgraph_to_mermaid(subgraph, r.store.node_map)
    return SubgraphSearchOut(
        query=node_id,
        hits=[],
        start_node_ids=[node_id],
        nodes=nodes,
        edges=edges,
        mermaid=mermaid,
    )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@app.post("/search", response_model=SearchOut, summary="向量检索（返回命中节点）")
def search(body: SearchIn) -> SearchOut:
    r = _get_retriever()
    raw_hits = r.store.similarity_search(body.query, k=body.k)
    hits = [HitOut(**h) for h in raw_hits]
    return SearchOut(query=body.query, hits=hits)


@app.post(
    "/search/subgraph",
    response_model=SubgraphSearchOut,
    summary="向量检索 + 子图遍历（返回 mermaid）",
)
def search_subgraph(body: SubgraphSearchIn) -> SubgraphSearchOut:
    r = _get_retriever()
    result = r.store.search_and_traverse(
        body.query, k=body.k, traverse_from_root=body.traverse_from_root
    )
    hits = [HitOut(**h) for h in result.hits]
    nodes: list[NodeOut] = []
    edges: list[EdgeOut] = []
    for nid in result.subgraph.nodes:
        attrs = result.subgraph.nodes[nid]
        nodes.append(NodeOut(
            id=nid,
            node_type=attrs.get("node_type", ""),
            data={k: v for k, v in attrs.items() if k != "node_type"},
        ))
    for u, v, data in result.subgraph.edges(data=True):
        edges.append(EdgeOut(
            source=u,
            target=v,
            edge_type=data.get("edge_type", ""),
            metadata={k: v_ for k, v_ in data.items() if k != "edge_type"},
        ))
    return SubgraphSearchOut(
        query=body.query,
        hits=hits,
        start_node_ids=result.start_node_ids,
        nodes=nodes,
        edges=edges,
        mermaid=result.mermaid,
    )


# ---------------------------------------------------------------------------
# RAG query
# ---------------------------------------------------------------------------

@app.post("/query", response_model=QueryOut, summary="完整 RAG 问答")
def query(body: QueryIn) -> QueryOut:
    r = _get_retriever()
    # Run the full LangGraph retrieval pipeline
    final_state: dict[str, Any] = r._pipeline.invoke({
        "query": body.question,
        "matched_hits": [],
        "contexts": [],
        "formatted_context": "",
        "answer": "",
    })
    hits = [HitOut(**h) for h in final_state.get("matched_hits", [])]
    return QueryOut(
        question=body.question,
        answer=final_state.get("answer", ""),
        hits=hits,
        context=final_state.get("formatted_context", ""),
    )
