"""
LangGraph-based knowledge graph retrieval pipeline.

Workflow: query → search_nodes → expand_context → generate_answer

Node-level retrieval: returns matched (node_id, node_type, score) tuples,
then expands only the relevant ancestor chain + local children rather than
the full SOP tree, keeping LLM context tight.

For SOP-level queries (broad intent), it falls back to full SOP context.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from graph.storage import GraphStore
from graph.utils import print_graph_topology

load_dotenv()

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class RetrievalState(TypedDict):
    query: str
    # search results: list of {node_id, node_type, sop_id, score, text}
    matched_hits: list[dict]
    # expanded context dicts (one per unique SOP or focused rule context)
    contexts: list[dict]
    # human-readable formatted context passed to the answer LLM
    formatted_context: str
    answer: str


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _create_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=0.3,
    )


# ---------------------------------------------------------------------------
# Context formatter
# ---------------------------------------------------------------------------

def _format_sop_context(ctx: dict) -> str:
    """Render full SOP context as structured text for the LLM."""
    lines = []
    sop = ctx.get("sop", {})
    lines.append(f"## SOP: {sop.get('name', '')} [{sop.get('issue_type', '')}]")
    if sop.get("sub_scenario"):
        lines.append(f"细分场景: {sop['sub_scenario']}")

    for step_item in ctx.get("steps", []):
        step = step_item["step"]
        lines.append(f"\n### 第{step['step_index']}步 目标: {step['goal']}")
        if step.get("acceptance_check"):
            lines.append(f"  ↳ 升级条件: {step['acceptance_check']}")

        for rule_item in step_item.get("rules", []):
            rule = rule_item["rule"]
            lines.append(f"\n  规则{rule['rule_index']}: {rule['condition']}")
            lines.append(f"  执行思路: {rule['execution_approach']}")
            if rule.get("reference_script"):
                lines.append(f"  参考话术: {rule['reference_script']}")
            for sub in rule_item.get("sub_rules", []):
                lines.append(f"    ↳ 追问[{sub['condition']}]")
                lines.append(f"       执行思路: {sub['execution_approach']}")
                if sub.get("reference_script"):
                    lines.append(f"       参考话术: {sub['reference_script']}")
            for tool in rule_item.get("tools", []):
                lines.append(f"    🔧 工具: {tool['name']} ({tool['tool_type']})")

    return "\n".join(lines)


def _format_rule_context(ctx: dict) -> str:
    """Render focused rule/sub-rule context."""
    lines = []
    node = ctx.get("node", {})
    ntype = node.get("node_type", "")

    ancestors = ctx.get("ancestors", [])
    if ancestors:
        breadcrumb = " → ".join(
            a.get("name") or f"{a['node_type']}[{a.get('step_index', a.get('rule_index', ''))}]"
            for a in ancestors
        )
        lines.append(f"路径: {breadcrumb}")

    if ntype == "SOPRule":
        lines.append(f"\n规则{node.get('rule_index', '')}: {node.get('condition', '')}")
        lines.append(f"执行思路: {node.get('execution_approach', '')}")
        if node.get("reference_script"):
            lines.append(f"参考话术: {node['reference_script']}")
        for sub in ctx.get("children", []):
            lines.append(f"  ↳ 追问[{sub['condition']}]")
            lines.append(f"     执行思路: {sub['execution_approach']}")
            if sub.get("reference_script"):
                lines.append(f"     参考话术: {sub['reference_script']}")
    elif ntype == "SOPSubRule":
        lines.append(f"\n追问分支: {node.get('condition', '')}")
        lines.append(f"执行思路: {node.get('execution_approach', '')}")
        if node.get("reference_script"):
            lines.append(f"参考话术: {node['reference_script']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Retrieval graph nodes
# ---------------------------------------------------------------------------

def search_nodes_node(state: RetrievalState, *, store: GraphStore) -> dict:
    """Vector search → top-k node hits with score."""
    hits = store.similarity_search(state["query"], k=8)
    return {"matched_hits": hits}


def expand_context_node(state: RetrievalState, *, store: GraphStore) -> dict:
    """
    Expand each hit to its context.

    Strategy:
    - If the top hit is a SOP node OR score is high (≥0.85 cosine): return full SOP context
    - Otherwise: return focused rule/sub-rule context + ancestors
    - Deduplicate by sop_id to avoid redundant context
    """
    hits = state.get("matched_hits", [])
    if not hits:
        return {"contexts": [], "formatted_context": "未找到相关知识图谱内容。"}

    seen_sop_ids: set[str] = set()
    contexts: list[dict] = []
    formatted_parts: list[str] = []

    # Sort by score descending (higher cosine = more relevant)
    sorted_hits = sorted(hits, key=lambda h: h["score"], reverse=True)

    for hit in sorted_hits[:5]:  # cap at 5 to keep context tight
        sop_id = hit.get("sop_id", "")
        node_id = hit["node_id"]
        node_type = hit["node_type"]
        score = hit["score"]

        # SOP-level hit or high-confidence match → full SOP context
        if node_type == "SOP" or score >= 0.85:
            target_sop_id = node_id if node_type == "SOP" else sop_id
            if target_sop_id and target_sop_id not in seen_sop_ids:
                seen_sop_ids.add(target_sop_id)
                ctx = store.get_sop_context(target_sop_id)
                if ctx:
                    contexts.append(ctx)
                    formatted_parts.append(_format_sop_context(ctx))
        else:
            # Rule/SubRule hit → focused context
            if sop_id not in seen_sop_ids:
                ctx = store.get_rule_context(node_id, node_type)
                if ctx:
                    # add sop_id to seen to prevent duplicate full SOP expansion
                    seen_sop_ids.add(sop_id)
                    contexts.append(ctx)
                    formatted_parts.append(_format_rule_context(ctx))

    if not formatted_parts:
        # fallback: return full SOP for the top hit's sop_id
        top_sop_id = sorted_hits[0].get("sop_id", "")
        if top_sop_id:
            ctx = store.get_sop_context(top_sop_id)
            if ctx:
                contexts.append(ctx)
                formatted_parts.append(_format_sop_context(ctx))

    formatted_context = "\n\n---\n\n".join(formatted_parts) or "未找到相关知识图谱内容。"
    return {"contexts": contexts, "formatted_context": formatted_context}


_ANSWER_SYSTEM = """\
你是一个客服知识库助手。根据提供的SOP知识图谱内容，回答用户的问题。

要求：
1. 直接引用相关SOP的执行思路和参考话术
2. 保留话术中的变量占位符（如 {{addon:11746}}）
3. 如果涉及多个步骤，按步骤顺序说明处理方案
4. 如果没有找到相关内容，明确说明
5. 回答使用中文

知识图谱内容：
{context}
"""


def generate_answer_node(state: RetrievalState) -> dict:
    """Use LLM to synthesize answer from retrieved context."""
    context = state.get("formatted_context", "")
    if not context or context == "未找到相关知识图谱内容。":
        return {"answer": "抱歉，知识库中未找到与您问题相关的SOP内容。"}

    llm = _create_llm()
    system = _ANSWER_SYSTEM.format(context=context)
    response = llm.invoke([
        SystemMessage(content=system),
        HumanMessage(content=state["query"]),
    ])
    answer = response.content if hasattr(response, "content") else str(response)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Build and compile the retrieval graph
# ---------------------------------------------------------------------------

def build_retrieval_graph(store: GraphStore):
    """Build LangGraph retrieval pipeline bound to a GraphStore instance."""
    from functools import partial

    builder = StateGraph(RetrievalState)
    builder.add_node("search_nodes", partial(search_nodes_node, store=store))
    builder.add_node("expand_context", partial(expand_context_node, store=store))
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "search_nodes")
    builder.add_edge("search_nodes", "expand_context")
    builder.add_edge("expand_context", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class KnowledgeGraphRetriever:
    """High-level retriever that wraps the LangGraph pipeline."""

    def __init__(self, graph_path: str = "data/graph.json"):
        print(f"[retriever] loading graph from {graph_path}...")
        self.store = GraphStore.load(graph_path)
        self._pipeline = build_retrieval_graph(self.store)
        print_graph_topology(self._pipeline, name="Retrieval Pipeline")
        print(f"[retriever] ready — "
              f"{self.store.nx_graph.number_of_nodes()} nodes, "
              f"{self.store.nx_graph.number_of_edges()} edges")

    def query(self, question: str, verbose: bool = False) -> str:
        """Ask a question against the knowledge graph."""
        final_state = self._pipeline.invoke({
            "query": question,
            "matched_hits": [],
            "contexts": [],
            "formatted_context": "",
            "answer": "",
        })
        if verbose:
            hits = final_state.get("matched_hits", [])
            print(f"\n[retriever] top hits for '{question}':")
            for h in hits[:3]:
                print(f"  [{h['node_type']}] {h['text'][:80]}... score={h['score']:.3f}")
            print(f"\n[retriever] context:\n{final_state.get('formatted_context', '')[:500]}...\n")
        return final_state.get("answer", "")

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Raw vector search without answer generation."""
        return self.store.similarity_search(query, k=k)
