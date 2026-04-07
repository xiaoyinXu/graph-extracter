"""
LangGraph-based SOP knowledge graph extraction pipeline.

Workflow: raw_text → extract_sops (LLM) → build_graph → save_graph

Extraction strategy (three-tier fallback):
  1. Explicit Function/Tool Calling  ← primary: LLM calls the tool with structured args
  2. JSON mode + schema in prompt    ← fallback when tool-calling unsupported
  3. Plain text → regex repair       ← last resort

The tool schema is built DYNAMICALLY from schema/customer_service.yaml via
graph.schema_loader.build_tool_from_schema — no static Pydantic models involved.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Callable, Optional, TYPE_CHECKING

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError
from typing_extensions import TypedDict

from graph.models import (
    ExtractionOutput,
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    ExtractedTool,
)
from graph.schema_loader import build_tool_from_schema
from graph.utils import print_graph_topology

load_dotenv()

SCHEMA_PATH = os.getenv("SCHEMA_PATH", "schema/customer_service.yaml")

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ExtractionState(TypedDict):
    raw_text: str
    extracted_output: Optional[ExtractionOutput]
    graph: Optional[KnowledgeGraph]
    errors: list[str]
    retry_count: int


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _create_llm(temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Extraction prompt + Tool schema
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
你是一个SOP流程知识图谱提取专家。请调用 extract_sop_knowledge_graph 工具，\
从给定的SOP文本中提取完整的知识图谱结构。

## 实体类型说明
- SOP：标准作业流程，含名称、问题类型(issue_type)、细分场景(sub_scenario)、触发样本(trigger_samples)
- SOPStep：步骤，含步骤编号(step_index)、目标(goal)、用户接受检验点(acceptance_check)
- SOPRule：规则，含规则编号(rule_index)、触发条件(condition)、执行思路(execution_approach)、参考话术(reference_script)
- SOPSubRule：追问分支，含条件(condition)、执行思路、参考话术
- Tool：工具调用，含名称(name)、类型(tool_type)

## 枚举值
issue_type: SHIPPING_QUERY | BILLING | TRACKING | DELIVERY_URGE | STUDENT_DISCOUNT |
            COMPLAINT | ORDER_PLACING | REFUND | OTHER | HANDOVER_GAP | STATION_STAY_GAP | POST_DELIVERY_LOSS
tool_type:  TICKET | GROUP_CHAT | QUERY | NOTIFY | ESCALATE

## 文本结构规律
- "第X步" → SOPStep，step_index = X；"目标：" → goal
- 数字序号 → SOPRule（rule_index = 序号）；"执行思路：" / "参考话术：" → 对应字段
- "如果用户追问..." → SOPSubRule；"如果用户不认可...则执行第X步" → acceptance_check
- 【工单新】→ Tool(name="工单新", tool_type="TICKET")
- 【拉群】→ Tool(name="拉群", tool_type="GROUP_CHAT")

## ID命名规范
sop_{场景简写}_001 / {sop_id}_step_{N} / {step_id}_rule_{N} / {rule_id}_subrule_{N} / tool_{name}_001

## 触发样本
从规则触发条件和文本中抽取代表性客户话术（如"怎么包裹不动了""几点能到"等）
"""


def _get_tool_schema() -> dict:
    """
    Dynamically build the OpenAI function/tool schema from the LinkML YAML schema.
    Called once per extraction; result could be cached if performance matters.
    """
    return build_tool_from_schema(
        schema_path=SCHEMA_PATH,
        root_class="SOP",
        list_field_name="sops",
    )


def _get_tool_name() -> str:
    return _get_tool_schema()["function"]["name"]


# ---------------------------------------------------------------------------
# Repair helper for plain-text fallback
# ---------------------------------------------------------------------------

def _try_parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Three extraction strategies
# ---------------------------------------------------------------------------

def _extract_via_tool_calling(
    llm: ChatOpenAI, text: str
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 1 — Explicit Function / Tool Calling (primary).

    Binds the DYNAMICALLY BUILT tool schema (from YAML schema) as an OpenAI
    function/tool and forces the LLM to call it.
    Parses AIMessage.tool_calls[0]["args"] (or additional_kwargs for providers
    that surface calls differently), then validates with Pydantic ExtractionOutput.
    """
    tool_schema = _get_tool_schema()
    tool_name = tool_schema["function"]["name"]
    llm_with_tool = llm.bind_tools(
        [tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )
    response: AIMessage = llm_with_tool.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=text),
    ])

    # Primary path: LangChain-parsed tool_calls
    tool_calls = getattr(response, "tool_calls", [])
    if tool_calls:
        args = tool_calls[0].get("args", {})
        return ExtractionOutput.model_validate(args), ""

    # Fallback path: raw additional_kwargs (some OpenAI-compatible providers)
    raw_calls = response.additional_kwargs.get("tool_calls", [])
    if raw_calls:
        args_raw = raw_calls[0].get("function", {}).get("arguments", "{}")
        return ExtractionOutput.model_validate(json.loads(args_raw)), ""

    return None, f"tool_calling: LLM did not invoke '{tool_name}'"


def _extract_via_json_mode(
    llm: ChatOpenAI, text: str
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 2 — JSON mode with dynamically built schema in system prompt.
    Used when the model does not support tool/function calling.
    """
    tool_schema = _get_tool_schema()
    schema_hint = json.dumps(tool_schema["function"]["parameters"], ensure_ascii=False, indent=2)
    prompt = (
        f"{_SYSTEM_PROMPT}\n\n"
        '## 输出格式\n以合法 JSON 返回，结构为 {"sops": [...]}。\n\n'
        f"## JSON Schema（从YAML schema动态生成）\n{schema_hint}"
    )
    json_llm = llm.bind(response_format={"type": "json_object"})
    response = json_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=text),
    ])
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _try_parse_json(raw)
    if parsed:
        return ExtractionOutput.model_validate(parsed), ""
    return None, "json_mode: could not parse valid JSON from response"


def _extract_via_plain_text(
    llm: ChatOpenAI, text: str
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 3 — Plain text + regex JSON repair (last resort).
    """
    response = llm.invoke([
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=f'请以JSON格式返回提取结果（{{"sops": [...]}}）：\n\n{text}'),
    ])
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _try_parse_json(raw)
    if parsed:
        return ExtractionOutput.model_validate(parsed), ""
    return None, "plain_text: no valid JSON found in response"


# ---------------------------------------------------------------------------
# Extraction node — tries all three strategies in order
# ---------------------------------------------------------------------------

def extract_sops_node(state: ExtractionState) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    retry = state.get("retry_count", 0)
    llm = _create_llm()

    strategies: list[tuple[str, Callable[[ChatOpenAI, str], tuple[Optional[ExtractionOutput], str]]]] = [
        ("function_calling", _extract_via_tool_calling),
        ("json_mode",        _extract_via_json_mode),
        ("plain_text",       _extract_via_plain_text),
    ]

    for strategy_name, strategy_fn in strategies:
        try:
            result, err = strategy_fn(llm, state["raw_text"])
            if result is not None:
                print(f"[extractor] extraction succeeded via {strategy_name}")
                return {
                    "extracted_output": result,
                    "errors": errors,
                    "retry_count": retry + 1,
                }
            errors.append(f"{strategy_name}: {err}")
        except (ValidationError, Exception) as exc:
            errors.append(f"{strategy_name} raised: {exc}")

    return {"extracted_output": None, "errors": errors, "retry_count": retry + 1}


# ---------------------------------------------------------------------------
# Build graph node (ExtractionOutput → KnowledgeGraph)
# ---------------------------------------------------------------------------

def _unique_id() -> str:
    return str(uuid.uuid4())[:8]


def build_graph_node(state: ExtractionState) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    output: Optional[ExtractionOutput] = state.get("extracted_output")

    if output is None:
        errors.append("build_graph: no extracted_output, skipping")
        return {"graph": KnowledgeGraph(), "errors": errors}

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    tool_registry: dict[str, str] = {}  # tool_name → node_id

    def add_node(nid: str, ntype: str, data: dict[str, Any]) -> None:
        nodes.append(GraphNode(id=nid, node_type=ntype, data=data))

    def add_edge(src: str, tgt: str, etype: str, meta: dict[str, Any] | None = None) -> None:
        edges.append(GraphEdge(source=src, target=tgt, edge_type=etype, metadata=meta or {}))

    def ensure_tool(tool: ExtractedTool) -> str:
        if tool.name not in tool_registry:
            tool_registry[tool.name] = tool.id
            add_node(tool.id, "Tool", {
                "name": tool.name,
                "tool_type": tool.tool_type.value,
                "description": tool.description,
            })
        return tool_registry[tool.name]

    for sop in output.sops:
        # SOP node
        add_node(sop.id, "SOP", {
            "name": sop.name,
            "issue_type": sop.issue_type.value,
            "sub_scenario": sop.sub_scenario,
            "trigger_samples": sop.trigger_samples,
        })

        # Steps sorted by step_index
        sorted_steps = sorted(sop.steps, key=lambda s: s.step_index)
        for i, step in enumerate(sorted_steps):
            add_node(step.id, "SOPStep", {
                "step_index": step.step_index,
                "goal": step.goal,
                "acceptance_check": step.acceptance_check,
                "sop_id": sop.id,
            })
            add_edge(sop.id, step.id, "HAS_STEP", {"step_index": step.step_index})

            # NEXT_STEP edge to next step (when user rejects)
            if i + 1 < len(sorted_steps):
                add_edge(step.id, sorted_steps[i + 1].id, "NEXT_STEP", {"when": "REJECTED"})

            # Rules sorted by rule_index
            sorted_rules = sorted(step.rules, key=lambda r: r.rule_index)
            for rule in sorted_rules:
                add_node(rule.id, "SOPRule", {
                    "rule_index": rule.rule_index,
                    "condition": rule.condition,
                    "execution_approach": rule.execution_approach,
                    "reference_script": rule.reference_script,
                    "step_id": step.id,
                    "sop_id": sop.id,
                })
                add_edge(step.id, rule.id, "HAS_RULE", {"rule_index": rule.rule_index})

                # Tools
                for tool in rule.used_tools:
                    tid = ensure_tool(tool)
                    add_edge(rule.id, tid, "USES_TOOL", {})

                # SubRules
                for j, sub in enumerate(rule.sub_rules, start=1):
                    add_node(sub.id, "SOPSubRule", {
                        "condition": sub.condition,
                        "execution_approach": sub.execution_approach,
                        "reference_script": sub.reference_script,
                        "rule_id": rule.id,
                        "step_id": step.id,
                        "sop_id": sop.id,
                    })
                    add_edge(rule.id, sub.id, "HAS_SUB_RULE", {"index": j})

    kg = KnowledgeGraph(nodes=nodes, edges=edges)
    return {"graph": kg, "errors": errors}


# ---------------------------------------------------------------------------
# Save graph node
# ---------------------------------------------------------------------------

def save_graph_node(state: ExtractionState) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    kg: Optional[KnowledgeGraph] = state.get("graph")
    if kg is None:
        errors.append("save_graph: no graph to save")
        return {"errors": errors}

    os.makedirs("data", exist_ok=True)
    path = "data/graph.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write(kg.model_dump_json(indent=2))

    node_count = len(kg.nodes)
    edge_count = len(kg.edges)
    print(f"[extractor] Graph saved: {node_count} nodes, {edge_count} edges → {path}")
    return {"errors": errors}


# ---------------------------------------------------------------------------
# Routing: retry if extraction failed and retry_count < 3
# ---------------------------------------------------------------------------

def should_retry(state: ExtractionState) -> str:
    if state.get("extracted_output") is None and state.get("retry_count", 0) < 3:
        return "extract_sops"
    return "build_graph"


# ---------------------------------------------------------------------------
# Build and compile the extraction graph
# ---------------------------------------------------------------------------

def build_extraction_graph() -> CompiledStateGraph:
    builder = StateGraph(ExtractionState)
    builder.add_node("extract_sops", extract_sops_node)
    builder.add_node("build_graph", build_graph_node)
    builder.add_node("save_graph", save_graph_node)

    builder.add_edge(START, "extract_sops")
    builder.add_conditional_edges("extract_sops", should_retry)
    builder.add_edge("build_graph", "save_graph")
    builder.add_edge("save_graph", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_build(raw_text: str) -> KnowledgeGraph:
    """Extract SOP entities from raw text and persist knowledge graph."""
    graph = build_extraction_graph()
    print_graph_topology(graph, name="Extraction Pipeline")
    final_state = graph.invoke({
        "raw_text": raw_text,
        "extracted_output": None,
        "graph": None,
        "errors": [],
        "retry_count": 0,
    })
    if final_state.get("errors"):
        for err in final_state["errors"]:
            print(f"[extractor] warning: {err}")
    return final_state.get("graph") or KnowledgeGraph()
