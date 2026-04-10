"""
LangGraph-based knowledge graph extraction pipeline.

Workflow: raw_text → extract_entities (LLM) → build_graph → save_graph

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
from typing import Any, Optional, TYPE_CHECKING

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
)
from graph.schema_loader import (
    build_tool_from_schema,
    build_system_prompt,
    get_class_field_specs,
    get_relation_specs,
    field_to_edge_type,
    get_root_class,
    get_root_list_key,
)
from graph.utils import print_graph_topology

load_dotenv()

SCHEMA_PATH = os.getenv("SCHEMA_PATH", "schema/customer_service.yaml")
_ROOT_CLASS: str = get_root_class(SCHEMA_PATH)
_LIST_FIELD: str = get_root_list_key(SCHEMA_PATH, _ROOT_CLASS)

# ---------------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------------

class ExtractionState(TypedDict):
    raw_text: str
    schema_path: str   # empty string = use module-level SCHEMA_PATH default
    extracted_output: Optional[ExtractionOutput]
    graph: Optional[KnowledgeGraph]
    errors: list[str]
    retry_count: int
    validation_issues: list[dict[str, Any]]   # populated by validate_graph_node


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
# Three extraction strategies
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
    llm: ChatOpenAI,
    text: str,
    schema_path: str,
    root_class: str,
    list_field: str,
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 1 — Explicit Function / Tool Calling (primary).

    Binds the DYNAMICALLY BUILT tool schema (from YAML schema) as an OpenAI
    function/tool and forces the LLM to call it.
    Parses AIMessage.tool_calls[0]["args"] (or additional_kwargs for providers
    that surface calls differently), then validates with Pydantic ExtractionOutput.
    """
    tool_schema = build_tool_from_schema(
        schema_path=schema_path, root_class=root_class, list_field_name=list_field
    )
    system_prompt = build_system_prompt(
        schema_path=schema_path, root_class=root_class, list_field_name=list_field
    )
    tool_name = tool_schema["function"]["name"]
    llm_with_tool = llm.bind_tools(
        [tool_schema],
        tool_choice={"type": "function", "function": {"name": tool_name}},
    )
    response: AIMessage = llm_with_tool.invoke([
        SystemMessage(content=system_prompt),
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
    llm: ChatOpenAI,
    text: str,
    schema_path: str,
    root_class: str,
    list_field: str,
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 2 — JSON mode with dynamically built schema in system prompt.
    Used when the model does not support tool/function calling.
    """
    tool_schema = build_tool_from_schema(
        schema_path=schema_path, root_class=root_class, list_field_name=list_field
    )
    schema_hint = json.dumps(tool_schema["function"]["parameters"], ensure_ascii=False, indent=2)
    system_prompt = build_system_prompt(
        schema_path=schema_path, root_class=root_class, list_field_name=list_field
    )
    prompt = (
        f"{system_prompt}\n\n"
        f'## 输出格式\n以合法 JSON 返回，结构为 {{"{list_field}": [...]}}。\n\n'
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
    llm: ChatOpenAI,
    text: str,
    schema_path: str,
    root_class: str,
    list_field: str,
) -> tuple[Optional[ExtractionOutput], str]:
    """
    Strategy 3 — Plain text + regex JSON repair (last resort).
    """
    system_prompt = build_system_prompt(
        schema_path=schema_path, root_class=root_class, list_field_name=list_field
    )
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f'请以JSON格式返回提取结果（{{"{list_field}": [...]}}）：\n\n{text}'),
    ])
    raw = response.content if hasattr(response, "content") else str(response)
    parsed = _try_parse_json(raw)
    if parsed:
        return ExtractionOutput.model_validate(parsed), ""
    return None, "plain_text: no valid JSON found in response"


# ---------------------------------------------------------------------------
# Extraction node — tries all three strategies in order
# ---------------------------------------------------------------------------

def extract_entities_node(state: ExtractionState) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    retry = state.get("retry_count", 0)
    llm = _create_llm()

    # Resolve schema config from state (fallback to module-level defaults)
    _sp = state.get("schema_path") or SCHEMA_PATH
    _rc = get_root_class(_sp)
    _lf = get_root_list_key(_sp, _rc)

    strategies: list[tuple[str, Any]] = [
        ("function_calling", _extract_via_tool_calling),
        ("json_mode",        _extract_via_json_mode),
        ("plain_text",       _extract_via_plain_text),
    ]

    for strategy_name, strategy_fn in strategies:
        try:
            result, err = strategy_fn(llm, state["raw_text"], _sp, _rc, _lf)
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


def _to_snake(s: str) -> str:
    """Convert CamelCase to snake_case: 'SubRule' → 'sub_rule'."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
    s = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", s)
    return s.lower()


def _class_to_id_key(class_name: str, root_class: str) -> str:
    """Derive the ancestor-id key for a class relative to the root.

    Examples (root_class='SOP'): 'SOPStep' → 'step_id', 'SOP' → 'sop_id'.
    For classes that don't start with root_class, the full snake_case name is used.
    """
    remainder = class_name[len(root_class):] if class_name.startswith(root_class) else class_name
    if not remainder:
        remainder = root_class
    return f"{_to_snake(remainder)}_id"


def _add_entity(
    obj: dict[str, Any],
    class_name: str,
    ancestor_ids: dict[str, str],
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    node_registry: set[str],
    schema_path: str,
    root_class: str,
) -> None:
    """Recursively add a node and its children to the graph (schema-driven)."""
    node_id: str = obj.get("id") or str(uuid.uuid4())

    rel_specs = get_relation_specs(schema_path, class_name)
    relation_field_names = {spec.field_name for spec in rel_specs}

    # Scalar data: all non-id, non-relation fields + accumulated ancestor ids
    data: dict[str, Any] = {
        k: v for k, v in obj.items()
        if k != "id" and k not in relation_field_names
    }
    data.update(ancestor_ids)

    if node_id not in node_registry:
        node_registry.add(node_id)
        nodes.append(GraphNode(id=node_id, node_type=class_name, data=data))

    # Ancestor ids for direct children of this node
    child_ancestor_ids = {**ancestor_ids, _class_to_id_key(class_name, root_class): node_id}

    for spec in rel_specs:
        children_raw = obj.get(spec.field_name)
        if not children_raw:
            continue

        child_list: list[dict] = children_raw if isinstance(children_raw, list) else [children_raw]

        # Sort by *_index field if the child class has one
        try:
            child_field_specs = get_class_field_specs(schema_path, spec.target_class)
            index_field = next(
                (s.name for s in child_field_specs if s.name.endswith("_index")),
                None,
            )
        except Exception:
            index_field = None

        if index_field:
            child_list = sorted(child_list, key=lambda c: c.get(index_field, 0))

        prev_child_id: Optional[str] = None
        for i, child_obj in enumerate(child_list):
            child_id: str = child_obj.get("id") or str(uuid.uuid4())

            # Edge from parent to child
            edge_meta: dict[str, Any] = ({index_field: child_obj.get(index_field)}
                                         if index_field else {"index": i + 1})
            edges.append(GraphEdge(
                source=node_id,
                target=child_id,
                edge_type=spec.edge_type,
                metadata=edge_meta,
            ))

            # Sequential NEXT_* edge between consecutive ordered siblings
            if index_field and prev_child_id is not None:
                next_edge_type = f"NEXT_{_to_snake(spec.target_class[len(root_class):] or spec.target_class).upper()}"
                edges.append(GraphEdge(
                    source=prev_child_id,
                    target=child_id,
                    edge_type=next_edge_type,
                    metadata={"when": "REJECTED"},
                ))

            _add_entity(child_obj, spec.target_class, child_ancestor_ids, nodes, edges, node_registry, schema_path, root_class)
            prev_child_id = child_id


def build_graph_node(state: ExtractionState) -> dict[str, Any]:
    errors = list(state.get("errors", []))
    output: Optional[ExtractionOutput] = state.get("extracted_output")

    if output is None:
        errors.append("build_graph: no extracted_output, skipping")
        return {"graph": KnowledgeGraph(), "errors": errors}

    _sp = state.get("schema_path") or SCHEMA_PATH
    _rc = get_root_class(_sp)
    _lf = get_root_list_key(_sp, _rc)

    nodes: list[GraphNode] = []
    edges: list[GraphEdge] = []
    node_registry: set[str] = set()  # deduplicate shared nodes (e.g. same Tool)

    raw = output.model_dump(mode="json")
    for root_obj in raw.get(_lf, []):
        _add_entity(root_obj, _rc, {}, nodes, edges, node_registry, _sp, _rc)

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
# Validate graph node — schema-driven entity attribute checks
# ---------------------------------------------------------------------------

# Primitive LinkML range names that map to Python scalar types
_PRIMITIVE_RANGES: frozenset[str] = frozenset({
    "string", "str", "integer", "int", "float", "double",
    "decimal", "boolean", "bool", "uri", "uriorcurie",
    "date", "datetime", "curie", "ncname",
})


def validate_graph_node(state: ExtractionState) -> dict[str, Any]:
    """Schema-driven validation of every node in the extracted KnowledgeGraph.

    For each node, loads the FieldSpec list for its ``node_type`` from the
    LinkML YAML (via ``get_class_field_specs``), then checks:

    - ERROR  : required field absent or None/empty in node.data
    - ERROR  : field value not in allowed enum values
    - WARNING: recommended (non-required) field is None or empty string
    - WARNING: multivalued field is present but empty list

    Issues are collected into ``state["validation_issues"]`` as dicts:
    ``{node_id, node_type, field, severity, message}``.

    If any ERROR exists and retry_count < 3, the routing sends the pipeline
    back to ``extract_sops`` for re-extraction.
    """
    errors = list(state.get("errors", []))
    kg: Optional[KnowledgeGraph] = state.get("graph")
    issues: list[dict[str, Any]] = []

    if kg is None:
        errors.append("validate_graph: no graph to validate")
        return {"errors": errors, "validation_issues": issues}

    # Cache field specs per node_type to avoid re-parsing YAML repeatedly
    specs_cache: dict[str, list] = {}
    _sp = state.get("schema_path") or SCHEMA_PATH

    # --- Structural check: duplicate node IDs ---
    seen_ids: dict[str, int] = {}
    for node in kg.nodes:
        seen_ids[node.id] = seen_ids.get(node.id, 0) + 1
    for nid, count in seen_ids.items():
        if count > 1:
            issues.append({
                "node_id": nid, "node_type": "?",
                "field": "id", "severity": "ERROR",
                "message": f"Duplicate node id appears {count} times",
            })

    # --- Structural check: edge endpoints exist ---
    node_id_set = {n.id for n in kg.nodes}
    for edge in kg.edges:
        for endpoint, label in ((edge.source, "source"), (edge.target, "target")):
            if endpoint not in node_id_set:
                issues.append({
                    "node_id": f"{edge.source}→{edge.target}",
                    "node_type": "Edge", "field": label, "severity": "ERROR",
                    "message": f"Edge {label} '{endpoint}' not found in node list",
                })

    # --- Per-node attribute validation ---
    for node in kg.nodes:
        ntype = node.node_type

        if ntype not in specs_cache:
            try:
                specs_cache[ntype] = get_class_field_specs(_sp, ntype)
            except Exception as exc:
                issues.append({
                    "node_id": node.id, "node_type": ntype,
                    "field": "*", "severity": "WARNING",
                    "message": f"Cannot load schema for '{ntype}': {exc}",
                })
                specs_cache[ntype] = []

        for spec in specs_cache[ntype]:
            # Relation fields (range is another class, not a primitive or enum)
            # are not stored as scalar attributes in node.data — skip them.
            if spec.range not in _PRIMITIVE_RANGES and not spec.enum_values:
                continue

            # The identifier field (e.g. "id") lives on node.id, not node.data.
            # It is always populated by GraphNode construction — just verify it's non-empty.
            if spec.name == "id":
                if not node.id or not str(node.id).strip():
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": "id", "severity": "ERROR",
                        "message": "Identifier field 'id' is empty",
                    })
                continue

            value = node.data.get(spec.name)

            # --- Missing or empty required field ---
            if spec.required:
                if value is None:
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": spec.name, "severity": "ERROR",
                        "message": f"Required field '{spec.name}' is missing (None)",
                    })
                    continue
                if not spec.multivalued and isinstance(value, str) and not value.strip():
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": spec.name, "severity": "ERROR",
                        "message": f"Required field '{spec.name}' is empty string",
                    })
                    continue
                if spec.multivalued and isinstance(value, list) and len(value) == 0:
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": spec.name, "severity": "WARNING",
                        "message": f"Required multivalued field '{spec.name}' is empty list",
                    })
                    continue

            # --- Optional but present: warn if empty ---
            elif not spec.required and value is not None:
                if not spec.multivalued and isinstance(value, str) and not value.strip():
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": spec.name, "severity": "WARNING",
                        "message": f"Optional field '{spec.name}' is empty string",
                    })

            # --- Enum value check ---
            if spec.enum_values and value is not None and not spec.multivalued:
                if str(value) not in spec.enum_values:
                    issues.append({
                        "node_id": node.id, "node_type": ntype,
                        "field": spec.name, "severity": "ERROR",
                        "message": (
                            f"Field '{spec.name}' value '{value}' not in allowed "
                            f"enum values: {spec.enum_values}"
                        ),
                    })

    # Log summary
    error_count = sum(1 for i in issues if i["severity"] == "ERROR")
    warn_count  = sum(1 for i in issues if i["severity"] == "WARNING")
    print(f"[validator] {len(kg.nodes)} nodes checked — "
          f"{error_count} errors, {warn_count} warnings")
    for issue in issues:
        prefix = "✗" if issue["severity"] == "ERROR" else "⚠"
        print(f"  {prefix} [{issue['node_type']}] {issue['node_id']}.{issue['field']}: "
              f"{issue['message']}")

    return {"errors": errors, "validation_issues": issues}


# ---------------------------------------------------------------------------
# Routing: retry if extraction failed and retry_count < 3
# ---------------------------------------------------------------------------

def should_retry(state: ExtractionState) -> str:
    if state.get("extracted_output") is None and state.get("retry_count", 0) < 3:
        return "extract_entities"
    return "build_graph"


def should_retry_after_validation(state: ExtractionState) -> str:
    """Re-extract if there are schema ERRORs and we haven't exhausted retries."""
    issues = state.get("validation_issues", [])
    has_errors = any(i["severity"] == "ERROR" for i in issues)
    if has_errors and state.get("retry_count", 0) < 3:
        print(f"[validator] ERROR issues found, retrying extraction "
              f"(attempt {state.get('retry_count', 0) + 1}/3)")
        return "extract_entities"
    return "save_graph"


# ---------------------------------------------------------------------------
# Build and compile the extraction graph
# ---------------------------------------------------------------------------

def build_extraction_graph() -> CompiledStateGraph:
    builder = StateGraph(ExtractionState)
    builder.add_node("extract_entities", extract_entities_node)
    builder.add_node("build_graph",    build_graph_node)
    builder.add_node("validate_graph", validate_graph_node)
    builder.add_node("save_graph",     save_graph_node)

    builder.add_edge(START, "extract_entities")
    builder.add_conditional_edges("extract_entities", should_retry)
    builder.add_edge("build_graph", "validate_graph")
    builder.add_conditional_edges("validate_graph", should_retry_after_validation)
    builder.add_edge("save_graph", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_and_build(raw_text: str, schema_path: Optional[str] = None) -> KnowledgeGraph:
    """Extract entities from raw text and persist knowledge graph.

    Args:
        raw_text:    Source document text to extract from.
        schema_path: Optional path to a LinkML YAML schema.  When omitted the
                     module-level ``SCHEMA_PATH`` (defaulting to
                     ``schema/customer_service.yaml``) is used.
    """
    graph = build_extraction_graph()
    print_graph_topology(graph, name="Extraction Pipeline")
    final_state = graph.invoke({
        "raw_text": raw_text,
        "schema_path": schema_path or "",
        "extracted_output": None,
        "graph": None,
        "errors": [],
        "retry_count": 0,
        "validation_issues": [],
    })
    if final_state.get("errors"):
        for err in final_state["errors"]:
            print(f"[extractor] warning: {err}")
    return final_state.get("graph") or KnowledgeGraph()
