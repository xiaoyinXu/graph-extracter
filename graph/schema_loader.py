"""
Dynamic OpenAI function/tool schema builder from a LinkML YAML schema.

Converts LinkML class definitions into a JSON Schema that can be passed to
llm.bind_tools(), letting the LLM use function/tool calling to return
structured extraction results.

LinkML → JSON Schema mappings
------------------------------
  range: string/integer/float/boolean  → primitive JSON Schema types
  range: <EnumName>                    → {"type": "string", "enum": [...values]}
  range: <ClassName>  (no cycle)       → recursively inlined object schema
  range: <ClassName>  (cycle detected) → {"type": "string"}  (ID reference)
  multivalued: true                    → {"type": "array", "items": ...}
  required: true / identifier: true    → added to JSON Schema "required" list
  minimum_value                        → "minimum" on numeric types
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

# -----------------------------------------------------------------------
# LinkML built-in / imported type aliases → JSON Schema primitives
# -----------------------------------------------------------------------
_LINKML_TYPES: dict[str, dict[str, Any]] = {
    "string":     {"type": "string"},
    "str":        {"type": "string"},
    "integer":    {"type": "integer"},
    "int":        {"type": "integer"},
    "float":      {"type": "number"},
    "double":     {"type": "number"},
    "decimal":    {"type": "number"},
    "boolean":    {"type": "boolean"},
    "bool":       {"type": "boolean"},
    "uri":        {"type": "string", "format": "uri"},
    "uriorcurie": {"type": "string"},
    "date":       {"type": "string", "format": "date"},
    "datetime":   {"type": "string", "format": "date-time"},
    "curie":      {"type": "string"},
    "ncname":     {"type": "string"},
}

# Fields that carry execution/runtime metadata — skip during extraction
_DEFAULT_SKIP_FIELDS: frozenset[str] = frozenset({
    "mcp_endpoint",
    "input_schema",
    "health_status",
    "mermaid_code",
    "source_doc",
    "quality_score",
    "success_rate",
    "version",
})


# -----------------------------------------------------------------------
# Internal: resolve a single LinkML range → JSON Schema dict
# -----------------------------------------------------------------------

def _resolve_range(
    range_type: str,
    classes: dict[str, Any],
    enums: dict[str, Any],
    default_range: str,
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict[str, Any]:
    if not range_type:
        range_type = default_range

    # 1. Built-in LinkML primitive type (case-insensitive)
    lower = range_type.lower()
    if lower in _LINKML_TYPES:
        return dict(_LINKML_TYPES[lower])

    # 2. Enum
    if range_type in enums:
        values = list(enums[range_type].get("permissible_values", {}).keys())
        return {"type": "string", "enum": values}

    # 3. Class reference
    if range_type in classes:
        if range_type in visiting:
            # Circular reference detected — use a plain string (ID reference)
            return {"type": "string", "description": f"{range_type} ID（循环引用，仅填写 id 字段值）"}
        if depth >= max_depth:
            return {"type": "object", "description": f"{range_type}（已达最大递归深度）"}
        return _class_to_json_schema(
            range_type, classes, enums, default_range,
            visiting, skip_fields, max_depth, depth,
        )

    # 4. Fallback — treat as string
    return {"type": "string"}


# -----------------------------------------------------------------------
# Internal: convert one LinkML class definition → JSON Schema object
# -----------------------------------------------------------------------

def _class_to_json_schema(
    class_name: str,
    classes: dict[str, Any],
    enums: dict[str, Any],
    default_range: str,
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict[str, Any]:
    class_def: dict[str, Any] = classes.get(class_name) or {}
    attributes: dict[str, Any] = class_def.get("attributes") or {}

    properties: dict[str, dict[str, Any]] = {}
    required_fields: list[str] = []

    # Prevent re-entering this class (cycle guard for children)
    inner_visiting = visiting | {class_name}

    for attr_name, attr_def in attributes.items():
        if attr_name in skip_fields:
            continue
        if attr_def is None:
            attr_def = {}

        range_type: str = attr_def.get("range") or default_range
        multivalued: bool = bool(attr_def.get("multivalued", False))
        is_required: bool = bool(attr_def.get("required", False))
        is_identifier: bool = bool(attr_def.get("identifier", False))
        description: str = _clean_text(attr_def.get("description") or "")
        minimum_value: Optional[int | float] = attr_def.get("minimum_value")

        # Resolve range to JSON Schema
        prop = _resolve_range(
            range_type, classes, enums, default_range,
            inner_visiting, skip_fields, max_depth, depth + 1,
        )

        # Apply minimum constraint on numeric types
        if minimum_value is not None and prop.get("type") in ("integer", "number"):
            prop = {**prop, "minimum": minimum_value}

        # Attach description
        if description:
            prop = {"description": description, **prop}

        # Wrap in array for multivalued attributes
        if multivalued:
            prop = {
                "type": "array",
                "items": {k: v for k, v in prop.items() if k != "description"},
                **({"description": description} if description else {}),
            }

        properties[attr_name] = prop

        if is_required or is_identifier:
            required_fields.append(attr_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required_fields:
        schema["required"] = required_fields

    class_desc = _clean_text(class_def.get("description") or "")
    if class_desc:
        schema["description"] = class_desc

    return schema


def _clean_text(text: str) -> str:
    """Normalize YAML multiline strings to single-line."""
    return " ".join(text.split()) if text else ""


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def build_tool_from_schema(
    schema_path: str,
    root_class: str = "SOP",
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    skip_fields: Optional[set[str]] = None,
    max_depth: int = 5,
    list_field_name: str = "sops",
) -> dict[str, Any]:
    """
    Load a LinkML YAML schema and build an OpenAI function/tool definition.

    Parameters
    ----------
    schema_path     : path to the LinkML YAML schema file
    root_class      : class to use as the extraction root (default "SOP")
    tool_name       : override the generated function name
    tool_description: override the generated description
    skip_fields     : attribute names to exclude (default: execution/metadata fields)
    max_depth       : max recursion depth for nested class inlining (default 5)
    list_field_name : name of the top-level list property (default "sops")

    Returns
    -------
    dict — OpenAI-compatible function/tool definition ready for llm.bind_tools()
    """
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    classes: dict[str, Any] = raw.get("classes") or {}
    enums: dict[str, Any] = raw.get("enums") or {}
    default_range: str = raw.get("default_range") or "string"
    schema_name: str = raw.get("name") or "schema"
    schema_desc: str = _clean_text(raw.get("description") or "")

    _skip = frozenset(skip_fields) if skip_fields is not None else _DEFAULT_SKIP_FIELDS

    # Convert root class to JSON Schema
    root_schema = _class_to_json_schema(
        root_class, classes, enums, default_range,
        visiting=frozenset(), skip_fields=_skip, max_depth=max_depth, depth=0,
    )

    # Wrap as {"<list_field_name>": [<root_schema>, ...]}
    parameters = {
        "type": "object",
        "properties": {
            list_field_name: {
                "type": "array",
                "description": f"从文本中提取的 {root_class} 列表",
                "items": root_schema,
            }
        },
        "required": [list_field_name],
    }

    resolved_name = tool_name or f"extract_{schema_name.lower().replace('-', '_')}"
    resolved_desc = tool_description or (
        f"从文本中提取 {root_class} 知识图谱结构（{schema_desc[:120]}）"
        if schema_desc else f"从文本中提取 {root_class} 知识图谱结构"
    )

    return {
        "type": "function",
        "function": {
            "name": resolved_name,
            "description": resolved_desc,
            "parameters": parameters,
        },
    }


def get_enum_values(schema_path: str, enum_name: str) -> list[str]:
    """Helper: return permissible values for a named enum in the schema."""
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    enums = raw.get("enums") or {}
    return list((enums.get(enum_name) or {}).get("permissible_values", {}).keys())


# -----------------------------------------------------------------------
# Field spec: structured representation of one LinkML attribute
# -----------------------------------------------------------------------

@dataclass
class FieldSpec:
    name: str
    required: bool           # required: true OR identifier: true
    range: str               # "string", "integer", "float", "boolean", or class/enum name
    multivalued: bool
    enum_values: list[str]   # non-empty only when range is an enum name
    description: str
    indexed: bool = False    # x_index: true → include field text in vector index
    min_index_chars: int = 0 # x_index_min_chars → skip short values (e.g. noise filter)


def get_class_field_specs(schema_path: str, class_name: str) -> list[FieldSpec]:
    """Return field specs for every attribute of *class_name* in the schema.

    Parameters
    ----------
    schema_path : path to the LinkML YAML file
    class_name  : class name to inspect (e.g. "SOP", "SOPStep", "SOPRule")

    Returns
    -------
    list[FieldSpec] — one entry per attribute defined for the class
    """
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    classes: dict[str, Any] = raw.get("classes") or {}
    enums: dict[str, Any] = raw.get("enums") or {}
    default_range: str = raw.get("default_range") or "string"

    class_def: dict[str, Any] = classes.get(class_name) or {}
    attributes: dict[str, Any] = class_def.get("attributes") or {}

    specs: list[FieldSpec] = []
    for attr_name, attr_def in attributes.items():
        if attr_def is None:
            attr_def = {}
        range_name: str = attr_def.get("range") or default_range
        is_required: bool = bool(attr_def.get("required") or attr_def.get("identifier"))
        multivalued: bool = bool(attr_def.get("multivalued", False))
        description: str = _clean_text(attr_def.get("description") or "")
        enum_values: list[str] = (
            list((enums[range_name].get("permissible_values") or {}).keys())
            if range_name in enums else []
        )
        indexed: bool = bool(attr_def.get("x_index", False))
        min_index_chars: int = int(attr_def.get("x_index_min_chars", 0))
        specs.append(FieldSpec(
            name=attr_name,
            required=is_required,
            range=range_name,
            multivalued=multivalued,
            enum_values=enum_values,
            description=description,
            indexed=indexed,
            min_index_chars=min_index_chars,
        ))

    return specs


# -----------------------------------------------------------------------
# Relation spec: inter-class relationships derived from schema
# -----------------------------------------------------------------------

@dataclass
class RelationSpec:
    """Describes a relation attribute whose range is another class (not a primitive/enum)."""
    field_name: str      # YAML attribute name, e.g. "steps"
    target_class: str    # range class name, e.g. "SOPStep"
    multivalued: bool    # True = list relation, False = single reference
    edge_type: str       # derived edge type string, e.g. "HAS_STEP"


def field_to_edge_type(field_name: str) -> str:
    """Derive an edge type from a LinkML relation attribute name.

    Convention (matches existing graph.json edge types):
      used_<plural>  →  USES_<SINGULAR>   e.g. ``used_tools``  → ``USES_TOOL``
      <plural>s      →  HAS_<SINGULAR>    e.g. ``steps``       → ``HAS_STEP``
                                               ``sub_rules``    → ``HAS_SUB_RULE``
      <singular>     →  <UPPER>           e.g. ``next_step``   → ``NEXT_STEP``
    """
    if field_name.startswith("used_"):
        rest = field_name[5:]           # "tools"
        singular = rest.rstrip("s")     # "tool"
        return f"USES_{singular.upper()}"
    if field_name.endswith("s"):
        singular = field_name.rstrip("s")  # "step", "rule", "sub_rule"
        return f"HAS_{singular.upper()}"
    return field_name.upper()           # "NEXT_STEP", "CATEGORY", …


def get_relation_specs(schema_path: str, class_name: str) -> list[RelationSpec]:
    """Return all relation fields for *class_name* (range is a class, not a primitive/enum).

    Parameters
    ----------
    schema_path : path to the LinkML YAML file
    class_name  : class to inspect

    Returns
    -------
    list[RelationSpec] — one entry per relation attribute (ordered as in YAML)
    """
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    classes: dict[str, Any] = raw.get("classes") or {}
    enums: dict[str, Any] = raw.get("enums") or {}
    default_range: str = raw.get("default_range") or "string"

    class_def: dict[str, Any] = classes.get(class_name) or {}
    attributes: dict[str, Any] = class_def.get("attributes") or {}

    specs: list[RelationSpec] = []
    for attr_name, attr_def in attributes.items():
        if attr_def is None:
            attr_def = {}
        range_name: str = attr_def.get("range") or default_range
        # Skip primitives and enums — only class references are relations
        if range_name.lower() in _LINKML_TYPES or range_name in enums:
            continue
        if range_name not in classes:
            continue
        multivalued: bool = bool(attr_def.get("multivalued", False))
        specs.append(RelationSpec(
            field_name=attr_name,
            target_class=range_name,
            multivalued=multivalued,
            edge_type=field_to_edge_type(attr_name),
        ))

    return specs


def get_all_class_names(schema_path: str) -> list[str]:
    """Return all class names defined in the schema."""
    with open(schema_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return list((raw.get("classes") or {}).keys())
