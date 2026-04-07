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

from pathlib import Path
from typing import Optional

import yaml

# -----------------------------------------------------------------------
# LinkML built-in / imported type aliases → JSON Schema primitives
# -----------------------------------------------------------------------
_LINKML_TYPES: dict[str, dict] = {
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
    classes: dict,
    enums: dict,
    default_range: str,
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict:
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
    classes: dict,
    enums: dict,
    default_range: str,
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict:
    class_def = classes.get(class_name) or {}
    attributes: dict = class_def.get("attributes") or {}

    properties: dict[str, dict] = {}
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
        minimum_value = attr_def.get("minimum_value")

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

    schema: dict = {"type": "object", "properties": properties}
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
) -> dict:
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

    classes: dict = raw.get("classes") or {}
    enums: dict = raw.get("enums") or {}
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
