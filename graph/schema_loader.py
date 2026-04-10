"""
Schema loader using linkml-runtime SchemaView.

All schema operations (class discovery, slot enumeration, JSON Schema generation,
system-prompt building) are driven by the LinkML YAML via SchemaView — no
hardcoded class or field names in production code.

Key functions
-------------
build_tool_from_schema  — OpenAI function/tool definition from schema root class
build_system_prompt     — dynamic LLM system prompt derived from schema
get_class_field_specs   — FieldSpec list for a class (includes x_index annotations)
get_relation_specs      — RelationSpec list (fields whose range is another class)
get_enum_values         — permissible values for a named enum
get_all_class_names     — list of all class names in the schema

x_index custom annotation
--------------------------
``x_index: true`` on a slot marks it for vector indexing.
``x_index_min_chars: N`` sets a minimum text length filter.
These are NOT standard LinkML — stripped before loading SchemaView and stored
separately in a per-schema annotations cache.

LinkML → JSON Schema type mapping
----------------------------------
  range: string / integer / float / boolean  → JSON Schema primitives
  range: <EnumName>                          → {"type": "string", "enum": [...]}
  range: <ClassName> (no cycle)              → recursively inlined object
  range: <ClassName> (cycle / max depth)     → {"type": "string"} (id ref)
  multivalued: true                          → {"type": "array", "items": ...}
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

import yaml
from linkml_runtime.utils.schemaview import SchemaView

# ---------------------------------------------------------------------------
# Custom extension keys (non-standard LinkML — stripped before SchemaView)
# ---------------------------------------------------------------------------
_CUSTOM_KEYS: frozenset[str] = frozenset({"x_index", "x_index_min_chars"})

# ---------------------------------------------------------------------------
# LinkML built-in type names → JSON Schema primitives (case-insensitive)
# ---------------------------------------------------------------------------
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

# Slots to omit from the extraction tool schema (runtime / metadata only)
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


# ---------------------------------------------------------------------------
# Module-level cache: schema_path → (SchemaView, custom_annotations)
# ---------------------------------------------------------------------------
_SCHEMA_CACHE: dict[str, tuple[SchemaView, dict[str, dict[str, dict]]]] = {}


def _load_schema(schema_path: str) -> tuple[SchemaView, dict[str, dict[str, dict]]]:
    """Load and cache (SchemaView, custom_annotations) for *schema_path*.

    ``x_index`` and ``x_index_min_chars`` are non-standard LinkML keys that
    cause SchemaView to raise TypeError.  They are stripped from slot
    definitions before constructing SchemaView and returned separately as
    ``{class_name: {slot_name: {key: value}}}``.
    """
    if schema_path in _SCHEMA_CACHE:
        return _SCHEMA_CACHE[schema_path]

    with open(schema_path, encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh)

    custom: dict[str, dict[str, dict]] = {}
    for cls_name, cls_def in (raw.get("classes") or {}).items():
        for slot_name, slot_def in ((cls_def or {}).get("attributes") or {}).items():
            if not slot_def:
                continue
            extracted = {k: slot_def.pop(k) for k in _CUSTOM_KEYS if k in slot_def}
            if extracted:
                custom.setdefault(cls_name, {})[slot_name] = extracted

    clean_yaml = yaml.dump(raw, allow_unicode=True, default_flow_style=False)
    sv = SchemaView(clean_yaml)
    _SCHEMA_CACHE[schema_path] = (sv, custom)
    return sv, custom


# ---------------------------------------------------------------------------
# Internal: JSON Schema generation via SchemaView
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""


def _resolve_range(
    sv: SchemaView,
    range_name: Optional[str],
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict[str, Any]:
    if not range_name:
        return {"type": "string"}

    lower = range_name.lower()

    # 1. Built-in LinkML primitive
    if lower in _LINKML_TYPES:
        return dict(_LINKML_TYPES[lower])

    # 2. Enum — expand permissible values inline
    if range_name in sv.all_enums():
        pvs = list(sv.get_enum(range_name).permissible_values.keys())
        return {"type": "string", "enum": pvs}

    # 3. Class reference — inline recursively (cycle / depth guard)
    if range_name in sv.all_classes():
        if range_name in visiting or depth >= max_depth:
            return {"type": "string", "description": f"{range_name} ID（循环引用，填写 id 字段值）"}
        return _class_to_json_schema(sv, range_name, visiting, skip_fields, max_depth, depth)

    # 4. Known type in SchemaView (imported from linkml:types)
    td = sv.get_type(range_name)
    if td is not None:
        uri = str(td.uri) if td.uri else ""
        if "integer" in uri or "int" in lower:
            return {"type": "integer"}
        if "float" in uri or "double" in uri or "decimal" in uri:
            return {"type": "number"}
        if "boolean" in uri or "bool" in lower:
            return {"type": "boolean"}
        return {"type": "string"}

    # 5. Fallback
    return {"type": "string"}


def _class_to_json_schema(
    sv: SchemaView,
    class_name: str,
    visiting: frozenset[str],
    skip_fields: frozenset[str],
    max_depth: int,
    depth: int,
) -> dict[str, Any]:
    """Convert one LinkML class to a JSON Schema object using SchemaView."""
    cls_def = sv.get_class(class_name)
    slots = sv.class_induced_slots(class_name)

    properties: dict[str, dict[str, Any]] = {}
    required_fields: list[str] = []
    inner_visiting = visiting | {class_name}

    for slot in slots:
        if slot.name in skip_fields:
            continue

        range_name: Optional[str] = slot.range
        multivalued: bool = bool(slot.multivalued)
        is_required: bool = bool(slot.required or slot.identifier)
        description: str = _clean_text(slot.description or "")
        minimum_value = slot.minimum_value

        prop = _resolve_range(sv, range_name, inner_visiting, skip_fields, max_depth, depth + 1)

        if minimum_value is not None and prop.get("type") in ("integer", "number"):
            prop = {**prop, "minimum": minimum_value}

        if description:
            prop = {"description": description, **prop}

        if multivalued:
            items = {k: v for k, v in prop.items() if k != "description"}
            prop = {"type": "array", "items": items}
            if description:
                prop["description"] = description

        properties[slot.name] = prop
        if is_required:
            required_fields.append(slot.name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required_fields:
        schema["required"] = required_fields
    if cls_def and cls_def.description:
        schema["description"] = _clean_text(cls_def.description)

    return schema


# ---------------------------------------------------------------------------
# Public: build OpenAI tool definition
# ---------------------------------------------------------------------------

def get_root_class(schema_path: str) -> str:
    """Return the name of the class marked ``tree_root: true`` in the schema.

    Raises ``ValueError`` if there is not exactly one such class.
    """
    sv, _ = _load_schema(schema_path)
    roots = [name for name, cls in sv.all_classes().items() if getattr(cls, "tree_root", False)]
    if len(roots) == 1:
        return roots[0]
    if not roots:
        raise ValueError(
            f"No class with tree_root: true found in {schema_path}. "
            "Add 'tree_root: true' to exactly one class in the YAML schema."
        )
    raise ValueError(
        f"Multiple tree_root classes found in {schema_path}: {roots}. "
        "Only one class may be marked tree_root: true."
    )


def get_root_list_key(schema_path: str, root_class: Optional[str] = None) -> str:
    """Derive the list-field key name for the root class.

    Convention: ``root_class.lower() + "s"`` (e.g. ``"SOP"`` → ``"sops"``).
    """
    if root_class is None:
        root_class = get_root_class(schema_path)
    return root_class.lower() + "s"


def build_tool_from_schema(
    schema_path: str,
    root_class: Optional[str] = None,
    tool_name: Optional[str] = None,
    tool_description: Optional[str] = None,
    skip_fields: Optional[set[str]] = None,
    max_depth: int = 5,
    list_field_name: Optional[str] = None,
) -> dict[str, Any]:
    """Build an OpenAI function/tool definition from a LinkML schema.

    Parameters
    ----------
    schema_path      : path to the LinkML YAML file
    root_class       : class used as the extraction root; defaults to the
                       class marked ``tree_root: true`` in the schema
    tool_name        : override the generated function name
    tool_description : override the generated description
    skip_fields      : slot names to omit (default: runtime/metadata slots)
    max_depth        : max recursion depth for nested class inlining
    list_field_name  : name of the top-level list property; defaults to
                       ``root_class.lower() + "s"``

    Returns
    -------
    dict — OpenAI-compatible ``{"type": "function", "function": {...}}`` dict
    """
    if root_class is None:
        root_class = get_root_class(schema_path)
    if list_field_name is None:
        list_field_name = get_root_list_key(schema_path, root_class)
    sv, _ = _load_schema(schema_path)
    _skip = frozenset(skip_fields) if skip_fields is not None else _DEFAULT_SKIP_FIELDS

    root_schema = _class_to_json_schema(
        sv, root_class, frozenset(), _skip, max_depth, depth=0
    )

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

    schema_name = sv.schema.name or "schema"
    schema_desc = _clean_text(sv.schema.description or "")

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


# ---------------------------------------------------------------------------
# Public: build dynamic system prompt from schema
# ---------------------------------------------------------------------------

def build_system_prompt(
    schema_path: str,
    root_class: Optional[str] = None,
    list_field_name: Optional[str] = None,
    skip_fields: Optional[set[str]] = None,
) -> str:
    """Generate a knowledge-graph extraction system prompt from the schema.

    The prompt is built entirely from the schema's class/slot descriptions,
    so it stays current whenever the YAML is updated — no domain-specific text.
    """
    if root_class is None:
        root_class = get_root_class(schema_path)
    if list_field_name is None:
        list_field_name = get_root_list_key(schema_path, root_class)
    sv, _ = _load_schema(schema_path)
    _skip = frozenset(skip_fields) if skip_fields is not None else _DEFAULT_SKIP_FIELDS
    all_classes = sv.all_classes()
    all_enums = sv.all_enums()

    tool_schema = build_tool_from_schema(
        schema_path, root_class=root_class, list_field_name=list_field_name
    )
    tool_fn_name = tool_schema["function"]["name"]

    schema_desc = _clean_text(sv.schema.description or "")

    # Build entity-type descriptions — only reachable classes from root
    visited: set[str] = set()
    order: list[str] = []

    def _collect(cls_name: str) -> None:
        if cls_name in visited or cls_name not in all_classes:
            return
        visited.add(cls_name)
        order.append(cls_name)
        for slot in sv.class_induced_slots(cls_name):
            if slot.name in _skip and slot.range in all_classes:
                continue
            if slot.range and slot.range in all_classes:
                _collect(slot.range)

    _collect(root_class)

    entity_lines: list[str] = []
    for cls_name in order:
        cls_def = sv.get_class(cls_name)
        cls_desc = _clean_text(cls_def.description or "") if cls_def else ""
        slots = sv.class_induced_slots(cls_name)
        scalar_slots = [
            s for s in slots
            if s.name not in _skip
            and (not s.range or s.range not in all_classes)
        ]
        field_summary = "、".join(
            f"{s.name}（{_clean_text(s.description or '')[:30]}）" if s.description else s.name
            for s in scalar_slots[:6]
        )
        entity_lines.append(
            f"- **{cls_name}**: {cls_desc[:60] if cls_desc else cls_name}  字段: {field_summary}"
        )

    # Enum summary
    enum_lines: list[str] = []
    for cls_name in order:
        for slot in sv.class_induced_slots(cls_name):
            if slot.range and slot.range in all_enums and slot.range not in _skip:
                pvs = list(sv.get_enum(slot.range).permissible_values.keys())
                enum_lines.append(f"- {slot.name} ({slot.range}): {" | ".join(pvs)}")

    # De-duplicate enum lines
    seen: set[str] = set()
    unique_enum_lines = [x for x in enum_lines if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]

    # ID naming convention from schema description or generic fallback
    id_convention = (
        "{root_lower}_{scenario}_001 / {parent_id}_{child_type}_{N}"
        .replace("{root_lower}", root_class.lower())
        .replace("{child_type}", "child")
    )

    sections = [
        f"你是一个知识图谱提取专家。请调用 {tool_fn_name} 工具，"
        f"从给定文本中提取完整的知识图谱结构。",
    ]
    if schema_desc:
        sections.append(f"\n领域背景: {schema_desc[:200]}")

    sections.append("\n## 实体类型说明")
    sections.extend(entity_lines)

    if unique_enum_lines:
        sections.append("\n## 枚举字段合法值")
        sections.extend(unique_enum_lines)

    sections.append("\n## 提取规则")
    sections.append("1. 保留原文措辞，不要改写或概括")
    sections.append("2. 每个实体生成全局唯一 id，格式: " + id_convention)
    sections.append("3. 保留所有层级关系，不要遗漏")
    sections.append("4. 枚举字段必须使用上方列出的合法值")
    sections.append("5. 若字段说明含【变量占位符】，在话术中保留占位符原文")
    sections.append("\n## 输出格式")
    sections.append(f"调用 {tool_fn_name} 工具，传入合法 JSON，不输出额外文字。")

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Public: FieldSpec / RelationSpec — structured schema metadata
# ---------------------------------------------------------------------------

@dataclass
class FieldSpec:
    """Structured representation of one LinkML slot."""
    name: str
    required: bool
    range: str
    multivalued: bool
    enum_values: list[str]
    description: str
    indexed: bool = False         # x_index: true in YAML
    min_index_chars: int = 0      # x_index_min_chars in YAML


def get_class_field_specs(schema_path: str, class_name: str) -> list[FieldSpec]:
    """Return FieldSpec for every induced slot of *class_name*.

    Custom ``x_index`` / ``x_index_min_chars`` annotations from the raw YAML
    are merged into the returned specs.
    """
    sv, custom = _load_schema(schema_path)
    all_enums = sv.all_enums()
    cls_custom = custom.get(class_name, {})

    specs: list[FieldSpec] = []
    for slot in sv.class_induced_slots(class_name):
        range_name: str = slot.range or "string"
        enum_values: list[str] = (
            list(sv.get_enum(range_name).permissible_values.keys())
            if range_name in all_enums else []
        )
        slot_custom = cls_custom.get(slot.name, {})
        specs.append(FieldSpec(
            name=slot.name,
            required=bool(slot.required or slot.identifier),
            range=range_name,
            multivalued=bool(slot.multivalued),
            enum_values=enum_values,
            description=_clean_text(slot.description or ""),
            indexed=bool(slot_custom.get("x_index", False)),
            min_index_chars=int(slot_custom.get("x_index_min_chars", 0)),
        ))

    return specs


@dataclass
class RelationSpec:
    """A slot whose range is another class (i.e. a graph relation)."""
    field_name: str     # YAML attribute name, e.g. "steps"
    target_class: str   # range class name, e.g. "SOPStep"
    multivalued: bool
    edge_type: str      # derived edge label, e.g. "HAS_STEP"


def field_to_edge_type(field_name: str) -> str:
    """Derive an edge label from a LinkML slot name.

    Convention (preserves existing graph.json edge types):
      used_<plural>  →  USES_<SINGULAR>   e.g. ``used_tools``  → ``USES_TOOL``
      <name>s        →  HAS_<SINGULAR>    e.g. ``steps``       → ``HAS_STEP``
                                               ``sub_rules``    → ``HAS_SUB_RULE``
      <other>        →  <UPPER>           e.g. ``next_step``   → ``NEXT_STEP``
    """
    if field_name.startswith("used_"):
        rest = field_name[5:]
        singular = rest.rstrip("s")
        return f"USES_{singular.upper()}"
    if field_name.endswith("s"):
        singular = field_name.rstrip("s")
        return f"HAS_{singular.upper()}"
    return field_name.upper()


def get_relation_specs(schema_path: str, class_name: str) -> list[RelationSpec]:
    """Return all relation slots for *class_name* (range is a class, not primitive/enum)."""
    sv, _ = _load_schema(schema_path)
    all_classes = sv.all_classes()
    all_enums = sv.all_enums()

    specs: list[RelationSpec] = []
    for slot in sv.class_induced_slots(class_name):
        range_name: str = slot.range or "string"
        if range_name.lower() in _LINKML_TYPES:
            continue
        if range_name in all_enums:
            continue
        if range_name not in all_classes:
            continue
        specs.append(RelationSpec(
            field_name=slot.name,
            target_class=range_name,
            multivalued=bool(slot.multivalued),
            edge_type=field_to_edge_type(slot.name),
        ))

    return specs


def get_enum_values(schema_path: str, enum_name: str) -> list[str]:
    """Return permissible values for a named enum in the schema."""
    sv, _ = _load_schema(schema_path)
    enum_def = sv.get_enum(enum_name)
    if enum_def is None:
        return []
    return list(enum_def.permissible_values.keys())


def get_all_class_names(schema_path: str) -> list[str]:
    """Return all class names defined in the schema."""
    sv, _ = _load_schema(schema_path)
    return list(sv.all_classes().keys())
