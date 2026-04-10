"""
Unit tests for graph/schema_loader.py — dynamic OpenAI tool schema builder.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
from typing import Any, Optional

import pytest

from graph.schema_loader import build_tool_from_schema, _SCHEMA_CACHE

SCHEMA_PATH = "schema/customer_service.yaml"


# ---------------------------------------------------------------------------
# Helper: build a tool from a minimal inline YAML schema
# ---------------------------------------------------------------------------

def _tool_from_yaml(yaml_src: str, root: str = "Root", list_field: str = "items") -> dict:
    """Write *yaml_src* to a temp file and build a tool schema from it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        fh.write(textwrap.dedent(yaml_src))
        path = fh.name
    try:
        _SCHEMA_CACHE.pop(path, None)
        return build_tool_from_schema(path, root_class=root, list_field_name=list_field)
    finally:
        os.unlink(path)
        _SCHEMA_CACHE.pop(path, None)

# ---------------------------------------------------------------------------
# Top-level tool structure
# ---------------------------------------------------------------------------

class TestBuildToolFromSchema:
    @pytest.fixture(scope="class")
    def tool(self):
        return build_tool_from_schema(SCHEMA_PATH)

    def test_tool_type_is_function(self, tool):
        assert tool["type"] == "function"

    def test_tool_has_function_key(self, tool):
        assert "function" in tool

    def test_function_name_derived_from_schema(self, tool):
        # schema name = "customer_service" → function name = "extract_customer_service"
        assert tool["function"]["name"] == "extract_customer_service"

    def test_function_has_description(self, tool):
        assert tool["function"].get("description")

    def test_parameters_is_object(self, tool):
        params = tool["function"]["parameters"]
        assert params["type"] == "object"

    def test_top_level_has_sops_array(self, tool):
        """ExtractionOutput / root class has a 'sops' array property."""
        props = tool["function"]["parameters"]["properties"]
        assert "sops" in props
        assert props["sops"]["type"] == "array"

    def test_sops_items_is_sop_object(self, tool):
        sops_schema = tool["function"]["parameters"]["properties"]["sops"]
        items = sops_schema["items"]
        assert items["type"] == "object"
        # SOP has at least these fields
        assert "id" in items["properties"]
        assert "name" in items["properties"]
        assert "issue_type" in items["properties"]


# ---------------------------------------------------------------------------
# Enum expansion
# ---------------------------------------------------------------------------

class TestEnumExpansion:
    @pytest.fixture(scope="class")
    def tool(self):
        return build_tool_from_schema(SCHEMA_PATH)

    def _find_issue_type(self, props: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Recursively search for 'issue_type' in nested properties."""
        for key, val in props.items():
            if key == "issue_type":
                return val
            if isinstance(val, dict):
                if "properties" in val:
                    result = self._find_issue_type(val["properties"])
                    if result:
                        return result
                if "items" in val and isinstance(val["items"], dict):
                    if "properties" in val["items"]:
                        result = self._find_issue_type(val["items"]["properties"])
                        if result:
                            return result
        return None

    def test_issue_type_is_enum_string(self, tool):
        props = tool["function"]["parameters"]["properties"]
        issue_type = self._find_issue_type(props)
        assert issue_type is not None, "issue_type field not found in schema"
        assert issue_type["type"] == "string"
        assert "enum" in issue_type
        assert "DELIVERY_URGE" in issue_type["enum"]
        assert "REFUND" in issue_type["enum"]

    def test_tool_type_is_enum_string(self, tool):
        props = tool["function"]["parameters"]["properties"]

        def find(p: dict[str, Any], target: str) -> Optional[dict[str, Any]]:
            for k, v in p.items():
                if k == target:
                    return v
                if isinstance(v, dict):
                    for sub in [v.get("properties", {}), v.get("items", {}).get("properties", {}) if isinstance(v.get("items"), dict) else {}]:
                        r = find(sub, target)
                        if r:
                            return r
            return None

        tool_type = find(props, "tool_type")
        assert tool_type is not None
        assert "enum" in tool_type
        assert "TICKET" in tool_type["enum"]


# ---------------------------------------------------------------------------
# Skip fields
# ---------------------------------------------------------------------------

class TestSkipFields:
    @pytest.fixture(scope="class")
    def tool(self):
        return build_tool_from_schema(SCHEMA_PATH)

    def _all_property_keys(self, obj: Any) -> set[str]:
        """Collect every property key name in the entire schema recursively."""
        keys = set()
        if not isinstance(obj, dict):
            return keys
        for k, v in obj.items():
            if k == "properties" and isinstance(v, dict):
                keys.update(v.keys())
            keys.update(self._all_property_keys(v))
        return keys

    def test_execution_fields_excluded(self, tool):
        all_keys = self._all_property_keys(tool)
        for field in ("mcp_endpoint", "health_status", "quality_score", "success_rate"):
            assert field not in all_keys, f"skip field '{field}' should not appear in schema"


# ---------------------------------------------------------------------------
# Multivalued → array
# ---------------------------------------------------------------------------

class TestMultivalued:
    @pytest.fixture(scope="class")
    def tool(self):
        return build_tool_from_schema(SCHEMA_PATH)

    def test_trigger_samples_is_array(self, tool):
        sops_items = tool["function"]["parameters"]["properties"]["sops"]["items"]
        trigger = sops_items["properties"].get("trigger_samples")
        assert trigger is not None
        assert trigger["type"] == "array"

    def test_steps_is_array(self, tool):
        sops_items = tool["function"]["parameters"]["properties"]["sops"]["items"]
        steps = sops_items["properties"].get("steps")
        assert steps is not None
        assert steps["type"] == "array"


# ---------------------------------------------------------------------------
# Circular reference detection
# ---------------------------------------------------------------------------

class TestCircularReference:
    def test_circular_ref_rendered_as_string(self):
        """Self-referencing class must be rendered as {"type": "string"}, not recurse."""
        tool = _tool_from_yaml("""
            id: test
            name: test
            default_range: string
            classes:
              Root:
                attributes:
                  id:
                    range: string
                    identifier: true
                  child:
                    range: Root
        """, root="Root")
        root_schema = tool["function"]["parameters"]["properties"]["items"]["items"]
        child_prop = root_schema["properties"]["child"]
        assert child_prop["type"] == "string"

    def test_no_infinite_recursion(self):
        """Mutual class references must not raise RecursionError."""
        tool = _tool_from_yaml("""
            id: test
            name: test
            default_range: string
            classes:
              Root:
                attributes:
                  id:
                    range: string
                    identifier: true
                  child:
                    range: Child
              Child:
                attributes:
                  id:
                    range: string
                    identifier: true
                  parent:
                    range: Root
        """, root="Root")
        assert tool["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------------------
# Primitive type resolution
# ---------------------------------------------------------------------------

class TestResolveRange:
    """Tests that LinkML primitive ranges are mapped to the correct JSON Schema types."""

    def _field_schema(self, range_type: str) -> dict[str, Any]:
        """Build a minimal one-field schema and return that field's JSON Schema."""
        tool = _tool_from_yaml(f"""
            id: test
            name: test
            default_range: string
            classes:
              Root:
                attributes:
                  myfield:
                    range: {range_type}
        """, root="Root")
        return tool["function"]["parameters"]["properties"]["items"]["items"]["properties"]["myfield"]

    def test_string(self):
        assert self._field_schema("string") == {"type": "string"}

    def test_integer(self):
        assert self._field_schema("integer") == {"type": "integer"}

    def test_float(self):
        assert self._field_schema("float") == {"type": "number"}

    def test_boolean(self):
        assert self._field_schema("boolean") == {"type": "boolean"}

    def test_case_insensitive(self):
        assert self._field_schema("String") == {"type": "string"}
        assert self._field_schema("INTEGER") == {"type": "integer"}

    def test_unknown_range_falls_back_to_string(self):
        result = self._field_schema("SomeUnknownType")
        assert result == {"type": "string"}

    def test_enum_range(self):
        tool = _tool_from_yaml("""
            id: test
            name: test
            default_range: string
            classes:
              Root:
                attributes:
                  color:
                    range: Color
            enums:
              Color:
                permissible_values:
                  RED: {}
                  GREEN: {}
                  BLUE: {}
        """, root="Root")
        color = tool["function"]["parameters"]["properties"]["items"]["items"]["properties"]["color"]
        assert color["type"] == "string"
        assert set(color["enum"]) == {"RED", "GREEN", "BLUE"}
