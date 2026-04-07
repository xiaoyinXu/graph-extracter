"""
Unit tests for graph/schema_loader.py — dynamic OpenAI tool schema builder.
"""
import pytest

from graph.schema_loader import build_tool_from_schema, _resolve_range, _class_to_json_schema

SCHEMA_PATH = "schema/customer_service.yaml"


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

    def _find_issue_type(self, props):
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

        def find(p, target):
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

    def _all_property_keys(self, obj):
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
        """
        If a class references itself (e.g. SOPStep.next_step → SOPStep),
        the circular reference must be rendered as {"type": "string"}, not recurse.
        """
        classes = {
            "Node": {
                "attributes": {
                    "id": {"range": "string", "identifier": True},
                    "child": {"range": "Node"},  # self-reference
                }
            }
        }
        enums: dict = {}
        schema = _class_to_json_schema(
            "Node", classes, enums,
            default_range="string",
            visiting=frozenset(),
            skip_fields=frozenset(),
            max_depth=10,
            depth=0,
        )
        child_prop = schema["properties"]["child"]
        # Should be a string (circular ref fallback), not another object
        assert child_prop["type"] == "string"

    def test_no_infinite_recursion(self):
        """Deep mutual refs must not raise RecursionError."""
        classes = {
            "A": {"attributes": {"b": {"range": "B"}}},
            "B": {"attributes": {"a": {"range": "A"}}},
        }
        # Should complete without RecursionError
        schema = _class_to_json_schema(
            "A", classes, {}, "string",
            frozenset(), frozenset(), max_depth=5, depth=0,
        )
        assert schema["type"] == "object"


# ---------------------------------------------------------------------------
# Primitive type resolution
# ---------------------------------------------------------------------------

class TestResolveRange:
    def _call(self, range_type: str, classes=None, enums=None):
        return _resolve_range(
            range_type,
            classes or {},
            enums or {},
            default_range="string",
            visiting=frozenset(),
            skip_fields=frozenset(),
            max_depth=5,
            depth=0,
        )

    def test_string(self):
        assert self._call("string") == {"type": "string"}

    def test_integer(self):
        assert self._call("integer") == {"type": "integer"}

    def test_float(self):
        assert self._call("float") == {"type": "number"}

    def test_boolean(self):
        assert self._call("boolean") == {"type": "boolean"}

    def test_case_insensitive(self):
        assert self._call("String") == {"type": "string"}
        assert self._call("INTEGER") == {"type": "integer"}

    def test_unknown_range_falls_back_to_string(self):
        result = self._call("SomeUnknownType")
        assert result == {"type": "string"}

    def test_enum_range(self):
        enums = {"Color": {"permissible_values": {"RED": {}, "GREEN": {}, "BLUE": {}}}}
        result = self._call("Color", enums=enums)
        assert result["type"] == "string"
        assert set(result["enum"]) == {"RED", "GREEN", "BLUE"}
