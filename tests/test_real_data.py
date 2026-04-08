"""
Integration-style unit tests that use the actual data/graph.json file.

Graph structure tested here:
  1 SOP (未到站-常规催派送)
  3 SOPSteps  (step_1 → step_2 → step_3  via NEXT_STEP)
  9 SOPRules  (3 in step_1, 5 in step_2, 1 in step_3)
  7 SOPSubRules
  4 Tools     (tool_query_001, tool_notify_001, tool_groupchat_001, tool_escalate_001)
  ─────────────────────────────────
  24 nodes, 33 edges total

All tests patch _create_embeddings → FakeEmbeddings so no OpenAI API calls
are made.  Semantic correctness of vector search is NOT verified (that
requires real embeddings); traversal logic is tested via mocked hits.
"""
from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import pytest

from graph.models import KnowledgeGraph
from graph.storage import GraphStore, SubgraphResult

# ---------------------------------------------------------------------------
# Well-known IDs from data/graph.json
# ---------------------------------------------------------------------------

SOP = "sop_delivery_urge_normal_001"

STEP_1 = SOP + "_step_1"
STEP_2 = SOP + "_step_2"
STEP_3 = SOP + "_step_3"

RULE_1_1 = STEP_1 + "_rule_1"
RULE_1_2 = STEP_1 + "_rule_2"
RULE_1_3 = STEP_1 + "_rule_3"

RULE_2_1 = STEP_2 + "_rule_1"
RULE_2_2 = STEP_2 + "_rule_2"
RULE_2_3 = STEP_2 + "_rule_3"
RULE_2_4 = STEP_2 + "_rule_4"
RULE_2_5 = STEP_2 + "_rule_5"

RULE_3_1 = STEP_3 + "_rule_1"

SUBRULE_1_1_1 = RULE_1_1 + "_subrule_1"
SUBRULE_1_1_2 = RULE_1_1 + "_subrule_2"
SUBRULE_1_2_1 = RULE_1_2 + "_subrule_1"
SUBRULE_1_2_2 = RULE_1_2 + "_subrule_2"
SUBRULE_2_1_1 = RULE_2_1 + "_subrule_1"
SUBRULE_2_1_2 = RULE_2_1 + "_subrule_2"
SUBRULE_2_5_1 = RULE_2_5 + "_subrule_1"

TOOL_QUERY    = "tool_query_001"
TOOL_NOTIFY   = "tool_notify_001"
TOOL_GROUPCHAT = "tool_groupchat_001"
TOOL_ESCALATE = "tool_escalate_001"

TOTAL_NODES = 24
TOTAL_EDGES = 33

# ---------------------------------------------------------------------------
# Module-scoped fixture (load once, reuse across all tests in this file)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_store() -> GraphStore:
    """Load data/graph.json with real embeddings from .env (OpenAI-compatible).

    The vector index is built once per test session using the configured
    EMBEDDING_MODEL / EMBEDDING_API_KEY / EMBEDDING_BASE_URL.
    No patching — this is the full production code path.
    """
    graph_path = Path(__file__).parent.parent / "data" / "graph.json"
    with open(graph_path, encoding="utf-8") as f:
        raw = json.load(f)
    kg = KnowledgeGraph.model_validate(raw)
    return GraphStore.from_kg(kg)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _hit(node_id: str, sop_id: str = SOP, score: float = 0.9) -> dict[str, Any]:
    return {"node_id": node_id, "node_type": "SOPRule",
            "sop_id": sop_id, "score": score, "text": "test"}


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

class TestRealDataLoading:
    def test_node_count(self, real_store: GraphStore) -> None:
        assert real_store.nx_graph.number_of_nodes() == TOTAL_NODES

    def test_edge_count(self, real_store: GraphStore) -> None:
        assert real_store.nx_graph.number_of_edges() == TOTAL_EDGES

    def test_node_type_distribution(self, real_store: GraphStore) -> None:
        types = {}
        for nid in real_store.nx_graph.nodes:
            t = real_store.nx_graph.nodes[nid]["node_type"]
            types[t] = types.get(t, 0) + 1
        assert types["SOP"] == 1
        assert types["SOPStep"] == 3
        assert types["SOPRule"] == 9
        assert types["SOPSubRule"] == 7
        assert types["Tool"] == 4

    def test_node_map_complete(self, real_store: GraphStore) -> None:
        assert len(real_store.node_map) == TOTAL_NODES

    def test_vector_store_built(self, real_store: GraphStore) -> None:
        assert real_store.vector_store is not None

    def test_all_ids_present(self, real_store: GraphStore) -> None:
        expected_ids = [
            SOP, STEP_1, STEP_2, STEP_3,
            RULE_1_1, RULE_1_2, RULE_1_3,
            RULE_2_1, RULE_2_2, RULE_2_3, RULE_2_4, RULE_2_5,
            RULE_3_1,
            SUBRULE_1_1_1, SUBRULE_1_1_2,
            SUBRULE_1_2_1, SUBRULE_1_2_2,
            SUBRULE_2_1_1, SUBRULE_2_1_2,
            SUBRULE_2_5_1,
            TOOL_QUERY, TOOL_NOTIFY, TOOL_GROUPCHAT, TOOL_ESCALATE,
        ]
        for nid in expected_ids:
            assert nid in real_store.nx_graph, f"Missing node: {nid}"


# ---------------------------------------------------------------------------
# Node data correctness
# ---------------------------------------------------------------------------

class TestRealNodeData:
    def test_sop_name(self, real_store: GraphStore) -> None:
        node = real_store.get_node(SOP)
        assert node is not None
        assert node["name"] == "未到站-常规催派送"

    def test_sop_issue_type(self, real_store: GraphStore) -> None:
        node = real_store.get_node(SOP)
        assert node["issue_type"] == "DELIVERY_URGE"

    def test_sop_trigger_samples_count(self, real_store: GraphStore) -> None:
        node = real_store.get_node(SOP)
        assert len(node["trigger_samples"]) == 11

    def test_sop_trigger_sample_content(self, real_store: GraphStore) -> None:
        node = real_store.get_node(SOP)
        samples = node["trigger_samples"]
        assert "几点能到" in samples
        assert "今天能不能到" in samples

    def test_step1_goal(self, real_store: GraphStore) -> None:
        node = real_store.get_node(STEP_1)
        assert node is not None
        assert "路由状态" in node["goal"]

    def test_step2_goal(self, real_store: GraphStore) -> None:
        node = real_store.get_node(STEP_2)
        assert "催促站点" in node["goal"] or "坐席协助" in node["goal"]

    def test_tool_query_name(self, real_store: GraphStore) -> None:
        node = real_store.get_node(TOOL_QUERY)
        assert node["name"] == "查路由"

    def test_tool_notify_name(self, real_store: GraphStore) -> None:
        node = real_store.get_node(TOOL_NOTIFY)
        assert node["name"] == "工单新"

    def test_tool_escalate_name(self, real_store: GraphStore) -> None:
        node = real_store.get_node(TOOL_ESCALATE)
        assert node["name"] == "升级处理"

    def test_rule1_1_condition(self, real_store: GraphStore) -> None:
        node = real_store.get_node(RULE_1_1)
        assert "路由位置" in node["condition"] or "询问包裹" in node["condition"]

    def test_subrule_condition(self, real_store: GraphStore) -> None:
        node = real_store.get_node(SUBRULE_1_1_1)
        assert node is not None
        assert "配送路线停滞" in node["condition"] or "包裹不动" in node["condition"]


# ---------------------------------------------------------------------------
# Edge structure
# ---------------------------------------------------------------------------

class TestRealEdgeStructure:
    def test_sop_has_three_steps(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        step_edges = [v for _, v, d in G.out_edges(SOP, data=True)
                      if d.get("edge_type") == "HAS_STEP"]
        assert len(step_edges) == 3

    def test_step1_next_step2(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        assert G.has_edge(STEP_1, STEP_2)
        assert G.edges[STEP_1, STEP_2]["edge_type"] == "NEXT_STEP"

    def test_step2_next_step3(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        assert G.has_edge(STEP_2, STEP_3)
        assert G.edges[STEP_2, STEP_3]["edge_type"] == "NEXT_STEP"

    def test_step1_has_three_rules(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        rule_edges = [v for _, v, d in G.out_edges(STEP_1, data=True)
                      if d.get("edge_type") == "HAS_RULE"]
        assert len(rule_edges) == 3

    def test_step2_has_five_rules(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        rule_edges = [v for _, v, d in G.out_edges(STEP_2, data=True)
                      if d.get("edge_type") == "HAS_RULE"]
        assert len(rule_edges) == 5

    def test_step3_has_one_rule(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        rule_edges = [v for _, v, d in G.out_edges(STEP_3, data=True)
                      if d.get("edge_type") == "HAS_RULE"]
        assert len(rule_edges) == 1

    def test_rule1_1_uses_query_tool(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        assert G.has_edge(RULE_1_1, TOOL_QUERY)
        assert G.edges[RULE_1_1, TOOL_QUERY]["edge_type"] == "USES_TOOL"

    def test_rule2_1_uses_notify_and_groupchat(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        assert G.has_edge(RULE_2_1, TOOL_NOTIFY)
        assert G.has_edge(RULE_2_1, TOOL_GROUPCHAT)

    def test_rule3_1_uses_escalate(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        assert G.has_edge(RULE_3_1, TOOL_ESCALATE)

    def test_rule1_1_has_two_subrules(self, real_store: GraphStore) -> None:
        G = real_store.nx_graph
        sr_edges = [v for _, v, d in G.out_edges(RULE_1_1, data=True)
                    if d.get("edge_type") == "HAS_SUB_RULE"]
        assert len(sr_edges) == 2
        assert SUBRULE_1_1_1 in sr_edges
        assert SUBRULE_1_1_2 in sr_edges

    def test_rule2_3_has_no_children(self, real_store: GraphStore) -> None:
        """rule_3 in step_2 has no sub_rules or tools."""
        G = real_store.nx_graph
        assert G.out_degree(RULE_2_3) == 0


# ---------------------------------------------------------------------------
# get_sop_context
# ---------------------------------------------------------------------------

class TestRealGetSopContext:
    def test_three_steps_returned(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        assert len(ctx["steps"]) == 3

    def test_steps_ordered_by_index(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        indices = [s["step"]["step_index"] for s in ctx["steps"]]
        assert indices == sorted(indices)

    def test_step1_has_three_rules(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step1_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_1)
        assert len(step1_ctx["rules"]) == 3

    def test_step2_has_five_rules(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step2_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_2)
        assert len(step2_ctx["rules"]) == 5

    def test_step3_has_one_rule(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step3_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_3)
        assert len(step3_ctx["rules"]) == 1

    def test_rule1_1_has_two_subrules(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step1_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_1)
        rule1_ctx = next(r for r in step1_ctx["rules"] if r["rule"]["id"] == RULE_1_1)
        assert len(rule1_ctx["sub_rules"]) == 2

    def test_rule1_1_has_query_tool(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step1_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_1)
        rule1_ctx = next(r for r in step1_ctx["rules"] if r["rule"]["id"] == RULE_1_1)
        tool_ids = [t["id"] for t in rule1_ctx["tools"]]
        assert TOOL_QUERY in tool_ids

    def test_rule2_1_has_notify_and_groupchat_tools(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step2_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_2)
        rule1_ctx = next(r for r in step2_ctx["rules"] if r["rule"]["id"] == RULE_2_1)
        tool_ids = [t["id"] for t in rule1_ctx["tools"]]
        assert TOOL_NOTIFY in tool_ids
        assert TOOL_GROUPCHAT in tool_ids

    def test_rule3_1_has_escalate_tool(self, real_store: GraphStore) -> None:
        ctx = real_store.get_sop_context(SOP)
        step3_ctx = next(s for s in ctx["steps"] if s["step"]["id"] == STEP_3)
        rule_ctx = step3_ctx["rules"][0]
        tool_ids = [t["id"] for t in rule_ctx["tools"]]
        assert TOOL_ESCALATE in tool_ids


# ---------------------------------------------------------------------------
# get_ancestors
# ---------------------------------------------------------------------------

class TestRealGetAncestors:
    def test_rule1_1_ancestors_contain_sop_and_step(self, real_store: GraphStore) -> None:
        ancestors = real_store.get_ancestors(RULE_1_1)
        ids = [a["id"] for a in ancestors]
        assert SOP in ids
        assert STEP_1 in ids

    def test_subrule_ancestors_contain_rule_step_sop(self, real_store: GraphStore) -> None:
        ancestors = real_store.get_ancestors(SUBRULE_1_1_1)
        ids = [a["id"] for a in ancestors]
        assert SOP in ids
        assert STEP_1 in ids
        assert RULE_1_1 in ids

    def test_sop_has_no_ancestors(self, real_store: GraphStore) -> None:
        assert real_store.get_ancestors(SOP) == []

    def test_ancestors_root_first(self, real_store: GraphStore) -> None:
        """SOP must come before SOPStep in the ancestor list."""
        ancestors = real_store.get_ancestors(RULE_1_1)
        types = [a["node_type"] for a in ancestors]
        assert types.index("SOP") < types.index("SOPStep")

    def test_step_ancestors_contain_sop(self, real_store: GraphStore) -> None:
        ancestors = real_store.get_ancestors(STEP_1)
        assert any(a["id"] == SOP for a in ancestors)


# ---------------------------------------------------------------------------
# get_rule_context
# ---------------------------------------------------------------------------

class TestRealGetRuleContext:
    def test_rule1_1_context_has_correct_node(self, real_store: GraphStore) -> None:
        ctx = real_store.get_rule_context(RULE_1_1, "SOPRule")
        assert ctx["node"]["id"] == RULE_1_1

    def test_rule1_1_siblings_include_rule1_2_and_rule1_3(self, real_store: GraphStore) -> None:
        ctx = real_store.get_rule_context(RULE_1_1, "SOPRule")
        sibling_ids = [s["id"] for s in ctx["siblings"]]
        assert RULE_1_2 in sibling_ids
        assert RULE_1_3 in sibling_ids

    def test_rule1_1_siblings_exclude_self(self, real_store: GraphStore) -> None:
        ctx = real_store.get_rule_context(RULE_1_1, "SOPRule")
        sibling_ids = [s["id"] for s in ctx["siblings"]]
        assert RULE_1_1 not in sibling_ids

    def test_rule1_1_children_are_subrules(self, real_store: GraphStore) -> None:
        ctx = real_store.get_rule_context(RULE_1_1, "SOPRule")
        child_ids = [c["id"] for c in ctx["children"]]
        assert SUBRULE_1_1_1 in child_ids
        assert SUBRULE_1_1_2 in child_ids

    def test_rule2_3_has_no_children(self, real_store: GraphStore) -> None:
        ctx = real_store.get_rule_context(RULE_2_3, "SOPRule")
        assert ctx["children"] == []

    def test_rule2_5_siblings_count(self, real_store: GraphStore) -> None:
        """step_2 has 5 rules, so rule_5 has 4 siblings."""
        ctx = real_store.get_rule_context(RULE_2_5, "SOPRule")
        assert len(ctx["siblings"]) == 4


# ---------------------------------------------------------------------------
# get_connected_subgraph
# ---------------------------------------------------------------------------

class TestRealGetConnectedSubgraph:
    def test_from_sop_root_includes_all_nodes(self, real_store: GraphStore) -> None:
        """Traversing from SOP root must reach every node in the graph."""
        sg = real_store.get_connected_subgraph(SOP)
        assert sg.number_of_nodes() == TOTAL_NODES

    def test_from_step1_excludes_sop_node(self, real_store: GraphStore) -> None:
        """step_1 has no edge back to the SOP root."""
        sg = real_store.get_connected_subgraph(STEP_1)
        assert SOP not in sg.nodes

    def test_from_step1_includes_step2_via_next_step(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph(STEP_1)
        assert STEP_2 in sg.nodes
        assert STEP_3 in sg.nodes

    def test_from_step1_includes_all_downstream_rules(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph(STEP_1)
        for rule_id in [RULE_1_1, RULE_1_2, RULE_1_3,
                        RULE_2_1, RULE_2_2, RULE_2_3, RULE_2_4, RULE_2_5,
                        RULE_3_1]:
            assert rule_id in sg.nodes, f"Expected {rule_id} in subgraph from STEP_1"

    def test_from_step2_excludes_step1_and_sop(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph(STEP_2)
        assert SOP not in sg.nodes
        assert STEP_1 not in sg.nodes

    def test_from_step2_includes_step3_and_its_rule(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph(STEP_2)
        assert STEP_3 in sg.nodes
        assert RULE_3_1 in sg.nodes

    def test_from_leaf_rule_only_rule_and_tool(self, real_store: GraphStore) -> None:
        """rule_1_3 has only one outgoing edge: USES_TOOL → tool_query_001."""
        sg = real_store.get_connected_subgraph(RULE_1_3)
        assert sg.number_of_nodes() == 2
        assert RULE_1_3 in sg.nodes
        assert TOOL_QUERY in sg.nodes

    def test_from_rule2_3_is_isolated(self, real_store: GraphStore) -> None:
        """rule_2_3 has no outgoing edges."""
        sg = real_store.get_connected_subgraph(RULE_2_3)
        assert sg.number_of_nodes() == 1
        assert RULE_2_3 in sg.nodes

    def test_from_tool_node_is_single_node(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph(TOOL_QUERY)
        assert sg.number_of_nodes() == 1

    def test_missing_node_returns_empty(self, real_store: GraphStore) -> None:
        sg = real_store.get_connected_subgraph("nonexistent_id")
        assert sg.number_of_nodes() == 0


# ---------------------------------------------------------------------------
# Mermaid output
# ---------------------------------------------------------------------------

class TestRealMermaidOutput:
    def test_mermaid_from_sop_starts_with_graph_td(self, real_store: GraphStore) -> None:
        from graph.utils import subgraph_to_mermaid
        sg = real_store.get_connected_subgraph(SOP)
        diagram = subgraph_to_mermaid(sg, real_store.node_map)
        assert diagram.startswith("graph TD")

    def test_mermaid_from_sop_contains_all_node_ids(self, real_store: GraphStore) -> None:
        from graph.utils import subgraph_to_mermaid
        sg = real_store.get_connected_subgraph(SOP)
        diagram = subgraph_to_mermaid(sg, real_store.node_map)
        # All node IDs should appear in the Mermaid definition
        for nid in [SOP, STEP_1, STEP_2, STEP_3, RULE_1_1, TOOL_QUERY, TOOL_ESCALATE]:
            assert nid in diagram, f"Node {nid!r} missing from Mermaid output"

    def test_mermaid_from_sop_contains_sop_label(self, real_store: GraphStore) -> None:
        from graph.utils import subgraph_to_mermaid
        sg = real_store.get_connected_subgraph(SOP)
        diagram = subgraph_to_mermaid(sg, real_store.node_map)
        assert "未到站-常规催派送" in diagram

    def test_mermaid_from_single_rule_is_minimal(self, real_store: GraphStore) -> None:
        from graph.utils import subgraph_to_mermaid
        sg = real_store.get_connected_subgraph(RULE_2_3)
        diagram = subgraph_to_mermaid(sg, real_store.node_map)
        assert "graph TD" in diagram
        # Only one node → no arrows expected
        assert "-->" not in diagram

    def test_mermaid_from_step1_printed(self, real_store: GraphStore) -> None:
        """Visual inspection via pytest -s."""
        from graph.utils import subgraph_to_mermaid
        sg = real_store.get_connected_subgraph(STEP_1)
        diagram = subgraph_to_mermaid(sg, real_store.node_map)
        print(f"\n--- Mermaid: subgraph from {STEP_1} ---\n{diagram}\n")
        assert "graph TD" in diagram


# ---------------------------------------------------------------------------
# search_and_traverse — full pipeline (real FakeEmbeddings, no mock)
# ---------------------------------------------------------------------------

class TestRealSearchAndTraverse:
    """End-to-end pipeline tests: real vector search → traversal.

    similarity_search is NOT mocked here.  The FakeEmbeddings-backed vector
    store drives the search; results are semantically meaningless but the
    pipeline (search → dedup → traverse → mermaid) runs in full.

    Assertions target structural correctness only, not semantic relevance.
    """

    def test_returns_subgraph_result_type(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("包裹在哪里", k=3)
        assert isinstance(result, SubgraphResult)

    def test_hits_is_nonempty_list(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("快递慢", k=5)
        assert isinstance(result.hits, list)
        assert len(result.hits) > 0

    def test_hits_have_required_keys(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("今天能到吗", k=3)
        for hit in result.hits:
            for key in ("node_id", "node_type", "sop_id", "score", "text"):
                assert key in hit, f"Missing key {key!r} in hit: {hit}"

    def test_hit_node_ids_exist_in_graph(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("路由查询", k=5)
        for hit in result.hits:
            assert hit["node_id"] in real_store.nx_graph, \
                f"node_id {hit['node_id']!r} from search not in graph"

    def test_start_node_ids_exist_in_graph(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("配送时间", k=3)
        for nid in result.start_node_ids:
            assert nid in real_store.nx_graph, \
                f"start_node_id {nid!r} not in graph"

    def test_subgraph_contains_all_start_nodes(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("包裹位置", k=3)
        for nid in result.start_node_ids:
            assert nid in result.subgraph.nodes

    def test_subgraph_edges_are_subset_of_original(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("催派送", k=3)
        G = real_store.nx_graph
        for u, v in result.subgraph.edges():
            assert G.has_edge(u, v), f"Spurious edge ({u}, {v}) in subgraph"

    def test_mermaid_starts_with_graph_td(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("包裹不动了", k=3)
        assert result.mermaid.startswith("graph TD")

    def test_start_node_ids_are_deduplicated(self, real_store: GraphStore) -> None:
        """k=10 may retrieve multiple docs for the same node; IDs must be unique."""
        result = real_store.search_and_traverse("配送", k=10)
        assert len(result.start_node_ids) == len(set(result.start_node_ids))

    def test_k_limits_hit_count(self, real_store: GraphStore) -> None:
        result = real_store.search_and_traverse("包裹", k=2)
        assert len(result.hits) <= 2

    def test_traverse_from_root_resolves_to_sop(self, real_store: GraphStore) -> None:
        """Every indexed node has sop_id=SOP, so traverse_from_root always
        resolves to the SOP root and returns the full 24-node graph."""
        result = real_store.search_and_traverse(
            "包裹不动了", k=3, traverse_from_root=True)
        assert SOP in result.start_node_ids
        assert result.subgraph.number_of_nodes() == TOTAL_NODES

    def test_traverse_from_root_subgraph_ge_default(self, real_store: GraphStore) -> None:
        """Root traversal must produce ≥ nodes than matched-node traversal."""
        result_default = real_store.search_and_traverse("今天能到吗", k=2)
        result_root = real_store.search_and_traverse(
            "今天能到吗", k=2, traverse_from_root=True)
        assert result_root.subgraph.number_of_nodes() >= \
               result_default.subgraph.number_of_nodes()

    def test_printed_mermaid(self, real_store: GraphStore) -> None:
        """Visual inspection via pytest -s."""
        result = real_store.search_and_traverse(
            "包裹不动了催派送", k=5, traverse_from_root=True)
        print(f"\n--- search_and_traverse (real embeddings, traverse_from_root=True) ---\n"
              f"{result.mermaid}\n")
        assert "graph TD" in result.mermaid

    # ------------------------------------------------------------------
    # Semantic correctness (real embeddings required)
    # ------------------------------------------------------------------

    def test_semantic_baoguobудongo_hits_sop(self, real_store: GraphStore) -> None:
        """'包裹不动了' is a verbatim SOP trigger sample → SOP node should appear in hits."""
        result = real_store.search_and_traverse("包裹不动了", k=5)
        hit_node_ids = [h["node_id"] for h in result.hits]
        # The SOP trigger sample "怎么包裹不动了" and subrule condition both
        # contain this phrase — at least one hit must map to the SOP or subrule
        assert any(
            nid in (SOP, SUBRULE_1_1_1, SUBRULE_1_1_2)
            for nid in hit_node_ids
        ), f"Expected SOP/subrule in hits, got: {hit_node_ids}"

    def test_semantic_jidian_neng_dao_hits_sop(self, real_store: GraphStore) -> None:
        """'几点能到' is a verbatim SOP trigger sample → top hit should be SOP."""
        result = real_store.search_and_traverse("几点能到", k=3)
        hit_node_ids = [h["node_id"] for h in result.hits]
        assert SOP in hit_node_ids, \
            f"Expected SOP in top-3 hits for '几点能到', got: {hit_node_ids}"

    def test_semantic_traverse_from_root_always_includes_sop(self, real_store: GraphStore) -> None:
        """traverse_from_root=True: every indexed node has sop_id=SOP, so
        the merged subgraph must always contain the full 24-node tree."""
        for query in ["包裹不动了", "几点能到", "催派送", "修改地址"]:
            result = real_store.search_and_traverse(query, k=3, traverse_from_root=True)
            assert result.subgraph.number_of_nodes() == TOTAL_NODES, \
                f"Expected {TOTAL_NODES} nodes for query {query!r}, " \
                f"got {result.subgraph.number_of_nodes()}"

    def test_semantic_escalate_query_hits_step3_rule(self, real_store: GraphStore) -> None:
        """'升级处理' is the name of tool_escalate_001; the semantically closest
        indexed text should pull in step_3's rule which uses that tool."""
        result = real_store.search_and_traverse("升级处理投诉", k=5)
        hit_node_ids = [h["node_id"] for h in result.hits]
        # step_3 rule_1 condition explicitly mentions escalation
        assert any(
            nid in (RULE_3_1, STEP_3)
            for nid in hit_node_ids
        ), f"Expected RULE_3_1 or STEP_3 in hits for escalation query, got: {hit_node_ids}"


# ---------------------------------------------------------------------------
# Vector index docs (smoke tests — not semantic correctness)
# ---------------------------------------------------------------------------

class TestRealVectorIndex:
    def test_similarity_search_returns_results(self, real_store: GraphStore) -> None:
        results = real_store.similarity_search("包裹在哪里", k=5)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_similarity_search_result_structure(self, real_store: GraphStore) -> None:
        results = real_store.similarity_search("包裹不动了", k=3)
        for hit in results:
            assert "node_id" in hit
            assert "node_type" in hit
            assert "sop_id" in hit
            assert "score" in hit
            assert "text" in hit

    def test_similarity_search_node_ids_exist_in_graph(self, real_store: GraphStore) -> None:
        results = real_store.similarity_search("催派送", k=5)
        for hit in results:
            assert hit["node_id"] in real_store.nx_graph, \
                f"node_id {hit['node_id']!r} from vector search not in graph"

    def test_similarity_search_k_limits_results(self, real_store: GraphStore) -> None:
        results = real_store.similarity_search("快递", k=2)
        assert len(results) <= 2
