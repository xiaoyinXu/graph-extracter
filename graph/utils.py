"""
Utility functions for the graph package.
"""
from __future__ import annotations

from collections import defaultdict


def print_graph_topology(compiled_graph, name: str = "Graph") -> None:
    """
    Print the topology of a compiled LangGraph graph.

    Reads node names and edges from the underlying StateGraph builder so that
    conditional edges added via add_conditional_edges() are fully visible — the
    default get_graph() API omits them in langgraph 0.5.x.

    Args:
        compiled_graph: A compiled LangGraph (result of StateGraph.compile()).
        name: Human-readable label printed as the header.
    """
    b = compiled_graph.builder

    # -- nodes ----------------------------------------------------------------
    # builder.nodes only has user-defined nodes; prepend/__end__ are implicit
    user_nodes: list[str] = list(b.nodes.keys())
    all_nodes = ["__start__"] + user_nodes + ["__end__"]

    # -- fixed edges ----------------------------------------------------------
    fixed_edges: set[tuple[str, str]] = set(b.edges)  # set of (src, tgt)

    # -- conditional edges (branches) ----------------------------------------
    # b.branches: dict[node_name → dict[fn_name → BranchSpec]]
    # BranchSpec.ends: None when no explicit mapping dict was passed
    conditional: dict[str, list[str]] = defaultdict(list)  # src → [fn_names]
    for src_node, fn_map in (b.branches or {}).items():
        for fn_name in fn_map:
            conditional[src_node].append(fn_name)

    # -- render ---------------------------------------------------------------
    sep = "─" * (len(name) + 14)
    print(f"\n┌{sep}┐")
    print(f"│  Graph: {name:<{len(sep) - 10}}│")
    print(f"└{sep}┘")

    print(f"Nodes ({len(all_nodes)}): {' → '.join(all_nodes)}\n")

    print("Edges:")
    # normalise labels
    src_w = max(len(s) for s, _ in fixed_edges | {(s, '') for s in conditional}) if (fixed_edges or conditional) else 12
    src_w = max(src_w, 12)

    def _edge(src: str, tgt: str, label: str) -> None:
        tgt_display = "__end__" if tgt == "__end__" else tgt
        print(f"  {src:<{src_w}}  ──►  {tgt_display:<20}  {label}")

    # emit fixed edges in a sensible order (START first, END last)
    ordered = sorted(
        fixed_edges,
        key=lambda e: (0 if e[0] == "__start__" else (2 if e[1] == "__end__" else 1), e[0]),
    )
    emitted = set()
    for src, tgt in ordered:
        _edge(src, tgt, "[fixed]")
        emitted.add(src)

    # emit conditional branches
    for src_node, fn_names in sorted(conditional.items()):
        for fn_name in fn_names:
            _edge(src_node, "(dynamic)", f"[conditional: {fn_name}]")

    print()
