"""
Entry point: build knowledge graph from SOP text, then run example queries.

Usage:
    uv run python main.py build       # extract + build + save graph
    uv run python main.py query       # interactive query loop
    uv run python main.py demo        # build then run demo queries (default)
"""
from __future__ import annotations

import sys
from pathlib import Path

GRAPH_PATH = "data/graph.json"
DOCS_PATH = "docs/流程数据.txt"


# ---------------------------------------------------------------------------
# Build phase
# ---------------------------------------------------------------------------

def build() -> bool:
    from graph.extractor import extract_and_build
    from graph.storage import GraphStore

    print("=" * 60)
    print("[build] Loading SOP text...")
    text = Path(DOCS_PATH).read_text(encoding="utf-8")

    print("[build] Extracting knowledge graph with LLM...")
    kg = extract_and_build(text)

    if not kg.nodes:
        print("[build] ERROR: extraction returned empty graph. Check LLM errors above.")
        return False

    print(f"[build] Extracted {len(kg.nodes)} nodes, {len(kg.edges)} edges")

    print("[build] Building store and vector index...")
    store = GraphStore.from_kg(kg)
    store.save(GRAPH_PATH)

    # show node type summary
    type_counts: dict[str, int] = {}
    for node in kg.nodes:
        type_counts[node.node_type] = type_counts.get(node.node_type, 0) + 1
    print("\n[build] Node summary:")
    for ntype, cnt in sorted(type_counts.items()):
        print(f"  {ntype}: {cnt}")
    return True


# ---------------------------------------------------------------------------
# Query phase
# ---------------------------------------------------------------------------

DEMO_QUERIES = [
    "用户催派送，询问包裹到哪里了，怎么处理？",
    "用户说等不到了，东西很急，能不能让快递送一下，怎么回复？",
    "用户问可以付费加急配送吗，怎么处理？",
    "用户想修改地址，能帮他改吗？",
    "用户抱怨物流太慢，要投诉，怎么回复？",
]


def query_loop() -> None:
    from graph.retriever import KnowledgeGraphRetriever

    if not Path(GRAPH_PATH).exists():
        print(f"[query] Graph not found at {GRAPH_PATH}. Run `python main.py build` first.")
        sys.exit(1)

    retriever = KnowledgeGraphRetriever(GRAPH_PATH)
    print("\n[query] Interactive mode. Enter a question (empty line to quit).\n")
    while True:
        try:
            q = input("问题> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not q:
            break
        answer = retriever.query(q, verbose=True)
        print(f"\n答案:\n{answer}\n{'=' * 60}\n")


def demo() -> None:
    from graph.retriever import KnowledgeGraphRetriever

    if not Path(GRAPH_PATH).exists():
        print("[demo] Graph not found, building first...")
        if not build():
            return

    retriever = KnowledgeGraphRetriever(GRAPH_PATH)
    print("\n" + "=" * 60)
    print("DEMO QUERIES")
    print("=" * 60)

    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"\n[Q{i}] {q}")
        print("-" * 50)
        answer = retriever.query(q, verbose=False)
        print(answer)
        print()


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if cmd == "build":
        build()
    elif cmd == "query":
        query_loop()
    elif cmd == "demo":
        demo()
    else:
        print(f"Unknown command: {cmd}. Use: build | query | demo")
        sys.exit(1)
