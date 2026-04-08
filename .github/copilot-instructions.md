# Copilot Instructions

## Project Overview

`graph-extracter` is a Python knowledge-graph extraction and retrieval system for customer-service SOPs (Standard Operating Procedures). It:

1. **Extracts** structured knowledge graphs from raw SOP text using an LLM (LangGraph pipeline).
2. **Stores** the graph as `data/graph.json` (NetworkX + Pydantic) and a vector index in Elasticsearch.
3. **Retrieves** relevant SOP context via vector search + subgraph traversal to answer natural-language queries.

## Tech Stack

- **Python ≥ 3.12**, package manager: `uv`
- **Pydantic v2** — all data models (`graph/models.py`)
- **LangGraph** — extraction pipeline (`graph/extractor.py`) and retrieval pipeline (`graph/retriever.py`)
- **LangChain / langchain-openai** — LLM calls and embeddings (`text-embedding-ada-002-2`, 1536 dims)
- **Elasticsearch 8.x** — vector store (`dense_vector`, cosine, 1536 dims) via `langchain-elasticsearch`
- **NetworkX** — in-memory directed graph for traversal
- **LinkML YAML** — schema source of truth (`schema/customer_service.yaml`)
- **pytest** — test runner; `pytest-mock` for patching

## Repository Layout

```
graph/
  models.py        # Pydantic models: KnowledgeGraph, GraphNode, GraphEdge, ExtractionOutput, enums
  extractor.py     # LangGraph extraction pipeline: extract_sops → build_graph → validate_graph → save_graph
  storage.py       # GraphStore: load/save JSON + build Elasticsearch vector index
  retriever.py     # LangGraph retrieval pipeline: search_nodes → expand_context → generate_answer
  schema_loader.py # LinkML YAML → JSON Schema + FieldSpec for validation/indexing
  utils.py         # Mermaid diagram generation, topology printing

schema/
  customer_service.yaml  # LinkML schema — single source of truth for all entity types and enums

data/
  graph.json       # Persisted KnowledgeGraph (nodes + edges)
  *.txt            # Source SOP documents

tests/
  conftest.py              # Shared fixtures: FakeEmbeddings (8-dim), minimal_kg, minimal_extraction_output
  test_extractor.py        # Unit tests for extraction + validation logic
  test_storage.py          # Unit tests for GraphStore load/save/search
  test_subgraph.py         # Unit tests for subgraph traversal
  test_search_traverse.py  # Unit tests for vector search + traversal pipeline
  test_models.py           # Unit tests for Pydantic models
  test_schema_loader.py    # Unit tests for schema_loader
  test_utils.py            # Unit tests for utils
  test_real_data.py        # Integration tests using real data/graph.json + real ES + real embeddings API

main.py  # CLI entry point: `uv run python main.py [build|query|demo]`
```

## Core Data Model

```
KnowledgeGraph
  nodes: list[GraphNode]   # id, node_type (SOP/SOPStep/SOPRule/SOPSubRule/Tool), data: dict
  edges: list[GraphEdge]   # source, target, edge_type, metadata: dict

Edge types: HAS_STEP, NEXT_STEP, HAS_RULE, HAS_SUB_RULE, USES_TOOL
```

Node hierarchy: `SOP → SOPStep → SOPRule → SOPSubRule` (+ `SOPRule → Tool` via `USES_TOOL`)

## Schema-Driven Design

- **`schema/customer_service.yaml`** is the single source of truth. Do **not** hardcode class names (`SOP`, `SOPStep`, etc.) or field names in business logic.
- `schema_loader.py` exposes:
  - `build_tool_from_schema()` — builds OpenAI function-calling schema dynamically
  - `get_class_field_specs(schema_path, class_name) → list[FieldSpec]` — returns per-attribute specs (name, required, range, multivalued, enum_values, description, indexed)
- The `indexed: bool` field on `FieldSpec` (driven by `x_index: true` in YAML) controls which fields are included in the ES vector index — **no hardcoded field names in `storage.py`**.
- The extraction prompt and tool schema are rebuilt from YAML on every run via `SCHEMA_PATH` env var.

## Key Conventions

### Type hints
All public functions and methods **must** have full type hints. Use `from __future__ import annotations` at the top of every module.

### Pydantic v2
Use `model_validate()` (not `parse_obj()`), `model_dump_json()` (not `.json()`).

### Testing
- **Unit tests** must patch both `_create_embeddings` AND `_create_vector_store` in `graph.storage` to avoid hitting real ES or the embeddings API:
  ```python
  @pytest.fixture
  def make_store(fake_embeddings, minimal_kg):
      with patch("graph.storage._create_embeddings", return_value=fake_embeddings), \
           patch("graph.storage._create_vector_store", side_effect=lambda docs, emb: InMemoryVectorStore.from_documents(docs, emb)):
          yield GraphStore.from_kg(minimal_kg)
  ```
- **`test_real_data.py`** is the only file that uses real ES + real embeddings — do NOT add mocks there.
- `FakeEmbeddings` (8-dim, deterministic, hash-based) lives in `conftest.py` and is available to all tests.
- Run unit tests: `python -m pytest --ignore=tests/test_real_data.py -q`
- Run all tests (requires ES + embeddings API): `python -m pytest -q`

### LangGraph pipelines
- State is a `TypedDict` (`ExtractionState`, `RetrievalState`).
- Each node function takes the state dict and returns a partial update dict.
- Routing functions return string node names.
- Max retry count is 3 (checked via `retry_count < 3`).

### Environment variables (`.env`)
| Variable | Purpose |
|---|---|
| `LLM_MODEL` | LLM model name (default: `gpt-4o`) |
| `LLM_API_KEY` | OpenAI-compatible API key for LLM |
| `LLM_BASE_URL` | LLM API base URL |
| `EMBEDDING_MODEL` | Embedding model (default: `text-embedding-ada-002-2`) |
| `EMBEDDING_API_KEY` | Embeddings API key |
| `EMBEDDING_BASE_URL` | Embeddings API base URL |
| `ES_HOSTS` | Elasticsearch URL (default: `http://localhost:9200`) |
| `ES_INDEX` | ES index name (default: `kg_index`) |
| `SCHEMA_PATH` | Path to LinkML YAML (default: `schema/customer_service.yaml`) |

### Elasticsearch index
- Always **dropped and recreated** on `GraphStore.from_kg()` / `GraphStore.load()` — the index always mirrors the current graph exactly.
- Dense vector: 1536 dims, cosine similarity, `ApproxRetrievalStrategy` (kNN).
- Each ES document metadata includes: `node_id`, `node_type`, `sop_id`, `schema` (full `node.data` as opaque object, `"enabled": false`).

### Mermaid output
`subgraph_to_mermaid(subgraph, node_map)` in `graph/utils.py` generates `graph TD` diagrams. Node labels use `node_type: id` format. Always test mermaid output starts with `"graph TD"`.

### Adding a new node type
1. Add the class to `schema/customer_service.yaml` with `x_index: true` on searchable text fields.
2. Add the corresponding Pydantic extraction model to `graph/models.py`.
3. Add edge wiring in `graph/extractor.py` `build_graph_node()`.
4. No changes needed in `storage.py` or `extractor.py` validation — both are fully schema-driven.

## Running

```bash
uv run python main.py build   # extract SOPs from docs/ and build graph + ES index
uv run python main.py query   # interactive Q&A loop
uv run python main.py demo    # run preset demo queries
```
