# Copilot 指令

## 项目概述

`graph-extracter` 是一个基于本体论(Ontology)的知识图谱构建和推理系统。

1.  **提取**：基于yaml本体论建模，使用LLM从语料中提取结构化知识图谱。
2.  **存储**：将图谱存储为 `data/graph.json`（NetworkX + Pydantic），并在Elasticsearch中存储向量索引。
3.  **检索**：通过向量召回+子图遍历来推理。

## 技术栈

- **Python ≥ 3.12**，包管理器：`uv`
- **Pydantic v2** — 所有数据模型 (`graph/models.py`)
- **LangGraph** — 提取管道 (`graph/extractor.py`) 和检索管道 (`graph/retriever.py`)
- **LangChain / langchain-openai** — LLM调用和嵌入 (`text-embedding-ada-002-2`, 1536维)
- **Elasticsearch 8.x** — 向量存储 (`dense_vector`, 余弦相似度, 1536维) 通过 `langchain-elasticsearch`
- **NetworkX** — 用于遍历的内存有向图
- **LinkML YAML** — schema的唯一真实来源 (`schema/customer_service.yaml`)
- **pytest** — 测试运行器；`pytest-mock` 用于修补

## 仓库布局

```
graph/
  models.py        # Pydantic 模型: KnowledgeGraph, GraphNode, GraphEdge, ExtractionOutput, enums
  extractor.py     # LangGraph 提取管道: extract_sops → build_graph → validate_graph → save_graph
  storage.py       # GraphStore: 加载/保存 JSON + 构建 Elasticsearch 向量索引
  retriever.py     # LangGraph 检索管道: search_nodes → expand_context → generate_answer
  schema_loader.py # LinkML YAML → JSON Schema + FieldSpec 用于验证/索引
  utils.py         # Mermaid 图表生成, 拓扑打印

schema/
  customer_service.yaml  # LinkML schema — 所有实体类型和枚举的唯一真实来源

data/
  graph.json       # 持久化的 KnowledgeGraph (节点 + 边)
  *.txt            # 源 SOP 文档

tests/
  conftest.py              # 共享 fixtures: FakeEmbeddings (8维), minimal_kg, minimal_extraction_output
  test_extractor.py        # 提取 + 验证逻辑的单元测试
  test_storage.py          # GraphStore 加载/保存/搜索的单元测试
  test_subgraph.py         # 子图遍历的单元测试
  test_search_traverse.py  # 向量搜索 + 遍历管道的单元测试
  test_models.py           # Pydantic 模型的单元测试
  test_schema_loader.py    # schema_loader 的单元测试
  test_utils.py            # utils 的单元测试
  test_real_data.py        # 使用真实 data/graph.json + 真实 ES + 真实嵌入 API 的集成测试

main.py  # CLI 入口点: `uv run python main.py [build|query|demo]`

.env 配置文件
```

## Schema驱动设计

- **`schema/customer_service.yaml`** 是唯一真实来源。**不要**在业务逻辑中硬编码类名（`SOP`, `SOPStep`等）或字段名。
- `schema_loader.py` 暴露:
  - `build_tool_from_schema()` — 动态构建 OpenAI 函数调用 schema
  - `get_class_field_specs(schema_path, class_name) → list[FieldSpec]` — 返回每个属性的规格 (名称, 是否必需, 范围, 是否多值, 枚举值, 描述, 是否索引)
- `FieldSpec` 上的 `indexed: bool` 字段 (由 YAML 中的 `x_index: true` 驱动) 控制哪些字段被包含在 ES 向量索引中 — **`storage.py` 中没有硬编码的字段名**。
- 提取提示和工具 schema 在每次运行时通过 `SCHEMA_PATH` 环境变量从 YAML 重建。

## 关键约定

### 类型提示
所有公共函数和方法**必须**有完整的类型提示。在每个模块顶部使用 `from __future__ import annotations`。

### Pydantic v2
使用 `model_validate()` (而不是 `parse_obj()`), `model_dump_json()` (而不是 `.json()`)。

### 测试
- **单元测试**必须同时修补 `graph.storage` 中的 `_create_embeddings` 和 `_create_vector_store`，以避免调用真实的 ES 或嵌入 API：
  ```python
  @pytest.fixture
  def make_store(fake_embeddings, minimal_kg):
      with patch("graph.storage._create_embeddings", return_value=fake_embeddings), \
           patch("graph.storage._create_vector_store", side_effect=lambda docs, emb: InMemoryVectorStore.from_documents(docs, emb)):
          yield GraphStore.from_kg(minimal_kg)
  ```
- **`test_real_data.py`** 是唯一使用真实 ES + 真实嵌入的文件 — 不要在那里添加 mock。
- `FakeEmbeddings` (8维, 确定性, 基于哈希) 位于 `conftest.py` 中，可供所有测试使用。
- 运行单元测试: `python -m pytest --ignore=tests/test_real_data.py -q`
- 运行所有测试 (需要 ES + 嵌入 API): `python -m pytest -q`

### LangGraph 管道
- 状态是一个 `TypedDict` (`ExtractionState`, `RetrievalState`)。
- 每个节点函数接收状态字典并返回一个部分更新字典。
- 路由函数返回字符串节点名称。
- 最大重试次数为 3 (通过 `retry_count < 3` 检查)。

### 环境变量 (`.env`)
| 变量 | 用途 |
|---|---|
| `LLM_MODEL` | LLM 模型名称 (默认: `gpt-4o`) |
| `LLM_API_KEY` | OpenAI 兼容的 LLM API 密钥 |
| `LLM_BASE_URL` | LLM API 基础 URL |
| `EMBEDDING_MODEL` | 嵌入模型 (默认: `text-embedding-ada-002-2`) |
| `EMBEDDING_API_KEY` | 嵌入 API 密钥 |
| `EMBEDDING_BASE_URL` | 嵌入 API 基础 URL |
| `ES_HOSTS` | Elasticsearch URL (默认: `http://localhost:9200`) |
| `ES_INDEX` | ES 索引名称 (默认: `kg_index`) |
| `SCHEMA_PATH` | LinkML YAML 路径 (默认: `schema/customer_service.yaml`) |

### Elasticsearch 索引
- 在 `GraphStore.from_kg()` / `GraphStore.load()` 时总是**被删除并重新创建** — 索引总是精确镜像当前图。
- 密集向量: 1536维, 余弦相似度, `ApproxRetrievalStrategy` (kNN)。
- 每个 ES 文档元数据包括: `node_id`, `node_type`, `sop_id`, `schema` (完整的 `node.data` 作为不透明对象, `"enabled": false`)。

### Mermaid 输出
`graph/utils.py` 中的 `subgraph_to_mermaid(subgraph, node_map)` 生成 `graph TD` 图表。节点标签使用 `node_type: id` 格式。始终测试 mermaid 输出以 `"graph TD"` 开头。

### 添加新的节点类型
1. 将类添加到 `schema/customer_service.yaml`，并在可搜索的文本字段上设置 `x_index: true`。
2. 将相应的 Pydantic 提取模型添加到 `graph/models.py`。
3. 在 `graph/extractor.py` 的 `build_graph_node()` 中添加边连接逻辑。
4. `storage.py` 或 `extractor.py` 的验证部分无需更改 — 两者都是完全由 schema 驱动的。

## 运行

```bash
uv run python main.py build   # 从 docs/ 提取 SOP 并构建图 + ES 索引
uv run python main.py query   # 交互式问答循环
uv run python main.py demo    # 运行预设的演示查询
```
