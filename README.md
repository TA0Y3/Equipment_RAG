# RAG 知识库智能问答系统

基于 **LangGraph + LangChain** 构建的企业级 RAG（检索增强生成）智能问答系统，支持多格式文档导入、多路混合检索与流式问答，面向 H3C / 华为 / 联想 / 奔图等产品的技术手册、用户指南场景。

---

## 系统架构

系统分为两大子服务，各自独立运行，通过 Milvus / MongoDB / MinIO 共享数据：

```
┌──────────────────────────────────────────────────┐
│             Import Service  (:8000)               │
│                                                   │
│  文件上传 → 入口校验 → PDF 转 MD (MinerU)          │
│    → 图片处理 → 文档分块 → 产品名称识别            │
│    → BGE-M3 向量化 → Milvus 入库                  │
└──────────────────────┬───────────────────────────┘
                       │   Milvus / MongoDB / MinIO
┌──────────────────────┴───────────────────────────┐
│             Query Service   (:8001)               │
│                                                   │
│  用户问题 → 产品确认 → 问题缓存命中检测            │
│    → 三路并行检索（向量 / HyDE / MCP 联网）        │
│    → RRF 融合 → BGE Rerank 重排 → LLM 生成回答    │
│    → SSE 流式推送                                 │
└──────────────────────────────────────────────────┘
```

## 核心特性

- **多格式文档导入**：支持 PDF / Markdown，PDF 经 MinerU 解析为结构化 Markdown（保留表格与图片）
- **图片智能处理**：自动提取文档图片上传 MinIO，并调用视觉大模型（Qwen-VL）生成图片摘要参与检索
- **混合向量检索**：BGE-M3 稠密向量（1024 维）+ 稀疏词法向量，Milvus WeightedRanker 加权融合
- **多路并行召回**：向量检索、HyDE 假设文档检索、MCP 联网搜索三路并行，RRF 融合后经 Reranker 精排
- **产品名称识别流水线**：导入时自动识别文档对应产品型号，查询时结合历史上下文确认用户意图，过滤无关文档
- **相似问题缓存**：改写后的问题向量化后先在缓存集合中命中检测，命中直接返回历史答案，跳过耗时检索
- **流式问答**：SSE 实时推送回答内容与任务进度；支持反问澄清与拒答兜底，避免幻觉
- **进度可视化**：导入任务逐节点反馈进度（含各节点耗时），前端实时轮询展示

## 技术栈

| 组件 | 技术选型 |
|---|---|
| Web 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph + LangChain |
| 向量数据库 | Milvus（1024 维稠密 + 稀疏混合向量） |
| 对话历史 | MongoDB |
| 对象存储 | MinIO |
| 嵌入模型 | BGE-M3（本地加载，GPU/CPU 可切换） |
| 重排序模型 | BGE Reranker Large（FlagEmbedding） |
| PDF 解析 | MinerU（远程 API） |
| 大模型 | 通义千问 Qwen 系列（阿里云百炼 DashScope，含 qwen3-vl 视觉模型） |
| Web 搜索 | MCP（DashScope WebSearch，Streamable HTTP 传输） |
| 流式输出 | SSE（Server-Sent Events） |
| 日志 | loguru（按天滚动，自动清理） |
| 测试评估 | Ragas（检索上下文召回评估） |

## 目录结构

```
RAG 知识库智能问答系统/
├── app/
│   ├── clients/            # 外部服务客户端（Milvus/Mongo/MinIO/Neo4j/检索缓存）
│   ├── conf/               # 配置类（dataclass 统一加载 .env）
│   ├── core/               # 核心工具（logger / prompt 加载）
│   ├── import_process/     # 文件导入子服务
│   │   ├── agent/          #   LangGraph 导入图（state + nodes + main_graph）
│   │   ├── api/            #   FastAPI 上传接口（file_import_service）
│   │   └── page/           #   导入页面前端（import.html）
│   ├── query_process/      # 查询子服务
│   │   ├── agent/          #   LangGraph 查询图（state + nodes + main_graph）
│   │   ├── api/            #   FastAPI 查询接口（query_service）
│   │   └── page/           #   聊天页面前端（chat.html）
│   ├── lm/                 # LLM/Embedding/Reranker 工具
│   ├── tool/               # 模型下载脚本（BGE-M3 / Reranker）
│   └── utils/              # 工具函数（SSE/任务状态/路径/限流/向量归一化）
├── prompts/                # 提示词模板（与代码解耦）
├── test/                   # 测试与评估脚本（含 Ragas 评估）
├── logs/                   # 日志输出（按天滚动）
├── output/                 # 导入中间文件（按 日期/任务ID 分层）
├── doc/                    # 测试用产品手册（PDF）
├── .env                    # 环境变量配置
├── pyproject.toml          # 项目依赖（uv 管理）
└── requirements.txt        # 依赖参考（pip）
```

## 核心流程

### 1. 文档导入流程（Import Service）

```
node_entry（入口校验，路由 PDF/MD）
    ├─ MD 直接导入 ──────────────┐
    └─ PDF 导入 → node_pdf_to_md │（MinerU 解析 + 轮询下载 ZIP + 提取 MD）
                                 ▼
                        node_md_img（图片处理：上传 MinIO + 视觉模型生成摘要）
                                 ▼
                    node_document_split（文档分块：自适应长度策略）
                                 ▼
              node_item_name_recognition（LLM 识别产品名称）
                                 ▼
                    node_bge_embedding（BGE-M3 稠密+稀疏向量化）
                                 ▼
                      node_import_milvus（入库：chunks + item_names 集合）
```

- 一个文件对应一个独立 `task_id`，前端轮询 `/status/{task_id}` 实时获取节点进度与耗时
- 产品名称作为检索过滤维度写入 Milvus，实现「产品级」精准召回
- 图片摘要由 Qwen-VL 视觉模型生成，融入文本后一并向量化，使图片内容可被检索

### 2. 智能问答流程（Query Service）

```
用户提问
   ▼
node_item_name_confirm（结合历史确认产品，支持反问澄清 / 拒答兜底）
   ├─ 已有 answer（反问/拒答）→ 直接输出
   ▼
node_query_cache（相似问题缓存命中检测，阈值 0.85）
   ├─ 命中 → 直接返回历史答案
   ▼
node_multi_search（三路并行分叉）
   ├── node_search_embedding      （BGE-M3 混合向量检索，产品名过滤）
   ├── node_search_embedding_hyde （LLM 生成假设文档后检索）
   └── node_web_search_mcp        （百炼 MCP 联网搜索）
   ▼
node_join → node_rrf（RRF 融合排序）→ node_rerank（BGE Reranker 精排）
   ▼
node_answer_output（组装上下文 + LLM 生成，SSE 流式推送）
```

## 快速开始

### 环境要求

- Python >= 3.11（项目使用 conda 环境 `rag_project`）
- 外部服务：Milvus、MongoDB、MinIO（按需）
- GPU 推荐（BGE-M3 / Reranker 本地推理），CPU 亦可（`BGE_DEVICE=cpu`）

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制并编辑 `.env`，关键配置项：

```env
# LLM（阿里云百炼 DashScope，OpenAI 兼容协议）
OPENAI_API_KEY=sk-
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen3.7-max          # 文本模型
VL_MODEL=qwen3-vl-flash                # 视觉模型（图片摘要）

# Embedding 模型（本地 BGE-M3）
BGE_M3_PATH=                          # 本地模型路径，留空自动下载 BAAI/bge-m3
BGE_DEVICE=cuda:0                     # 或 cpu
BGE_FP16=1                            # CPU 环境建议改为 0

# 重排序模型（本地 BGE Reranker Large）
BGE_RERANKER_LARGE=                   # 本地模型路径
BGE_RERANKER_DEVICE=cuda:0            # 或 cpu

# 向量库
MILVUS_URL=http://xxx.xxx.x.xxx
CHUNKS_COLLECTION=kb_chunks           # 切片集合
ITEM_NAME_COLLECTION=kb_item_names    # 产品实体集合

# 对话历史
MONGO_URL=mongodb://xxx.xxx.x.xxx
MONGO_DB_NAME=kb002

# 对象存储
MINIO_ENDPOINT=xxx.xxx.x.xxx:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=knowledge-base-files

# PDF 解析（MinerU 远程 API）
MINERU_BASE_URL=https://mineru.net/api/v4
MINERU_API_TOKEN=sk-xxxxxxxx

# 联网搜索（百炼 MCP WebSearch）
MCP_DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp

# 日志
LOG_CONSOLE_ENABLE=True
LOG_FILE_ENABLE=True
LOG_FILE_RETENTION=7 days
```

### 3. 下载模型（可选）

未配置本地模型路径时，运行下载脚本：

```bash
python -m app.tool.download_bgem3      # BGE-M3 嵌入模型
python -m app.tool.download_reranker   # BGE Reranker 重排序模型
```

### 4. 启动服务

```bash
# 导入服务（端口 8000）
python -m app.import_process.api.file_import_service

# 查询服务（端口 8001）
python -m app.query_process.api.query_service
```

### 5. 访问界面

- 文件导入：[http://127.0.0.1:8000/import.html](http://127.0.0.1:8000/import.html)
- 智能问答：[http://127.0.0.1:8001/chat.html](http://127.0.0.1:8001/chat.html)
- Swagger 文档：`/docs`（两个服务均可访问）

## API 接口

**导入服务（:8000）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/import.html` | 文件导入页面 |
| POST | `/upload` | 多文件批量上传（form-data），自动触发导入全流程，返回 `task_ids` |
| GET | `/status/{task_id}` | 查询导入任务进度：全局状态 + 已完成/运行中节点 + 各节点耗时 |

**查询服务（:8001）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat.html` | 智能问答页面 |
| POST | `/query` | 发起问答（`is_stream` 控制流式/非流式） |
| GET | `/stream/{session_id}` | SSE 流式获取回答与任务进度 |
| GET | `/history/{session_id}` | 查询历史对话 |
| DELETE | `/history/{session_id}` | 清除历史对话 |
| GET | `/health` | 健康检查 |

## 关键设计决策

**多路混合检索 + RRF 融合 + Rerank 精排**
单一路径容易漏召回。系统并行执行 BGE-M3 混合向量检索、HyDE 假设文档检索与 MCP 联网搜索三路召回，通过 RRF 统一融合，再经 BGE Reranker 交叉编码精排，兼顾召回完整性与排序准确性。

**产品名称识别流水线**
针对产品手册业务场景：导入时用 LLM 从文档中识别产品型号写入 Milvus 实体集合；查询时结合会话历史确认用户意图指向的产品（支持反问澄清、低置信度拒答），以产品名为过滤条件做精准检索，避免跨产品误召回。

**相似问题缓存**
每轮问答结束后，将「改写问题 + 答案」向量化写入 Milvus 缓存集合（容量上限 20 条）。新问题先做缓存命中检测（相似度阈值 0.85），命中直接复用历史答案，显著降低高频相似问题的响应延迟与 LLM 调用成本。

**HyDE 策略**
当用户问题较短或不清晰时，先用 LLM 生成一篇假设性回答文档，用该文档的嵌入去检索，缩小 query-document 之间的语义 gap。

**图片内容可检索化**
导入时提取文档图片：图片本体上传 MinIO（URL 持久化）、内容交由 Qwen-VL 生成文字摘要，摘要随正文一起分块向量化。回答阶段可还原图片 URL 嵌入最终答案，实现「图表级」问答。

**流式体验与进度反馈**
查询侧采用 SSE 推送增量回答；导入侧通过任务状态轮询 API 逐节点反馈进度（含节点中文名与耗时），长任务过程透明可见。

**API 限流**
图片摘要等高频 LLM 调用内置滑动窗口限流器，避免触发大模型服务端限流。

## 测试与评估

```bash
# 环境/日志/CUDA 基础验证
python test/01-env和系统环境变量的优先级.py
python test/02-日志测试.py
python test/03-cuda测试.py

# LangGraph 图流程测试
python test/04-test_graph_flow.py
python test/05-test-main-graph.py

# 检索上下文召回评估（Ragas，基于 test/ragas_testset_50.json 测试集）
python test/eval_context_recall.py
```
