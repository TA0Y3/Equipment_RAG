### RAG 知识库智能问答系统

基于 LangGraph 构建的企业级 RAG（检索增强生成）智能问答系统，支持多格式文档导入、多路混合检索与流式问答，面向 H3C / 华为 / 联想等产品的技术手册、用户指南场景。

---

### 架构概览

系统分为两大子服务，各自独立运行，通过 Milvus / MongoDB 共享数据：

```
┌─────────────────────────────────────────┐
│           Import Service :8000           │
│   PDF/MD → MinerU → 分块 → BGE-M3       │
│       嵌入 → Milvus 向量库               │
└──────────────┬──────────────────────────┘
               │  Milvus / MongoDB
┌──────────────┴──────────────────────────┐
│           Query Service :8001            │
│   用户问题 → 产品识别 → 多路并行检索     │
│   → RRF 融合 → BGE Rerank → LLM 回答    │
└──────────────────────────────────────────┘
```

---

### 技术栈

| 组件 | 技术选型 |
|---|---|
| 框架 | FastAPI + Uvicorn |
| 工作流编排 | LangGraph + LangChain |
| 向量数据库 | Milvus（1024 维稠密 + 稀疏混合向量） |
| 对话历史 | MongoDB |
| 对象存储 | MinIO |
| 嵌入模型 | BGE-M3 |
| 重排序 | BGE Reranker Large |
| PDF 解析 | MinerU |
| 大模型 | 通义千问 Qwen 系列（阿里云百炼 DashScope） |
| Web 搜索 | MCP (DashScope WebSearch) |
| 流式输出 | SSE (Server-Sent Events) |

---

### 目录结构

```
RAG 知识库智能问答系统/
├── app/
│   ├── clients/            # 外部服务客户端（Milvus/Mngo/MinIO）
│   ├── conf/               # 配置类（dataclass）
│   ├── core/               # 核心工具（Logger/Prompt 加载）
│   ├── import_process/     # 文件导入子服务
│   │   ├── agent/          #   LangGraph 导入图（状态 + 节点 + 编排）
│   │   ├── api/            #   FastAPI 上传接口
│   │   └── page/           #   导入页面前端
│   ├── query_process/      # 查询子服务
│   │   ├── agent/          #   LangGraph 查询图（状态 + 节点 + 编排）
│   │   ├── api/            #   FastAPI 查询接口
│   │   └── page/           #   聊天页面前端
│   ├── lm/                 # LLM/Embedding/Reranker 工具
│   ├── tool/               # 模型下载脚本
│   └── utils/              # 工具函数（SSE/任务/路径/限流）
├── prompts/                # 提示词模板
├── test/                   # 测试脚本
├── logs/                   # 日志输出
├── output/                 # 导入中间文件
├── doc/                    # 测试用产品手册（PDF）
├── .env                    # 环境变量配置
├── pyproject.toml          # 项目依赖（uv 管理）
└── requirements.txt        # 依赖参考
```

---

### 核心流程

#### 1. 文档导入流程

```
文件上传 → 入口校验 → PDF 转 MD (MinerU) → 图片处理
→ 文档分块 → 产品名称识别 → BGE-M3 向量化 → Milvus 入库
```

- 支持 PDF 和 Markdown 两种输入格式
- PDF 通过 MinerU 转换为结构化 Markdown，保留表格与图片
- 分块策略自适应文档长度
- 自动识别文档中的产品型号名称，用于后续精确检索

#### 2. 智能问答流程

```
用户提问 → 产品名称确认 → 多路并行检索
   ├── Dense Vector Search   (BGE-M3 稠密向量)
   ├── HyDE Search           (假设性文档嵌入)
   └── Web Search            (MCP 外部搜索)
→ 结果合并 → RRF 融合排序 → BGE Reranker 重排 → LLM 生成回答
```

- 当问题涉及多个候选产品时，系统会反问用户进行澄清
- 当未匹配到任何产品时，直接返回拒绝回答，避免幻觉
- 回答中可嵌入文档原始图片（如图表、示意图）

---

### 快速开始

**环境要求：**
- Python >= 3.11
- Milvus 服务
- MongoDB 服务
- MinIO 服务

**1. 安装依赖**

```bash
# 使用 uv（推荐）
uv sync

# 或使用 pip
pip install -r requirements.txt
```

**2. 配置环境变量**

复制并编辑 `.env`，关键配置项：

```env
# LLM（阿里云百炼）
OPENAI_API_KEY=sk-xxxxxxxx
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
LLM_DEFAULT_MODEL=qwen-flash

# Embedding 模型（本地 BGE-M3 路径）
BGE_M3_PATH=
BGE_DEVICE=cpu          # 或 cuda:0

# 向量库
MILVUS_URL=
CHUNKS_COLLECTION=kb_chunks

# 对话历史
MONGO_URL=mongodb:
MONGO_DB_NAME=kb002
```

**3. 下载模型**

```bash
python -m app.tool.download_bgem3      # BGE-M3 嵌入模型
python -m app.tool.download_reranker   # BGE Reranker 重排序模型
```

**4. 启动服务**

```bash
# 导入服务（端口 8000）
python -m app.import_process.api.file_import_service

# 查询服务（端口 8001）
python -m app.query_process.api.query_service
```

**5. 访问界面**

- 文件导入：[http://127.0.0.1:8000/import.html](http://127.0.0.1:8000/import.html)
- 智能问答：[http://127.0.0.1:8001/chat.html](http://127.0.0.1:8001/chat.html)

---

### API 接口

**导入服务（:8000）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/import.html` | 文件导入页面 |
| POST | `/upload` | 多文件上传，自动触发导入全流程 |
| GET | `/status/{task_id}` | 查询导入任务进度与状态 |

**查询服务（:8001）**

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/chat.html` | 智能问答页面 |
| POST | `/query` | 发起问答（支持 stream/non-stream） |
| GET | `/stream/{session_id}` | SSE 流式获取回答 |
| GET | `/history/{session_id}` | 查询历史对话 |
| DELETE | `/history/{session_id}` | 清除历史对话 |
| GET | `/health` | 健康检查 |

---

### 关键设计决策

**多路混合检索**
单一路径容易漏召回，系统同时走嵌入搜索、HyDE 假设文档搜索与 Web 搜索三条路径，通过 RRF 统一融合后再经 BGE Reranker 精排，提高回答完整性和准确性。

**产品名称识别流水线**
针对产品手册的实际业务场景，系统在导入时自动识别每份文档对应的产品型号，查询阶段再通过 LLM 结合历史上下文确认用户意图指向的产品，过滤无关文档检索。

**HyDE 策略**
当用户问题较短或不清晰时，先用 LLM 生成一篇假设性回答文档，用该文档的嵌入去检索，减少 query-document 语义 gap。

**流式 + 进度反馈**
查询采用 SSE 实时推送回答内容；导入通过任务状态轮询 API 向前端逐节点反馈进度。
