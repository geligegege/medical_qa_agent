# Medical QA Agentic RAG（医疗问诊系统）

一个面向医疗问答与初步问诊场景的 Agentic RAG 项目，基于 LangGraph + LangChain + FAISS + FastAPI。

核心目标：在保证安全与相关性的前提下，提供可解释、可扩展、可部署的医疗问答能力。

## 功能特性

- 多阶段输入安全校验：提示词注入检测、毒性检测、长度限制。
- 医疗主题识别：过滤非医疗问题，避免跑题回答。
- 语义检索：基于 FAISS 的医疗知识上下文召回。
- 回答质量检查：语言一致性、相关性校验与失败回退。
- 支持本地与云端 LLM：可选 Ollama / DashScope 兼容模型。
- API + Web UI：默认通过 FastAPI 提供服务。

## 技术栈

| 类别 | 技术 |
|---|---|
| Language | Python 3.9+ |
| Orchestration | LangGraph |
| LLM | LangChain, Ollama, DashScope-compatible API |
| Retrieval | FAISS |
| Backend | FastAPI, Uvicorn |
| Data | Polars |
| Safety | LLM Guard |
| Deploy | Docker, Docker Compose |

## 快速开始

### 1. 获取项目

```bash
git clone <your-repo-url>
cd medical-qa-agent
```

### 2. 安装依赖（推荐）

```bash
uv sync --frozen --no-dev
```

### 3. 配置 .env

参考 .env.example 创建 .env。

常见两种模式：

- 本地模型模式

```env
USE_LOCAL_LLM=true
OLLAMA_MODEL_NAME=llama3.2:1b
```

- 云端模型模式

```env
USE_LOCAL_LLM=false
DASHSCOPE_API_KEY=your_key_here
QWEN_MODEL_NAME=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

## 本地运行

### 1. 预处理并构建索引

```bash
.\.venv\Scripts\python.exe -m src.indexing.build_medical_index
```

### 2. 启动 API

```bash
.\.venv\Scripts\python.exe -m uvicorn src.api.medical_qa_api:app --reload
```

### 3. 打开服务

- App: http://localhost:8000
- Health: http://localhost:8000/health

## Docker 运行

```bash
docker compose up --build
```

访问地址：

- App: http://localhost:8000
- Health: http://localhost:8000/health

说明：

- ollama 服务负责本地模型。
- data-indexing 先完成索引构建，再启动 bot-api。

## API

| Method | Path | Description |
|---|---|---|
| GET | /health | 服务健康检查 |
| POST | /answer | 提交医疗问题并获取回答 |

请求示例：

```json
{
  "question": "我最近咳嗽三天并伴随低烧，需要去医院吗？"
}
```

## 项目结构

```text
src/
  api/        FastAPI 与前端静态页面
  graph/      LangGraph 工作流节点与状态
  indexing/   医疗数据预处理与向量索引构建
  evaluation/ 问诊问答评估脚本
data/
  medical_data.jsonl
  indexes/
```

## 常用命令

```bash
# 模块方式运行图工作流示例
.\.venv\Scripts\python.exe -m src.graph.medical_qa_workflow

# 执行评估
.\.venv\Scripts\python.exe -m src.evaluation.evaluate_medical_qa
```

## 使用声明

本系统用于医疗知识问答与信息辅助，不构成专业医疗诊断、治疗建议或处方依据。
出现紧急症状或病情加重时，请及时前往正规医疗机构就诊。
