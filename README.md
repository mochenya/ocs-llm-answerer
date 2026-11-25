<div align="center">

# OCS LLM Answerer

面向 [OCS 网课助手](https://docs.ocsjs.com/) 自定义题库接口的 LLM 答题后端。

[![Python](https://img.shields.io/badge/Python-3.14%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Responses_API-412991?logo=openai&logoColor=white)](https://platform.openai.com/docs/api-reference/responses)
[![SQLite](https://img.shields.io/badge/SQLite-Cache-003B57?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![uv](https://img.shields.io/badge/uv-Project_Manager-2B0231)](https://docs.astral.sh/uv/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

把传统静态题库变成一个可以推理、可缓存、可追溯的本地优先智能题库。

</div>

> 项目处于早期阶段，当前重点是稳定 OCS 接入、LLM 答题、SQLite 缓存和请求审计链路。

## 目录

- [项目亮点](#项目亮点)
- [快速上手](#快速上手)
- [OCS 配置示例](#ocs-配置示例)
- [常用配置](#常用配置)
- [工作原理](#工作原理)
- [项目架构](#项目架构)
- [技术栈](#技术栈)
- [文档](#文档)
- [项目信息](#项目信息)
- [许可证](#许可证)

## 项目亮点

| 能力 | 说明 |
| --- | --- |
| OCS 兼容 | 提供答题接口、图片题支持与可导入的题库订阅配置。 |
| 智能缓存 | 先查 SQLite（WAL 模式）缓存，未命中再请求 LLM，避免同题重复消耗 token。 |
| 结构化与纠错重试 | 支持 OpenAI Responses API 结构化输出与代码块容错解析；解析或题型校验失败时在同一会话中保留原响应（含思考块）并追加错误反馈重试。 |
| 请求审计 | 按每次上游调用 1:1 记录配置快照、原始响应、状态、耗时和 token 用量。 |
| 模块解耦 | 严格单向分层架构（`core ← llm ← answer ← api/database`），业务提示词与底层模型协议彻底分离。 |
| 本地优先 | 默认监听 `127.0.0.1:8000`，密钥通过 `.env` 和环境变量管理。 |

## 快速上手

第一次使用时，按下面顺序完成配置、启动服务，然后把题库订阅地址填入 OCS。

| 步骤 | 你要做什么 | 关键结果 |
| --- | --- | --- |
| 1 | 准备 `uv` | 可以安装依赖并运行服务 |
| 2 | 复制本地配置文件 | 生成 `.env` 和 `config/providers.json` |
| 3 | 填入 LLM API Key | 服务可以调用模型 |
| 4 | 启动本地服务 | `http://127.0.0.1:8000` 可访问 |
| 5 | 在 OCS 中填写订阅地址 | OCS 可以调用本服务答题 |

### 1. 准备环境

- Python `>=3.14`
- [uv](https://docs.astral.sh/uv/) 用于依赖、运行、测试和构建
- 可访问 OpenAI Responses API 或兼容该接口的服务

### 2. 安装依赖并生成配置

```bash
uv sync
cp .env.example .env
cp config/providers.example.json config/providers.json
```

### 3. 填入模型密钥

编辑 `.env`，至少填入 provider 使用的 API Key：

```env
OPENAI_API_KEY=sk-...
```

如果你使用 OpenAI 兼容网关，可以继续编辑 `config/providers.json`，按需调整 `base_url`、`model` 和 `extra_body`。

### 4. 启动服务

```bash
uv run ocs-llm-answerer --reload
```

服务默认运行在：

```text
http://127.0.0.1:8000
```

### 5. 填入 OCS

推荐在 OCS 题库订阅地址中填写：

```text
http://127.0.0.1:8000/ocs-answerer.json
```

这个地址会返回 OCS 可导入的数组配置，并自动带上本服务当前的答题接口地址。公开订阅始终不包含 API Key。若设置了 `OCS_LLM_ANSWERER_API_KEY`，请使用下方的手动配置，在 `X-API-Key` 中填写相同密钥后导入 OCS。

## OCS 配置示例

未启用 API Key 鉴权时，推荐直接使用订阅地址：

```text
http://127.0.0.1:8000/ocs-answerer.json
```

启用 API Key 鉴权时，使用下面的完整 JSON，将 `change-me` 替换为 `.env` 中的 `OCS_LLM_ANSWERER_API_KEY`，再粘贴到 OCS 自定义题库配置。未启用鉴权时可以删除 `X-API-Key` 请求头。注意：OCS 题库配置必须是数组，含密钥的配置不要公开分享。

```json
[
  {
    "name": "OCS LLM Answerer",
    "homepage": "http://127.0.0.1:8000",
    "url": "http://127.0.0.1:8000/api/v1/answer",
    "method": "post",
    "contentType": "json",
    "headers": {
      "Content-Type": "application/json",
      "X-API-Key": "change-me"
    },
    "data": {
      "title": "${title}",
      "type": "${type}",
      "options": "${options}"
    },
    "handler": "return (res)=> res.code === 1 ? [res.question, res.answer] : undefined"
  }
]
```

如果你的 OCS 环境需要油猴跨域请求模式，请在 `.env` 中设置：

```env
OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE=GM_xmlhttpRequest
```

## 常用配置

应用配置来自 `.env` 和环境变量，provider 运行配置来自 `config/providers.json`。

```env
OCS_LLM_ANSWERER_API_KEY=
OCS_LLM_ANSWERER_DATABASE_PATH=data/cache.sqlite3
OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH=config/providers.json
OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE=fetch
OCS_LLM_ANSWERER_IMAGE_ENABLED=false
OCS_LLM_ANSWERER_MAX_ANSWER_ATTEMPTS=3
OPENAI_API_KEY=
```

`config/providers.json` 示例：

```json
{
  "active_provider": "openai",
  "providers": {
    "openai": {
      "adapter": "openai_responses",
      "base_url": "https://api.openai.com/v1",
      "api_key_env": "OPENAI_API_KEY",
      "model": "gpt-4o-mini",
      "timeout_seconds": 30,
      "max_retries": 2,
      "extra_body": {}
    }
  }
}
```

更多配置说明见 [docs/configuration.md](docs/configuration.md)。

## 工作原理

```mermaid
flowchart LR
    A["OCS 网课助手"] --> B["API 入站防腐层"]
    B --> C["标准化题目与哈希"]
    C --> D{"SQLite WAL 缓存命中?"}
    D -- 是 --> E["返回答案"]
    D -- 否 --> F["创建单请求 LLMSession"]
    F --> G["调用 LLM 适配器并容错解析"]
    G --> H["校验题型规则与格式化"]
    H --> I{"解析或题型失败?"}
    I -- "是且未达上限" --> J["记录失败流水并追加错误反馈"]
    J --> G
    I -- "否(成功)" --> K["记录成功流水"]
    K --> L["写入答案缓存"]
    L --> E
```

服务启动时会初始化 SQLite（默认启用 `WAL` 日志模式与 `busy_timeout`），并根据 `config/providers.json` 创建当前启用的 `LLMClient`。每次答题请求先经过 `api/` 防腐层解析 OCS 载荷（兼容 `text/plain` JSON 与可选 Base64 图片），再统一换行、去除外围空白并整理选项，同时保留正文内部空白与选项顺序。缓存命中时直接返回答案；未命中时为本次请求创建独立的 `LLMSession`（原样保留上游 `output` 数组中的 `reasoning` 思考块与 `phase` 字段），调用模型并执行容错 JSON 解析与题型语义校验。若输出解析或题型校验失败，会在同一会话中追加错误原因重试（默认最多尝试 3 次，可由 `OCS_LLM_ANSWERER_MAX_ANSWER_ATTEMPTS` 控制），每次上游调用均独立记录到 `llm_requests`，仅通过校验的结果才会写入 `answer_cache`。

## 项目架构

代码采用严格单向无环分层（由 `tests/test_architecture.py` 静态守护），底层模型能力与答题业务完全解耦，由 `main.py` 作为组合根（Composition Root）统一装配：

```mermaid
flowchart TD
    CLI["cli.py"] --> MAIN["main.py 组合根"]
    MAIN --> API["api/ 入站 HTTP 防腐层"]
    MAIN --> ANSWER["answer/ 答题核心业务"]
    MAIN --> DB["database/ SQLite 仓储适配"]
    MAIN --> LLM["llm/ 通用模型能力层"]
    MAIN --> CORE["core/ 纯配置"]

    API --> ANSWER
    DB --> ANSWER
    ANSWER --> LLM
    LLM --> CORE
```

| 目录 / 模块 | 职责边界 |
| --- | --- |
| `src/ocs_llm_answerer/core/` | 纯环境变量与 `providers.json` 配置校验，零内部业务依赖。 |
| `src/ocs_llm_answerer/llm/` | 业务无关的通用模型能力层：定义 `LLMClient` / `LLMSession` 协议、多模态输入原语（`TextPart`/`ImagePart`/`UserInput`）、泛型结果、容错 JSON/代码块解析器与客户端工厂。 |
| `src/ocs_llm_answerer/llm/adapters/` | 具体上游协议适配器（当前为 `openai_responses.py`，负责维护单请求原生消息数组与异常转换，便于后续扩展 `openai_chat_completions` 等协议）。 |
| `src/ocs_llm_answerer/answer/` | 答题核心领域层：题目标准化与哈希、OCS 提示词与多模态资源构建（`prompting.py` / `prompts/`）、唯一结构化载荷 `AnswerPayload`、题型校验格式化、最小仓储协议（`ports.py`）与纠错重试编排（`service.py`）。 |
| `src/ocs_llm_answerer/database/` | SQLite 出站仓储适配层：统一管理 `WAL`、`busy_timeout`、外键约束及 `AnswerCacheRepository` / `LLMRequestRepository` 两个独立短事务实现。 |
| `src/ocs_llm_answerer/api/` | OCS HTTP 入站防腐层：处理 `text/plain` 载荷、Base64 图片校验、`suggestion_*` 归约、鉴权、OpenAPI 补全与脱敏错误映射。 |
| `src/ocs_llm_answerer/main.py` | 应用组合根与 FastAPI `lifespan` 生命周期管理，明确托管自有资源的创建与关闭。 |

更多分层契约、事务与失败边界详见 [docs/development.md](docs/development.md)。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 项目管理 | `uv` |
| Web 框架 | FastAPI、Uvicorn |
| 数据模型 | Pydantic、Pydantic Settings |
| LLM 调用 | OpenAI Python SDK、Responses API |
| 数据库 | SQLite、aiosqlite |
| 测试与质量 | pytest、ruff |
| 打包 | hatchling、uv build |

## 文档

- [配置说明](docs/configuration.md)
- [API 参考](docs/api.md)
- [开发指南](docs/development.md)

## 项目信息

| 项目 | 内容 |
| --- | --- |
| 作者 | mochenya |
| 邮箱 | 74086519+mochenya@users.noreply.github.com |
| 仓库 | <https://github.com/mochenya/ocs-llm-answerer.git> |

## 许可证

本项目使用 [MIT License](LICENSE)。
