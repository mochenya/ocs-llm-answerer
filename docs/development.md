# 开发指南

本文档面向参与开发或维护本项目的人，覆盖开发命令、代码边界、测试、版本和打包。普通使用者只需要阅读根目录的 `README.md`。

## 基本原则

- 使用 `uv` 管理依赖、运行命令、测试和构建。
- 保持改动小而明确，不为单次需求引入不必要的抽象。
- provider API Key 只放在环境变量或 `.env`，不要写入 `config/providers.json`。
- 修改 OCS 兼容行为前先补测试，尤其是请求体解析、订阅配置和答案格式。
- SQLite schema 变更需要同步调整缓存、请求流水相关测试。
- `LLMClient` / `LLMSession` 返回泛型结构化输出（答题用例为 `AnswerPayload`）；题型校验和 OCS 格式转换集中在 `answer/formatting.py`，成功缓存只接收通过校验的答案。
- 新增和实质修改的核心函数使用 Google 风格 Docstring，说明职责、参数、返回值与可预期异常。

## 核心职责

- `AnswerService`：编排缓存查询、单请求模型会话、输出解析/题型校验纠正重试和仓储持久化，不直接执行 SQL。
- `LLMClient` / `LLMSession`：业务无关的通用模型调用协议；由具体适配器（`llm/adapters/`）维护单次请求的原生消息数组（含思考块与 `phase`）及容错结构化解析。
- `AnswerCacheRepository`：封装 SQLite 答案缓存读写、题目信息更新和命中统计。
- `LLMRequestRepository`：封装 SQLite 调用流水写入，按每次上游模型调用 1:1 保留成功和失败审计信息。

## 类型与依赖

包依赖遵循严格单向无环图（由 `tests/test_architecture.py` 静态守护）：`core ← llm ← answer ← (api, database)`，`main.py` 作为组合根统一装配。

- `core/config.py` 只定义纯环境变量与 `providers.json` 配置模型，零内部业务依赖。
- `api/schemas.py` 只定义 HTTP 输入输出，保留 OCS 的 `type` 别名和选项字符串兼容。
- `api/dependencies.py` 与 `api/images.py` 将请求转换为内部 `Question`（含 `media_type` + 纯 Base64 的 `QuestionImage`），原始 UTF-8 JSON 单独存入 `RequestAudit`。
- `answer/models.py` 定义内部题目、唯一答案载荷 `AnswerPayload`、缓存记录、`AnswerResult` 和不可变审计对象 `LLMCallAudit`；不包含 HTTP 成功标识或数据库选项 JSON 编码。
- `answer/prompting.py` 与 `answer/prompts/` 负责将 OCS 标准化题目（含 `[图片N]` 占位符）及校验报错组装为业务无关的 `UserInput`。
- `answer/ports.py` 定义服务需要的两个最小仓储协议（`AnswerCachePort`、`LLMRequestPort.record(LLMCallAudit)`）。SQLite 实现位于 `database/`，服务单元测试使用内存替身。
- `llm/models.py` 定义业务与 SDK 无关的输入原语（`TextPart`、`ImagePart`、`UserInput`）、泛型结果 `LLMCallResult[T]`、用量与配置快照；`llm/client.py` 定义 `LLMClient`/`LLMSession` 协议；`llm/parsing.py` 提供纯函数容错解析器；`llm/adapters/` 存放具体厂商协议实现；`llm/factory.py` 负责构造客户端。
- `database/connection.py` 统一初始化 `WAL` 日志模式，并为每次连接配置 `busy_timeout = 5000`、外键约束、`RepositoryError` 转换；仓储自行决定提交时机，不使用横跨模型网络调用的数据库事务。

## 失败与事务

成功路径在单次请求作用域的 `LLMSession` 内执行模型调用与题型校验，随后依次提交成功流水和缓存。当收到模型输出文本但触发结构解析失败（`ProviderOutputError`）或题型规则失败（`InvalidAnswerError`）时，会在同一会话中追加错误反馈重试，最多执行 `OCS_LLM_ANSWERER_MAX_ANSWER_ATTEMPTS` 次（默认 `3` 次 = 初次 1 次 + 最多纠正 2 次）；每次上游调用均 1:1 记录到 `llm_requests`，网络超时、上游 HTTP 故障或未知异常不进入纠正重试。

成功流水中的 `SUCCESS` 表示模型结果通过了题型校验，不代表后续缓存提交或 HTTP 响应发送成功。

| 失败位置 | 行为 |
| --- | --- |
| 缓存读取或命中统计 | 返回 `503`，不调用模型。 |
| 结构解析或题型校验失败 | 记录该次失败流水（含原文和用量），在剩余次数内追加错误反馈于同会话重试；耗尽后返回 `502`，不写缓存。 |
| 上游超时、限流、HTTP 故障 | 记录失败流水且不触发纠正重试，错误交由 API 层映射。 |
| 失败流水再次写入失败 | 记录服务日志，保留原答题异常及其 HTTP 映射。 |
| 成功流水写入失败 | 返回 `503`，不写缓存，避免缓存没有可追溯来源。 |
| 成功流水已提交，缓存写入失败 | 返回 `503`，保留已提交成功流水，不把它改记为模型失败。 |

这里刻意保留两个短事务，不将成功流水与缓存合并。缓存失败后重试可能再次调用模型，当前没有请求去重或自动修复缓存机制。

`RepositoryError` 和 Provider 错误保留原异常链。API 不回传上游响应原文、密钥或数据库诊断文本；未知编程错误仍按 `500` 处理。

## 资源所有权

应用工厂内部创建的 `LLMClient` 由应用负责关闭，包括运行退出和后续依赖组装失败。通过 `create_app(llm_client=...)` 注入的实例由注入方管理，应用不关闭它。直接创建 `OpenAIResponsesClient` 的调用方应在 `finally` 中执行 `await client.aclose()`。

## 契约测试

- `tests/test_architecture.py` 通过 AST 静态守护 `core ← llm ← answer ← (api, database)` 单向依赖。
- `tests/test_parsing.py` 验证纯函数结构化解析器对直接 JSON、Markdown 代码块和包裹文本的解析与错误保留。
- `tests/test_service.py` 不依赖 HTTP、真实 SQLite 或网络，覆盖单请求会话、纠正重试循环、调用顺序和失败边界。
- `tests/test_provider.py` 验证提示词构造、`OpenAIResponsesSession` 原样保留 `reasoning` 思考块与 `phase`、以及多会话隔离。
- `tests/test_contracts.py` 验证 OpenAPI 引用、两种请求体传输、错误映射和 SQLite 故障下的独立提交语义。
- `tests/test_lifespan.py` 验证自有与注入资源的关闭责任。
- `tests/test_llm_requests.py` 使用真实 SDK 和模拟 HTTP 传输，不访问真实模型服务。

新增及实质修改的函数应补齐参数、返回值类型注解和 Google 风格 Docstring。结构化动态输入在边界处校验，不通过扩大 `Any` 或忽略所有类型错误规避约束。

## 常用命令

安装依赖并准备本地配置：

```bash
uv sync
cp .env.example .env
cp config/providers.example.json config/providers.json
```

启动后端：

```bash
uv run ocs-llm-answerer --reload
```

等价模块方式：

```bash
uv run python -m ocs_llm_answerer --reload
```

指定监听地址、端口和日志级别：

```bash
uv run ocs-llm-answerer --host 127.0.0.1 --port 8000 --reload --log-level info
```

查看 CLI 帮助和版本：

```bash
uv run ocs-llm-answerer --help
uv run ocs-llm-answerer --version
```

## 质量检查

提交前至少运行：

```bash
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

需要自动格式化或应用安全 lint 修复时：

```bash
uv run ruff format .
uv run ruff check . --fix
```

运行聚焦测试：

```bash
uv run pytest tests/test_provider.py
uv run pytest tests/test_answer_api.py::test_health_endpoint
uv run pytest -q
```

## 版本管理

项目版本以 `pyproject.toml` 的 `[project].version` 为唯一来源。不要手写重复版本常量；`ocs_llm_answerer.__version__` 会读取已安装包的元数据。

查看或修改版本：

```bash
uv version --short
uv version --bump patch
uv version --bump minor
uv version 0.2.0
```

修改版本后运行：

```bash
uv lock
uv run pytest
uv build
```
