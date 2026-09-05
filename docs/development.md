# 开发指南

本文档面向参与开发或维护本项目的人，覆盖开发命令、代码边界、测试、版本和打包。普通使用者只需要阅读根目录的 `README.md`。

## 基本原则

- 使用 `uv` 管理依赖、运行命令、测试和构建。
- 保持改动小而明确，不为单次需求引入不必要的抽象。
- provider API Key 只放在环境变量或 `.env`，不要写入 `config/providers.json`。
- 修改 OCS 兼容行为前先补测试，尤其是请求体解析、订阅配置和答案格式。
- SQLite schema 变更需要同步调整缓存、请求流水相关测试。
- Provider 返回结构化答案项；题型校验和 OCS 格式转换集中在 `answer/formatting.py`，成功缓存只接收通过校验的答案。
- 新增和实质修改的核心函数使用 Google 风格 Docstring，说明职责、参数、返回值与可预期异常。

## 核心职责

- `AnswerService`：编排答题流程，协调 Provider 和仓储，不直接执行 SQL。
- `AnswerCacheRepository`：封装 SQLite 答案缓存读写、题目信息更新和命中统计。
- `LLMRequestRepository`：封装 SQLite 调用流水写入，保留成功和失败调用的审计信息。

## 类型与依赖

- `api/schemas.py` 只定义 HTTP 输入输出，保留 OCS 的 `type` 别名和选项字符串兼容。
- `api/dependencies.py` 将请求转换为内部 `Question`，原始 UTF-8 JSON 单独存入 `RequestAudit`。
- `answer/models.py` 定义内部题目、缓存记录和 `AnswerResult`；不包含 HTTP 成功标识或数据库选项 JSON 编码。
- `answer/ports.py` 定义服务需要的两个最小仓储协议。SQLite 实现位于 `database/`，服务单元测试使用内存替身。
- `llm/models.py` 定义 SDK 无关的调用结果、用量和配置快照；`llm/provider.py` 只定义协议，具体适配器的创建位于 `llm/factory.py`。
- Provider 只接收标准化题目，不接收原始 HTTP 载荷。答案校验和 OCS 拼接仍集中在答题层。
- `database/connection.py` 统一外键配置、连接关闭和 `RepositoryError` 转换；仓储自行决定提交时机，不使用横跨模型网络调用的数据库事务。

## 失败与事务

成功路径依次执行模型调用、题型校验、成功流水提交、缓存提交。成功流水中的 `SUCCESS` 表示模型结果通过了题型校验，不代表后续缓存提交或 HTTP 响应发送成功。

| 失败位置 | 行为 |
| --- | --- |
| 缓存读取或命中统计 | 返回 `503`，不调用模型。 |
| Provider 调用或题型校验 | 记录失败流水，不写成功缓存；错误交由 API 层映射。 |
| 失败流水再次写入失败 | 记录服务日志，保留原答题异常及其 HTTP 映射。 |
| 成功流水写入失败 | 返回 `503`，不写缓存，避免缓存没有可追溯来源。 |
| 成功流水已提交，缓存写入失败 | 返回 `503`，保留已提交成功流水，不把它改记为模型失败。 |

这里刻意保留两个短事务，不将成功流水与缓存合并。缓存失败后重试可能再次调用模型，当前没有请求去重或自动修复缓存机制。

`RepositoryError` 和 Provider 错误保留原异常链。API 不回传上游响应原文、密钥或数据库诊断文本；未知编程错误仍按 `500` 处理。

## 资源所有权

应用工厂内部创建的 Provider 由应用负责关闭，包括运行退出和后续依赖组装失败。通过 `create_app(provider=...)` 注入的实例由注入方管理，应用不关闭它。直接创建 `OpenAIResponsesProvider` 的调用方应在 `finally` 中执行 `await provider.aclose()`。

## 契约测试

- `tests/test_service.py` 不依赖 HTTP、真实 SQLite 或网络，覆盖用例调用顺序和失败边界。
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
