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
