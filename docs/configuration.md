# 配置说明

本文档说明 OCS LLM Answerer 的运行配置、provider 配置和 OCS 接入相关选项。首次运行通常只需要复制示例文件并填入 API Key：

```bash
cp .env.example .env
cp config/providers.example.json config/providers.json
```

## 应用环境变量

应用通过 `.env` 或进程环境变量读取配置。真实环境变量优先级高于 `.env`，不会被 `.env` 覆盖。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OCS_LLM_ANSWERER_API_KEY` | 空 | `POST /api/v1/answer` 的可选访问密钥。留空时不校验，适合本地开发。 |
| `OCS_LLM_ANSWERER_DATABASE_PATH` | `data/cache.sqlite3` | SQLite 缓存和 LLM 请求流水数据库路径。 |
| `OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH` | `config/providers.json` | provider 运行配置文件路径。 |
| `OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE` | `fetch` | `/ocs-answerer.json` 输出的 OCS 请求模式，支持 `fetch` 和 `GM_xmlhttpRequest`。 |

示例：

```env
OCS_LLM_ANSWERER_API_KEY=change-me
OCS_LLM_ANSWERER_DATABASE_PATH=data/cache.sqlite3
OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH=config/providers.json
OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE=fetch
OPENAI_API_KEY=sk-...
```

## 访问密钥

`OCS_LLM_ANSWERER_API_KEY` 只保护 `POST /api/v1/answer`。配置后，请求必须携带：

```http
X-API-Key: change-me
```

`GET /ocs-answerer.json` 是公开订阅接口，始终不返回 `X-API-Key`。未启用鉴权时可以直接使用订阅；启用鉴权时，将订阅模板复制到 OCS 自定义题库配置，在 `headers` 中手动添加上述请求头，并填入 `.env` 中的相同密钥。不要把真实密钥或含密钥的题库配置提交到 Git；`.env` 已被 `.gitignore` 忽略。

## SQLite 数据库

服务启动时会初始化 SQLite schema。默认数据库路径是：

```text
data/cache.sqlite3
```

数据库包含三类数据：

| 表 | 用途 |
| --- | --- |
| `questions` | 标准化后的题目、题型、选项和原始请求 JSON。 |
| `answer_cache` | 题目答案缓存、provider、模型、命中次数和最近命中时间。 |
| `llm_requests` | 每次 LLM 调用的配置、状态、原始响应、错误、耗时和 token 用量。 |

`data/*.sqlite`、`data/*.sqlite3` 和 `data/*.db` 已被忽略，避免把本地题库缓存提交到仓库。

## Provider 配置

真实 provider 配置文件为 `config/providers.json`，从示例复制生成：

```bash
cp config/providers.example.json config/providers.json
```

示例配置：

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

字段说明：

| 字段 | 说明 |
| --- | --- |
| `active_provider` | 当前启用的 provider 名称，必须存在于 `providers`。 |
| `providers` | provider 配置映射，可以配置多个，但同一时间只启用一个。 |
| `adapter` | provider adapter 标识。当前支持 `openai_responses`。 |
| `base_url` | OpenAI API 或兼容网关的 HTTP(S) base URL。 |
| `api_key_env` | API Key 所在环境变量名。密钥值不写进 JSON。 |
| `model` | 请求的模型名称。 |
| `timeout_seconds` | SDK 请求超时时间，必须大于 0。 |
| `max_retries` | SDK 最大重试次数，必须大于等于 0。 |
| `extra_body` | 原样附加到 OpenAI Responses API 请求体，适合兼容网关的厂商参数。 |

当前 `openai_responses` adapter 要求后端兼容 OpenAI Responses API，不是 Chat Completions API。它返回结构化答案项及响应元数据，答题层统一校验题型规则，再转换为 OCS 需要的答案字符串；非法答案会记录为失败且不进入缓存。

## OCS 请求模式

`OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE` 控制 `/ocs-answerer.json` 中是否输出额外的 `type` 字段。

| 值 | 行为 |
| --- | --- |
| `fetch` | 默认值，不输出额外 `type` 字段，由 OCS 使用普通 fetch。 |
| `GM_xmlhttpRequest` | 在订阅配置中输出 `"type": "GM_xmlhttpRequest"`，用于需要油猴跨域请求的环境。 |

如果浏览器页面调用本地服务时遇到跨域问题，优先确认 OCS 端请求模式是否需要改为 `GM_xmlhttpRequest`。后端已经允许 OCS 页面从浏览器访问本机服务所需的 CORS 和 Private Network Access 预检。
