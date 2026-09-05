# API 参考

本文档面向开发和调试阶段，说明 OCS LLM Answerer 暴露的 HTTP 接口。

## `GET /health`

返回服务健康状态。

响应：

```json
{"status": "ok"}
```

## `GET /ocs-answerer.json`

返回 OCS AnswererWrapper 可导入的公开题库订阅模板。OCS 会把订阅结果当数组展开，所以这里必须返回数组，不能返回单个对象。该接口无需鉴权，始终不返回访问密钥。

响应示例：

```json
[
  {
    "name": "OCS LLM Answerer",
    "homepage": "http://127.0.0.1:8000",
    "url": "http://127.0.0.1:8000/api/v1/answer",
    "method": "post",
    "contentType": "json",
    "headers": {
      "Content-Type": "application/json"
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

未配置 `OCS_LLM_ANSWERER_API_KEY` 时可直接使用订阅。配置了密钥时，需将模板复制到 OCS 自定义题库配置，并在 `headers` 中手动填写 `X-API-Key`；密钥不会通过公开订阅分发。

## `POST /api/v1/answer`

答题接口。服务会先按标准化题目哈希查询 SQLite 缓存，缓存未命中时调用当前配置的 LLM provider。

请求头：

```http
X-API-Key: change-me
Content-Type: application/json
```

如果未配置 `OCS_LLM_ANSWERER_API_KEY`，`X-API-Key` 可以省略。

请求体：

```json
{
  "title": "1+1=?",
  "type": "single",
  "options": ["A. 2", "B. 3"]
}
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `title` | `string` | 题目文本，必填，不能是空字符串或纯空白。 |
| `type` | `string` | 题型，可选。空字符串会按未知题型处理。 |
| `options` | `string[]` 或 `string` | 选项，可选。支持数组和 OCS 常见的换行字符串。 |

支持的题型：

| `type` | 说明 |
| --- | --- |
| `single` | 单选题 |
| `multiple` | 多选题，答案使用 `#` 拼接，例如 `A#C` |
| `judgement` | 判断题，答案为 `true` 或 `false` |
| `completion` | 填空题，多空答案使用 `#` 拼接 |

未知的非空题型会返回 `422`。

题干和选项只统一换行并清理外围空白，内部空格、缩进和换行会保留。模型输入与缓存身份使用相同的标准化规则，原始请求体单独用于审计。

模型答案在进入缓存前统一校验：单选和判断题必须恰好有一项，判断值必须能归一化为 `true/false`；选择题编号必须在已提供的选项范围内，也允许精确匹配的完整选项文本。多选会去重并按编号排序。填空保留每项内容，任何空项都会被拒绝，避免答案错位。缺少选项时无法检查编号上界；未知题型只校验答案非空。

不符合题型规则的模型答案返回 `502`，JSON 示例为 `{"detail":"Invalid model answer: single requires exactly one answer"}`。失败会记录原始响应和已取得的用量，不写入答案缓存。

响应示例：

```json
{
  "code": 1,
  "question": "1+1=?",
  "answer": "A",
  "explanation": "解析内容",
  "confidence": 0.9,
  "provider": "openai",
  "model": "gpt-4o-mini",
  "cache_hit": false
}
```

响应字段：

| 字段 | 说明 |
| --- | --- |
| `code` | OCS handler 使用的成功标识，当前成功响应为 `1`。 |
| `question` | 标准化后的题目文本。 |
| `answer` | OCS 可消费的答案字符串。 |
| `explanation` | LLM 返回的中文解析。 |
| `confidence` | LLM 返回的置信度，范围 `0` 到 `1`。 |
| `provider` | 生成该答案的 provider 名称。 |
| `model` | 生成该答案的模型名称。 |
| `cache_hit` | `true` 表示来自 SQLite 缓存，`false` 表示本次调用了 LLM。 |

## OCS 请求体兼容

OCS 可能使用 `Content-Type: text/plain;charset=UTF-8` 发送 JSON 字符串。后端会手动读取请求体并按 JSON 解析，所以 `POST /api/v1/answer` 同时支持标准 JSON 和 text/plain 包裹的 JSON。

无效 JSON、非 UTF-8 请求体或 Pydantic 校验失败都会返回 `422`。
