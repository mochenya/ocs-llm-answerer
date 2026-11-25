"""从模型输出文本中提取并校验目标 Pydantic 结构，兼容 Markdown 代码块与包裹文本。"""

from __future__ import annotations

import json
import re

from pydantic import BaseModel, ValidationError

_CODE_FENCE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL | re.IGNORECASE)


def parse_structured_output[TOutput: BaseModel](
    raw_text: str,
    response_model: type[TOutput],
) -> TOutput:
    """按严格到宽松的三级策略解析结构化输出，只剥离外层包装而不篡改内容。

    解析顺序：
    1. 去除首尾空白后直接执行 Pydantic JSON 校验；
    2. 提取 Markdown 代码块（如 ```json ... ```）内容后校验；
    3. 使用标准库 JSONDecoder.raw_decode 从首个合法 JSON 对象起解析并校验。

    Args:
        raw_text: 模型返回的原始文本内容。
        response_model: 期望解析的目标 Pydantic 模型类型。

    Returns:
        已通过目标 Schema 校验的模型实例。

    Raises:
        ValidationError: 文本无法解析为合法 JSON，或 JSON 内容不符合目标 Schema。
    """
    stripped = raw_text.strip()
    try:
        return response_model.model_validate_json(stripped)
    except ValidationError as direct_error:
        best_error: ValidationError = direct_error

    for match in _CODE_FENCE.finditer(stripped):
        fenced = match.group(1).strip()
        if not fenced or fenced == stripped:
            continue
        try:
            return response_model.model_validate_json(fenced)
        except ValidationError as exc:
            best_error = _prefer_schema_error(best_error, exc)

    decoder = json.JSONDecoder()
    search_start = 0
    while (brace_index := stripped.find("{", search_start)) != -1:
        try:
            decoded, _ = decoder.raw_decode(stripped, brace_index)
        except json.JSONDecodeError:
            search_start = brace_index + 1
            continue
        if isinstance(decoded, dict):
            try:
                return response_model.model_validate(decoded)
            except ValidationError as exc:
                best_error = _prefer_schema_error(best_error, exc)
        search_start = brace_index + 1

    raise best_error


def _prefer_schema_error(
    current: ValidationError,
    candidate: ValidationError,
) -> ValidationError:
    """优先保留更具诊断价值的字段 Schema 校验错误，而非外层 JSON 语法错误。"""
    if _is_json_syntax_error(current) and not _is_json_syntax_error(candidate):
        return candidate
    return current


def _is_json_syntax_error(error: ValidationError) -> bool:
    """判断异常是否源于外层 JSON 语法解析失败。"""
    errors = error.errors()
    return len(errors) == 1 and errors[0].get("type") == "json_invalid"
