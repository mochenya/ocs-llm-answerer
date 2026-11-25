"""隐藏在 LLMClient / LLMSession 协议后的 OpenAI Responses API 适配器。"""

from __future__ import annotations

import json
from typing import Any, Never, cast

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from openai.types.responses import (
    ResponseInputContentParam,
    ResponseInputItemParam,
)
from pydantic import BaseModel, ValidationError

from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.llm.errors import (
    ProviderOutputError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ocs_llm_answerer.llm.models import (
    ImagePart,
    LLMCallResult,
    LLMProviderMetadata,
    LLMUsage,
    TextPart,
    UserInput,
)
from ocs_llm_answerer.llm.parsing import parse_structured_output


def _build_user_item(user_input: UserInput) -> ResponseInputItemParam:
    """将业务无关的多模态输入转换为 OpenAI Responses 原生 user 消息项。"""
    if len(user_input.parts) == 1 and isinstance(user_input.parts[0], TextPart):
        return {"role": "user", "content": user_input.parts[0].text}

    content: list[ResponseInputContentParam] = []
    for part in user_input.parts:
        if isinstance(part, TextPart):
            content.append({"type": "input_text", "text": part.text})
        elif isinstance(part, ImagePart):
            content.append({"type": "input_image", "image_url": part.data_url, "detail": "auto"})
    return {"role": "user", "content": content}


class OpenAIResponsesSession:
    """单次请求作用域的 Responses 会话，直接维护原生输入/输出数组以保留思考块。"""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        instructions: str,
        extra_body: dict[str, Any] | None,
    ) -> None:
        """初始化空对话历史，仅在单次业务请求内复用。

        Args:
            client: 由所属 OpenAIResponsesClient 托管生命周期的异步 SDK 客户端。
            model: 请求的模型名称。
            instructions: 本次会话的系统指令。
            extra_body: 透传给上游的厂商自定义参数。
        """
        self._client = client
        self._model = model
        self._instructions = instructions
        self._extra_body = extra_body
        self._items: list[ResponseInputItemParam] = []

    async def send_structured[TOutput: BaseModel](
        self,
        user_input: UserInput,
        response_model: type[TOutput],
    ) -> LLMCallResult[TOutput]:
        """发送追加后的原生消息数组，原样保留响应输出项并解析目标 Schema。

        Args:
            user_input: 本轮用户输入内容。
            response_model: 期望解析的目标 Pydantic 模型。

        Returns:
            包含结构化输出、原始响应、HTTP 状态和用量的调用结果。

        Raises:
            ProviderTimeoutError: 上游调用超时。
            ProviderUnavailableError: 连接失败、限流或上游暂时不可用。
            ProviderOutputError: 收到模型输出文本但无法解析为目标 Schema。
            ProviderResponseError: 上游 HTTP 错误、损坏响应体或缺失输出文本。
        """
        user_item = _build_user_item(user_input)
        request_items: list[ResponseInputItemParam] = [*self._items, user_item]
        try:
            raw_response = await self._client.responses.with_raw_response.parse(
                model=self._model,
                instructions=self._instructions,
                input=request_items,
                store=False,
                text_format=response_model,
                extra_body=self._extra_body,
            )
        except APITimeoutError as exc:
            raise ProviderTimeoutError(str(exc)) from exc
        except APIConnectionError as exc:
            raise ProviderUnavailableError(str(exc)) from exc
        except APIStatusError as exc:
            _raise_with_raw(exc, exc.response.text, exc.status_code)

        response_body_raw = raw_response.text
        try:
            response = raw_response.parse()
            usage = _extract_usage(response)
        except Exception as exc:
            _raise_with_raw(exc, response_body_raw, raw_response.status_code)

        parsed = getattr(response, "output_parsed", None)
        if isinstance(parsed, response_model):
            output_items = _extract_history_items(
                response,
                raw_body=response_body_raw,
                fallback_text=parsed.model_dump_json(),
            )
            self._items.append(user_item)
            self._items.extend(output_items)
            return LLMCallResult(
                parsed=parsed,
                response_body_raw=response_body_raw,
                http_status=raw_response.status_code,
                usage=usage,
            )

        try:
            output_text = _extract_output_text(response)
        except Exception as exc:
            _raise_with_raw(exc, response_body_raw, raw_response.status_code)

        output_items = _extract_history_items(
            response, raw_body=response_body_raw, fallback_text=output_text
        )
        self._items.append(user_item)
        self._items.extend(output_items)

        try:
            validated = parse_structured_output(output_text, response_model)
        except (ValidationError, ValueError) as exc:
            raise ProviderOutputError(
                f"{type(exc).__name__}: {exc}",
                raw_body=response_body_raw,
                status_code=raw_response.status_code,
                usage=usage,
                raw_output_text=output_text,
            ) from exc

        return LLMCallResult(
            parsed=validated,
            response_body_raw=response_body_raw,
            http_status=raw_response.status_code,
            usage=usage,
        )


class OpenAIResponsesClient:
    """持有 AsyncOpenAI 客户端并创建单请求会话的 Responses 适配器。"""

    def __init__(self, name: str, config: ProviderConfig, api_key: str) -> None:
        """创建由本适配器持有的异步客户端。

        Args:
            name: 配置中的 Provider 名称。
            config: 已校验的模型调用配置。
            api_key: 运行时读取的访问密钥，不写入审计记录。
        """
        self.name = name
        self.model = config.model
        self.request_metadata = LLMProviderMetadata(
            adapter=config.adapter,
            base_url=config.base_url,
            api_key_env=config.api_key_env,
            model=config.model,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            extra_body=config.extra_body,
        )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._extra_body = config.extra_body

    def create_session(self, *, instructions: str) -> OpenAIResponsesSession:
        """创建独立的单次请求会话，不与其他请求共享消息历史。

        Args:
            instructions: 本次会话使用的系统指令。

        Returns:
            持有共享 HTTP 客户端与独立消息数组的会话对象。
        """
        return OpenAIResponsesSession(
            client=self._client,
            model=self.model,
            instructions=instructions,
            extra_body=self._extra_body,
        )

    async def aclose(self) -> None:
        """关闭本适配器的 HTTP 客户端，关闭后不可继续调用模型。"""
        await self._client.close()


def _raise_with_raw(exc: Exception, raw_body: str, status_code: int) -> Never:
    """传播附带原始响应的错误，并保留原异常链。

    Args:
        exc: SDK 或响应解析抛出的异常。
        raw_body: 上游原始响应文本。
        status_code: 上游 HTTP 状态码。

    Raises:
        ProviderUnavailableError: 上游限流或暂时不可用。
        ProviderResponseError: 其他响应错误，保留原文和异常链。
    """
    error_type = ProviderUnavailableError if status_code in {429, 503} else ProviderResponseError
    raise error_type(
        f"{type(exc).__name__}: {exc}",
        raw_body=raw_body,
        status_code=status_code,
    ) from exc


def _extract_history_items(
    response: object,
    *,
    raw_body: str | None = None,
    fallback_text: str | None = None,
) -> list[ResponseInputItemParam]:
    """优先从原始响应 JSON 直接提取 output 数组，完整保留 reasoning 思考块和 phase 字段。

    直接读取 raw_body 中的 output 列表可避免 SDK model_dump 的序列化警告，同时不丢失
    厂商扩展思考字段；测试替身未提供完整 raw_body 时回退到 SDK output 对象或纯文本消息。
    """
    if raw_body:
        try:
            raw_json = json.loads(raw_body)
        except json.JSONDecodeError:
            raw_json = None
        if isinstance(raw_json, dict):
            raw_output = raw_json.get("output")
            if isinstance(raw_output, list) and raw_output:
                items = [
                    cast(ResponseInputItemParam, item)
                    for item in raw_output
                    if isinstance(item, dict)
                ]
                if items:
                    return items

    sdk_output = getattr(response, "output", None)
    if isinstance(sdk_output, list) and sdk_output:
        extracted: list[ResponseInputItemParam] = []
        for item in sdk_output:
            dumped = _dump_output_item(item)
            if dumped is not None:
                extracted.append(cast(ResponseInputItemParam, dumped))
        if extracted:
            return extracted

    if fallback_text is not None:
        return [{"role": "assistant", "content": fallback_text}]
    return []


def _dump_output_item(item: object) -> dict[str, Any] | None:
    """将单个 SDK 输出项转为字典并剥离仅客户端使用的 parsed 字段。"""
    if isinstance(item, dict):
        dumped: dict[str, Any] = dict(item)
    elif isinstance(item, BaseModel):
        dumped = item.model_dump(mode="json", exclude_none=True)
    else:
        return None

    content = dumped.get("content")
    if isinstance(content, list):
        cleaned_content = []
        for entry in content:
            if isinstance(entry, dict) and "parsed" in entry:
                cleaned = dict(entry)
                cleaned.pop("parsed", None)
                cleaned_content.append(cleaned)
            else:
                cleaned_content.append(entry)
        dumped["content"] = cleaned_content
    return dumped


def _extract_output_text(response: object) -> str:
    """同时从 SDK 便捷字段和结构化输出项中读取文本。

    Args:
        response: SDK 返回的响应对象。

    Returns:
        首个可用的输出文本，不删除或重新编码其内容。

    Raises:
        RuntimeError: 没有可用的输出文本。
    """
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if isinstance(text, str) and text:
                    return text

    dumped = (
        response.model_dump_json() if hasattr(response, "model_dump_json") else json.dumps(response)
    )
    raise RuntimeError(f"OpenAI response did not contain output text: {dumped}")


def _extract_usage(response: object) -> LLMUsage:
    """校验上游用量字段，兼容 SDK 对象与字典表示。

    Args:
        response: 包含可选 usage 字段的响应。

    Returns:
        缺失字段为 None 的用量记录。

    Raises:
        ValidationError: 上游用量不是非负整数。
    """
    usage = _get_field(response, "usage")
    input_token_details = _get_field(usage, "input_tokens_details")
    return LLMUsage.model_validate(
        {
            "input_tokens": _get_field(usage, "input_tokens"),
            "output_tokens": _get_field(usage, "output_tokens"),
            "total_tokens": _get_field(usage, "total_tokens"),
            "cached_tokens": _get_field(input_token_details, "cached_tokens"),
        }
    )


def _get_field(value: object, field_name: str) -> object:
    """读取 SDK 对象或字典中的可选字段，不执行响应序列化。

    Args:
        value: 上游对象、字典或 None。
        field_name: 待读取的字段名称。

    Returns:
        字段原值，缺失时为 None；调用方负责类型校验。
    """
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)
