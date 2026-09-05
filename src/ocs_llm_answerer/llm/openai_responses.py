from __future__ import annotations

import json
from typing import Never

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from ocs_llm_answerer.answer.models import Question
from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.llm.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ocs_llm_answerer.llm.models import LLMAnswer, LLMCallResult, LLMUsage
from ocs_llm_answerer.llm.prompting import SYSTEM_PROMPT, build_user_input


class OpenAIAnswerPayload(BaseModel):
    """LLM 针对单道 OCS 题目返回的结构化答案载荷。"""

    model_config = ConfigDict(
        extra="forbid",
        title="OCSQuestionAnswer",
        json_schema_extra={
            "description": "Structured answer payload returned by the LLM for one OCS question."
        },
    )

    answers: list[str] = Field(
        description=(
            "Final answer items. Use one item for single/judgement; multiple items "
            "for multiple-choice or multi-blank completion."
        ),
    )
    explanation: str = Field(
        description="Brief Chinese explanation for the answer.",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Confidence score from 0 to 1.",
    )


class OpenAIResponsesProvider:
    """隐藏在 LLMProvider 契约后的 OpenAI Responses API adapter。"""

    def __init__(self, name: str, config: ProviderConfig, api_key: str) -> None:
        """创建由本适配器持有的异步客户端。

        Args:
            name: 配置中的 Provider 名称。
            config: 已校验的模型调用配置。
            api_key: 运行时读取的访问密钥，不写入审计记录。
        """
        self.name = name
        self.model = config.model
        self.request_metadata = config.to_metadata()
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            max_retries=config.max_retries,
        )
        self._extra_body = config.extra_body

    async def aclose(self) -> None:
        """关闭本适配器的 HTTP 客户端，关闭后不可继续调用模型。"""
        await self._client.close()

    async def answer(self, request: Question) -> LLMCallResult:
        """获取结构化模型输出，将题型校验和 OCS 转换交给答题层。

        Args:
            request: 已标准化的题目请求。

        Returns:
            未拼接的答案项及原始响应、HTTP 状态和用量。

        Raises:
            ProviderTimeoutError: 上游调用超时。
            ProviderUnavailableError: 连接失败、限流或上游暂时不可用。
            ProviderResponseError: 上游响应错误或结构解析失败。
        """
        try:
            raw_response = await self._client.responses.with_raw_response.parse(
                model=self.model,
                instructions=SYSTEM_PROMPT,
                input=build_user_input(request),
                store=False,
                temperature=0.1,
                max_output_tokens=800,
                text_format=OpenAIAnswerPayload,
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
            payload = _extract_parsed_answer_payload(response)
            llm_answer = LLMAnswer(
                answers=payload.answers,
                explanation=payload.explanation,
                confidence=payload.confidence,
            )
            usage = _extract_usage(response)
        except Exception as exc:
            _raise_with_raw(exc, response_body_raw, raw_response.status_code)
        return LLMCallResult(
            answer=llm_answer,
            response_body_raw=response_body_raw,
            http_status=raw_response.status_code,
            usage=usage,
        )


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


def _extract_parsed_answer_payload(response: object) -> OpenAIAnswerPayload:
    """优先使用 SDK 解析结果，兼容只提供输出文本的响应。

    Args:
        response: SDK 返回的响应对象。

    Returns:
        已通过结构校验但尚未通过题型校验的答案。

    Raises:
        ValidationError: 输出文本不符合结构化答案 schema。
        RuntimeError: 响应不包含输出文本。
    """
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, OpenAIAnswerPayload):
        return parsed

    output_text = _extract_output_text(response)
    return OpenAIAnswerPayload.model_validate_json(output_text)


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
