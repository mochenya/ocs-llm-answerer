from __future__ import annotations

import json

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.core.models import AnswerRequest, LLMAnswer, LLMCallResult, LLMUsage
from ocs_llm_answerer.llm.answer_format import format_answers_for_ocs
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

    @field_validator("answers")
    @classmethod
    def answers_not_empty(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("answers must not be empty")
        return cleaned


class OpenAIResponsesProvider:
    """隐藏在 LLMProvider 契约后的 OpenAI Responses API adapter。"""

    def __init__(self, name: str, config: ProviderConfig, api_key: str) -> None:
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

    async def answer(self, request: AnswerRequest) -> LLMCallResult:
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
        response_body_raw = raw_response.text
        try:
            response = raw_response.parse()
            payload = _extract_parsed_answer_payload(response)
        except Exception as exc:
            _raise_with_raw(exc, response_body_raw)
        llm_answer = LLMAnswer(
            answer=format_answers_for_ocs(request, payload.answers),
            explanation=payload.explanation,
            confidence=payload.confidence,
        )
        return LLMCallResult(
            answer=llm_answer,
            response_body_raw=response_body_raw,
            http_status=raw_response.status_code,
            usage=_extract_usage(response),
        )


class _RawResponseError(RuntimeError):
    """把解析或校验错误与 LLM 原始响应体一起封装。"""

    def __init__(self, message: str, raw_body: str) -> None:
        super().__init__(message)
        self.raw_body = raw_body


def _raise_with_raw(exc: Exception, raw_body: str) -> None:
    raise _RawResponseError(
        f"{type(exc).__name__}: {exc}",
        raw_body,
    ) from exc


def _extract_parsed_answer_payload(response: object) -> OpenAIAnswerPayload:
    parsed = getattr(response, "output_parsed", None)
    if isinstance(parsed, OpenAIAnswerPayload):
        return parsed

    output_text = _extract_output_text(response)
    return OpenAIAnswerPayload.model_validate_json(output_text)


def _extract_output_text(response: object) -> str:
    """同时从 SDK 便捷字段和结构化输出项中读取文本。"""
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct.strip():
        return direct

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "output_text":
                text = getattr(content, "text", "")
                if text:
                    return text

    dumped = (
        response.model_dump_json() if hasattr(response, "model_dump_json") else json.dumps(response)
    )
    raise RuntimeError(f"OpenAI response did not contain output text: {dumped}")


def _extract_usage(response: object) -> LLMUsage:
    usage = _get_field(response, "usage")
    input_token_details = _get_field(usage, "input_tokens_details")
    return LLMUsage(
        input_tokens=_get_field(usage, "input_tokens"),
        output_tokens=_get_field(usage, "output_tokens"),
        total_tokens=_get_field(usage, "total_tokens"),
        cached_tokens=_get_field(input_token_details, "cached_tokens"),
    )


def _get_field(value: object, field_name: str) -> object:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)
