from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator


class OCSQuestionType(StrEnum):
    """OCS 题库集成传入的题目类型取值。"""

    SINGLE = "single"
    MULTIPLE = "multiple"
    JUDGEMENT = "judgement"
    COMPLETION = "completion"


class AnswerRequest(BaseModel):
    """兼容 OCS 的请求体。

    OCS 传入的字段名是 ``type``。内部使用 ``question_type``，让服务代码里的领域含义更清楚。
    """

    title: str = Field(min_length=1)
    question_type: OCSQuestionType | None = Field(default=None, alias="type")
    options: list[str] | str | None = None
    _raw_payload_json: str | None = PrivateAttr(default=None)

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("question_type", mode="before")
    @classmethod
    def blank_question_type_is_unknown(cls, value: object) -> object:
        """OCS 文档标注 type 可选，实际也可能传入空字符串。"""
        if value == "":
            return None
        return value

    @property
    def raw_payload_json(self) -> str | None:
        return self._raw_payload_json

    def set_raw_payload_json(self, value: str) -> None:
        self._raw_payload_json = value


class LLMAnswer(BaseModel):
    """任意 LLM provider 返回的标准化答案。"""

    answer: str = Field(min_length=1)
    explanation: str = ""
    confidence: float = Field(ge=0, le=1)


class LLMUsage(BaseModel):
    """LLM provider 响应报告的 token 用量。"""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class LLMCallResult(BaseModel):
    """Provider 原始响应数据和标准化答案。"""

    answer: LLMAnswer
    response_body_raw: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMProviderMetadata(BaseModel):
    """审计单次 provider 调用所需的配置细节。"""

    adapter: str
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: float
    max_retries: int
    extra_body: dict[str, Any] | None = None


class NormalizedQuestion(BaseModel):
    """为哈希、缓存和请求日志标准化后的题目字段。"""

    question_hash: str
    question: str
    question_type: str | None
    options_json: str | None
    question_raw_json: str | None = None


class AnswerResponse(BaseModel):
    """OCS 自定义题库处理函数期望的响应结构。"""

    code: int = 1
    question: str
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
    cache_hit: bool


class CachedAnswer(BaseModel):
    """已持久化的缓存记录。"""

    question_hash: str
    call_hash: str
    question: str
    question_type: str | None
    options_json: str | None
    question_raw_json: str | None = None
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
