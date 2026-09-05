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

    @field_validator("title")
    @classmethod
    def title_must_contain_text(cls, value: str) -> str:
        """拒绝纯空白题干，并将实际文本标准化留给答题层。

        Args:
            value: 已通过字符串类型及长度校验的题干。

        Returns:
            未改写的题干文本。

        Raises:
            ValueError: 题干不含任何非空白字符。
        """
        if not value.strip():
            raise ValueError("title must contain non-whitespace text")
        return value

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
    """Provider 返回的结构化答案，题型语义由答题层统一校验。

    Attributes:
        answers: 未进行 OCS 拼接的答案项；保留空项以便拒绝不完整的填空答案。
        explanation: 模型提供的简短依据。
        confidence: 模型自报的置信度。
    """

    answers: list[str]
    explanation: str = ""
    confidence: float = Field(ge=0, le=1)


class LLMUsage(BaseModel):
    """LLM provider 响应报告的 token 用量。"""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class LLMCallResult(BaseModel):
    """Provider 结果及审计数据。

    Attributes:
        answer: 未进行题型校验的结构化答案。
        response_body_raw: 原始响应文本，不要求内容为合法 JSON。
        http_status: 上游 HTTP 状态码。
        usage: 响应报告的 token 用量。
    """

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
    """通过题型校验且可追溯到模型调用的缓存答案。

    Attributes:
        question_hash: 稳定的题目身份。
        llm_request_id: 已存在的调用流水整数主键，由数据库外键保证引用有效。
        question: 标准化题干。
        question_type: OCS 题型或未知类型。
        options_json: 有序选项的 JSON 存储形式。
        question_raw_json: 本次请求的原始审计载荷。
        answer: 已转换为 OCS 格式的合法答案。
        explanation: 生成该答案时的模型依据。
        confidence: 生成该答案时的模型自报置信度。
        provider: 生成该答案的 Provider 名称。
        model: 生成该答案的模型名称。
    """

    question_hash: str
    llm_request_id: int = Field(gt=0)
    question: str
    question_type: str | None
    options_json: str | None
    question_raw_json: str | None = None
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
