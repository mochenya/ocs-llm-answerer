"""Provider 的通用配置快照和调用结果，不依赖具体 SDK。"""

from typing import Any

from pydantic import BaseModel, Field


class LLMAnswer(BaseModel):
    """模型的结构化答案，题型语义由答题层校验。

    Attributes:
        answers: 原始答案项，空项不得在业务校验之前被丢弃。
        explanation: 模型给出的依据。
        confidence: 模型自报置信度。
    """

    answers: list[str]
    explanation: str = ""
    confidence: float = Field(ge=0, le=1)


class LLMUsage(BaseModel):
    """上游报告的 token 用量，未报告的字段为 None。"""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class LLMCallResult(BaseModel):
    """模型调用结果及其审计信息。

    Attributes:
        answer: 尚未进行题型校验的结构化答案。
        response_body_raw: 原始响应文本，允许非 JSON 内容。
        http_status: 上游 HTTP 状态码。
        usage: 上游报告的 token 用量。
    """

    answer: LLMAnswer
    response_body_raw: str | None = None
    http_status: int | None = Field(default=None, ge=100, le=599)
    usage: LLMUsage = Field(default_factory=LLMUsage)


class LLMProviderMetadata(BaseModel):
    """用于审计的 Provider 配置快照，只记录密钥环境变量名而非密钥。"""

    adapter: str
    base_url: str
    api_key_env: str
    model: str
    timeout_seconds: float
    max_retries: int
    extra_body: dict[str, Any] | None = None
