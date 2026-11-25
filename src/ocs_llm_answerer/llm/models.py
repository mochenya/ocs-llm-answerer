"""业务无关的模型输入原语、调用结果和配置快照，不依赖具体 SDK 或答题模型。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class TextPart:
    """单段文本输入。"""

    text: str


@dataclass(frozen=True)
class ImagePart:
    """与具体厂商传输格式无关的 Base64 图片输入。"""

    media_type: str
    base64_data: str = field(repr=False)

    @property
    def data_url(self) -> str:
        """按需构造标准 Data URL，供 OpenAI 等需要 Data URL 的协议使用。"""
        return f"data:{self.media_type};base64,{self.base64_data}"


type ContentPart = TextPart | ImagePart


@dataclass(frozen=True)
class UserInput:
    """一次发送给模型会话的用户输入，由有序文本和图片内容块组成。"""

    parts: tuple[ContentPart, ...]

    @classmethod
    def from_text(cls, text: str) -> UserInput:
        """构造纯文本用户输入。"""
        return cls(parts=(TextPart(text=text),))


class LLMUsage(BaseModel):
    """上游报告的 token 用量，未报告的字段为 None。"""

    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class LLMCallResult[TOutput: BaseModel](BaseModel):
    """泛型模型调用结果及其审计信息。

    Attributes:
        parsed: 已通过目标 Pydantic Schema 解析的结构化输出。
        response_body_raw: 原始响应文本，允许非 JSON 内容。
        http_status: 上游 HTTP 状态码。
        usage: 上游报告的 token 用量。
    """

    parsed: TOutput
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
