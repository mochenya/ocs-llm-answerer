"""OCS HTTP 请求和响应契约。"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ocs_llm_answerer.answer.models import OCSQuestionType


class AnswerRequest(BaseModel):
    """兼容 OCS 字段别名和选项容器的外部请求。"""

    title: str = Field(min_length=1)
    question_type: OCSQuestionType | None = Field(default=None, alias="type")
    options: list[str] | str | None = None

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    @field_validator("title")
    @classmethod
    def title_must_contain_text(cls, value: str) -> str:
        """拒绝空白题干，不改写原始内容。

        Args:
            value: 已通过类型和长度校验的题干。

        Returns:
            未经标准化的题干。

        Raises:
            ValueError: 题干仅包含空白。
        """
        if not value.strip():
            raise ValueError("title must contain non-whitespace text")
        return value

    @field_validator(
        "question_type", mode="before", json_schema_input_type=OCSQuestionType | Literal[""] | None
    )
    @classmethod
    def blank_question_type_is_unknown(cls, value: object) -> object:
        """兼容 OCS 将未知题型表示为空字符串的情况。

        Args:
            value: 尚未进行枚举校验的外部输入。

        Returns:
            空字符串转换为 None，其他输入保持原样。
        """
        return None if value == "" else value


class AnswerResponse(BaseModel):
    """保持既有 OCS handler 所需的响应字段。"""

    code: int = 1
    question: str
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
    cache_hit: bool


class ErrorResponse(BaseModel):
    """统一描述公开错误及兼容既有字段校验错误列表。"""

    detail: str | list[dict[str, Any]]
