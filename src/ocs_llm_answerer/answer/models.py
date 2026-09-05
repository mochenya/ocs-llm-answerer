"""答题用例的数据模型，不包含 HTTP 字段别名或数据库 JSON 编码。"""

from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class OCSQuestionType(StrEnum):
    """当前答题业务支持的题型。"""

    SINGLE = "single"
    MULTIPLE = "multiple"
    JUDGEMENT = "judgement"
    COMPLETION = "completion"


class Question(BaseModel):
    """供服务和 Provider 使用的题目，不携带原始 HTTP 载荷。

    Attributes:
        title: 题干，内部有意义的空白需要保留。
        question_type: 已识别的题型，未知时为 None。
        options: 有序选项列表，传输层的字符串形式应先转换。
    """

    title: str = Field(min_length=1)
    question_type: OCSQuestionType | None = None
    options: list[str] | None = None

    @field_validator("title")
    @classmethod
    def title_must_contain_text(cls, value: str) -> str:
        """保护非 HTTP 调用入口，不允许空白题干进入业务流程。

        Args:
            value: 待检查的题干，不在此处改写空白。

        Returns:
            保持原样的非空白题干。

        Raises:
            ValueError: 题干仅含空白。
        """
        if not value.strip():
            raise ValueError("title must contain non-whitespace text")
        return value


@dataclass(frozen=True)
class RequestAudit:
    """与业务题目分离的请求快照。

    Attributes:
        raw_payload_json: 未经改写的 UTF-8 JSON；非 HTTP 调用可以不提供。
    """

    raw_payload_json: str | None = None


class NormalizedQuestion(BaseModel):
    """已计算稳定身份的题目，选项仍保持结构化列表。"""

    question_hash: str
    question: str
    question_type: OCSQuestionType | None
    options: list[str] | None


class AnswerResult(BaseModel):
    """一次答题的业务结果，不包含 OCS 的 HTTP 成功标识。"""

    question: str
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
    cache_hit: bool


class CachedAnswer(BaseModel):
    """已校验且引用成功调用的答案，不暴露数据库列的编码形式。

    Attributes:
        question_hash: 标准化题目的稳定身份。
        llm_request_id: 已提交的调用流水主键。
        question: 标准化题干。
        question_type: 题型，未知时为 None。
        options: 有序选项，由仓储负责 JSON 编解码。
        answer: 已转换为 OCS 格式的合法答案。
        explanation: 模型给出的依据。
        confidence: 模型自报置信度。
        provider: 答案来源 Provider 名称。
        model: 答案来源模型名称。
    """

    question_hash: str
    llm_request_id: int = Field(gt=0)
    question: str
    question_type: OCSQuestionType | None
    options: list[str] | None
    answer: str
    explanation: str
    confidence: float
    provider: str
    model: str
