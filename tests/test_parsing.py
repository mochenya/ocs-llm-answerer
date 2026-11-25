"""验证业务无关的容错结构化输出解析器。"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ocs_llm_answerer.llm.parsing import parse_structured_output


class _SamplePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answers: list[str]
    explanation: str
    confidence: float = Field(ge=0, le=1)


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"answers":["A"],"explanation":"直接 JSON","confidence":0.9}',
        '  \n{"answers":["A"],"explanation":"带外围空白","confidence":0.9}\n ',
        '```json\n{"answers":["A"],"explanation":"json 代码块","confidence":0.9}\n```',
        '```JSON\n{"answers":["A"],"explanation":"大写 JSON 代码块","confidence":0.9}\n```',
        '```\n{"answers":["A"],"explanation":"无语言标记代码块","confidence":0.9}\n```',
        '思考完毕，答案如下：\n```json\n{"answers":["A"],"explanation":"带前后说明","confidence":0.9}\n```\n请查收。',
        '经过分析，最终输出为 {"answers":["A"],"explanation":"内嵌 JSON","confidence":0.9} 以上。',
    ],
)
def test_parse_structured_output_accepts_json_fences_and_wrapped_text(raw_text: str) -> None:
    parsed = parse_structured_output(raw_text, _SamplePayload)
    assert parsed.answers == ["A"]
    assert parsed.confidence == 0.9


def test_parse_structured_output_prefers_schema_error_inside_code_fence() -> None:
    """当代码块内 JSON 语法合法但字段违反 Schema 时，保留精确字段错误供纠正重试使用。"""
    raw_text = '```json\n{"answers":["A"],"explanation":"越界","confidence":1.5}\n```'
    with pytest.raises(ValidationError) as exc_info:
        parse_structured_output(raw_text, _SamplePayload)
    assert exc_info.value.errors()[0]["loc"] == ("confidence",)


@pytest.mark.parametrize("raw_text", ["", "   ", "完全不是 JSON", "```json\nnot-json\n```"])
def test_parse_structured_output_rejects_unparseable_text(raw_text: str) -> None:
    with pytest.raises(ValidationError):
        parse_structured_output(raw_text, _SamplePayload)
