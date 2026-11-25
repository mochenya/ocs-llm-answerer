"""验证提示词构造、AnswerPayload Schema、OpenAI Responses 会话原语与思考块保留。"""

from __future__ import annotations

import asyncio
import json
import warnings
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from ocs_llm_answerer.answer.models import AnswerPayload, OCSQuestionType, Question, QuestionImage
from ocs_llm_answerer.answer.prompting import (
    SYSTEM_PROMPT,
    build_question_input,
    build_question_text,
)
from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.llm.adapters.openai_responses import (
    OpenAIResponsesClient,
    _extract_usage,
)
from ocs_llm_answerer.llm.errors import (
    ProviderOutputError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ocs_llm_answerer.llm.models import UserInput


def _make_config(extra_body: dict[str, object] | None = None) -> ProviderConfig:
    return ProviderConfig(
        adapter="openai_responses",
        base_url="https://fake.local/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
        extra_body=extra_body,
    )


def test_system_prompt_describes_ocs_answer_format() -> None:
    assert SYSTEM_PROMPT.startswith("你是一名全能的网课题目解答员")
    assert 'multiple：answers 写多个选项编号，按选项顺序排列，例如 ["A", "C"]' in SYSTEM_PROMPT
    assert "completion：answers 写需要填入的文本；多个空分别写多个元素" in SYSTEM_PROMPT
    assert '不要写 "A#C" 或 "A,C"' in SYSTEM_PROMPT
    assert '"answers": ["A"]' in SYSTEM_PROMPT
    assert "后端会把答案列表转换" not in SYSTEM_PROMPT
    assert "不要使用 markdown 或代码块" in SYSTEM_PROMPT


def test_build_question_text_uses_delimited_sections() -> None:
    request = Question(
        title="1+1=?",
        question_type=OCSQuestionType.SINGLE,
        options=["A. 2", "B. 3"],
    )

    user_text = build_question_text(request)

    assert "<question>" in user_text
    assert "题型：single" in user_text
    assert "<options>\nA. 2\nB. 3\n</options>" in user_text


def test_answer_payload_preserves_items_for_business_validation() -> None:
    """结构解析不删除空项，避免填空答案在业务校验前已经发生错位。"""
    payload = AnswerPayload(
        answers=[" A ", ""],
        explanation="依据",
        confidence=0.8,
    )

    assert payload.answers == [" A ", ""]


def test_answer_payload_schema_guides_model_output() -> None:
    schema = AnswerPayload.model_json_schema()
    answers_description = schema["properties"]["answers"]["description"]
    explanation_description = schema["properties"]["explanation"]["description"]
    confidence_description = schema["properties"]["confidence"]["description"]

    assert schema["title"] == "OCSQuestionAnswer"
    assert schema["description"] == (
        "Structured answer payload returned by the LLM for one OCS question."
    )
    assert answers_description == (
        "Final answer items. Use one item for single/judgement; multiple items "
        "for multiple-choice or multi-blank completion."
    )
    assert explanation_description == "Brief Chinese explanation for the answer."
    assert confidence_description == "Confidence score from 0 to 1."


@pytest.mark.parametrize("with_images", [False, True])
def test_openai_responses_session_uses_pydantic_text_format_without_hardcoded_sampling(
    with_images: bool,
) -> None:
    """使用目标 Pydantic 模型作为 text_format，不硬编码 temperature 或 max_output_tokens。"""
    extra_body = {"reasoning": {"effort": "low"}}
    client = OpenAIResponsesClient("fake", _make_config(extra_body=extra_body), "fake-key")
    fake_sdk = _ScriptedOpenAIClient(
        [
            _FakeRawResponse(
                parsed=AnswerPayload(
                    answers=["A", "C"],
                    explanation="list 和 dict 是可变类型。",
                    confidence=0.95,
                )
            )
        ]
    )
    client._client = fake_sdk

    images = (
        [
            QuestionImage("image/png", "YQ==", "a"),
            QuestionImage("image/png", "Yg==", "b"),
        ]
        if with_images
        else []
    )
    question = Question(
        title="以下哪些是 Python 可变数据类型？",
        question_type=OCSQuestionType.MULTIPLE,
        options=["A. list", "B. tuple", "C. dict", "D. str"],
        images=images,
    )
    session = client.create_session(instructions=SYSTEM_PROMPT)
    result = asyncio.run(session.send_structured(build_question_input(question), AnswerPayload))

    kwargs = fake_sdk.responses.with_raw_response.calls[0]
    assert kwargs["text_format"] is AnswerPayload
    assert kwargs["store"] is False
    assert kwargs["extra_body"] == extra_body
    assert "temperature" not in kwargs
    assert "max_output_tokens" not in kwargs
    assert "text" not in kwargs
    if with_images:
        assert kwargs["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_question_text(question)},
                    {"type": "input_text", "text": "[图片1]"},
                    {"type": "input_image", "image_url": images[0].data_url, "detail": "auto"},
                    {"type": "input_text", "text": "[图片2]"},
                    {"type": "input_image", "image_url": images[1].data_url, "detail": "auto"},
                ],
            }
        ]
    else:
        assert kwargs["input"] == [{"role": "user", "content": build_question_text(question)}]
    assert result.parsed.answers == ["A", "C"]
    assert result.response_body_raw == '{"raw":true}'
    assert result.http_status == 200
    assert result.usage.total_tokens == 30


def test_openai_responses_session_parses_fenced_json_fallback() -> None:
    """上游未返回 output_parsed 且文本包裹在 ```json 代码块中时也能正确解析。"""
    client = OpenAIResponsesClient("fake", _make_config(), "fake-key")
    client._client = _ScriptedOpenAIClient(
        [
            _FakeRawResponse(
                output_text='```json\n{"answers":["A"],"explanation":"依据","confidence":0.8}\n```'
            )
        ]
    )

    session = client.create_session(instructions=SYSTEM_PROMPT)
    result = asyncio.run(session.send_structured(UserInput.from_text("1+1=?"), AnswerPayload))

    assert result.parsed.answers == ["A"]
    assert result.parsed.explanation == "依据"
    assert result.parsed.confidence == 0.8


def test_openai_responses_session_preserves_reasoning_and_phase_across_retries() -> None:
    """多轮纠正重试时，直接从原始响应提取完整 output 数组（含 reasoning 思考块与 phase）。"""
    reasoning_item = {
        "id": "rs_123",
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "先看选项"}],
        "encrypted_content": "enc_abc",
        "status": "completed",
    }
    bad_message_item = {
        "id": "msg_123",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": "final_answer",
        "content": [{"type": "output_text", "text": "not-json"}],
    }
    first_raw = json.dumps({"output": [reasoning_item, bad_message_item]}, ensure_ascii=False)
    second_payload = AnswerPayload(answers=["A"], explanation="已修正", confidence=0.92)

    client = OpenAIResponsesClient("fake", _make_config(), "fake-key")
    fake_sdk = _ScriptedOpenAIClient(
        [
            _FakeRawResponse(output_text="not-json", raw_text=first_raw),
            _FakeRawResponse(parsed=second_payload),
        ]
    )
    client._client = fake_sdk

    session = client.create_session(instructions=SYSTEM_PROMPT)
    with pytest.raises(ProviderOutputError) as exc_info:
        asyncio.run(session.send_structured(UserInput.from_text("题目原文"), AnswerPayload))

    assert exc_info.value.raw_body == first_raw
    assert exc_info.value.status_code == 200
    assert exc_info.value.usage.total_tokens == 3
    assert "ValidationError" in str(exc_info.value)

    second_result = asyncio.run(
        session.send_structured(UserInput.from_text("请输出合法 JSON"), AnswerPayload)
    )
    assert second_result.parsed == second_payload
    assert fake_sdk.responses.with_raw_response.calls[1]["input"] == [
        {"role": "user", "content": "题目原文"},
        reasoning_item,
        bad_message_item,
        {"role": "user", "content": "请输出合法 JSON"},
    ]


def test_separate_sessions_do_not_share_message_history() -> None:
    """同一个 OpenAIResponsesClient 创建的不同会话彼此完全隔离。"""
    payload = AnswerPayload(answers=["A"], explanation="ok", confidence=0.9)
    client = OpenAIResponsesClient("fake", _make_config(), "fake-key")
    fake_sdk = _ScriptedOpenAIClient(
        [_FakeRawResponse(parsed=payload), _FakeRawResponse(parsed=payload)]
    )
    client._client = fake_sdk

    first_session = client.create_session(instructions="inst-1")
    second_session = client.create_session(instructions="inst-2")
    asyncio.run(first_session.send_structured(UserInput.from_text("题 1"), AnswerPayload))
    asyncio.run(second_session.send_structured(UserInput.from_text("题 2"), AnswerPayload))

    assert fake_sdk.responses.with_raw_response.calls[0]["input"] == [
        {"role": "user", "content": "题 1"}
    ]
    assert fake_sdk.responses.with_raw_response.calls[1]["input"] == [
        {"role": "user", "content": "题 2"}
    ]


def test_session_history_is_not_polluted_by_transport_errors() -> None:
    """上游网络超时或连接失败不得向会话历史写入半残消息。"""
    import httpx
    from openai import APITimeoutError

    payload = AnswerPayload(answers=["A"], explanation="ok", confidence=0.9)
    client = OpenAIResponsesClient("fake", _make_config(), "fake-key")
    timeout_error = APITimeoutError(request=httpx.Request("POST", "https://fake.local/v1"))
    fake_sdk = _ScriptedOpenAIClient([timeout_error, _FakeRawResponse(parsed=payload)])
    client._client = fake_sdk

    session = client.create_session(instructions=SYSTEM_PROMPT)
    with pytest.raises(ProviderTimeoutError):
        asyncio.run(session.send_structured(UserInput.from_text("第一次尝试"), AnswerPayload))

    result = asyncio.run(session.send_structured(UserInput.from_text("第二次重试"), AnswerPayload))
    assert result.parsed == payload
    assert fake_sdk.responses.with_raw_response.calls[1]["input"] == [
        {"role": "user", "content": "第二次重试"}
    ]


def test_extract_usage_reads_responses_token_details() -> None:
    usage = _extract_usage(
        {
            "usage": {
                "input_tokens": 265,
                "input_tokens_details": {
                    "cached_tokens": 192,
                },
                "output_tokens": 654,
                "total_tokens": 919,
            }
        }
    )

    assert usage.input_tokens == 265
    assert usage.output_tokens == 654
    assert usage.total_tokens == 919
    assert usage.cached_tokens == 192


def test_extract_usage_does_not_serialize_parsed_response() -> None:
    response = _ResponseWithWarningDump()

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        usage = _extract_usage(response)

    assert captured == []
    assert usage.input_tokens == 10
    assert usage.output_tokens == 20
    assert usage.total_tokens == 30
    assert usage.cached_tokens == 2
    assert not response.model_dump_called


def test_invalid_usage_keeps_raw_response_in_non_retryable_provider_error() -> None:
    """用量解析失败应保留原始响应，并抛出不可纠正重试的 ProviderResponseError。"""
    config = _make_config()

    async def exercise() -> None:
        client = OpenAIResponsesClient("fake", config, "fake-key")
        await client.aclose()
        response = _FakeParsedResponse(
            parsed=AnswerPayload(answers=["A"], explanation="", confidence=0.9)
        )
        response.usage = {"input_tokens": -1}
        raw = SimpleNamespace(text="exact raw body", status_code=200, parse=lambda: response)
        fake_sdk = SimpleNamespace(
            responses=SimpleNamespace(
                with_raw_response=SimpleNamespace(parse=AsyncMock(return_value=raw))
            )
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(client, "_client", fake_sdk)
            session = client.create_session(instructions=SYSTEM_PROMPT)
            with pytest.raises(ProviderResponseError) as raised:
                await session.send_structured(UserInput.from_text("题目"), AnswerPayload)
        assert not isinstance(raised.value, ProviderOutputError)
        assert raised.value.raw_body == "exact raw body"
        assert raised.value.status_code == 200
        assert raised.value.__cause__ is not None

    asyncio.run(exercise())


class _ScriptedOpenAIClient:
    def __init__(self, responses: list[_FakeRawResponse | Exception]) -> None:
        self.responses = SimpleNamespace(with_raw_response=_ScriptedWithRawResponse(responses))


class _ScriptedWithRawResponse:
    def __init__(self, responses: list[_FakeRawResponse | Exception]) -> None:
        self._responses = responses
        self.calls: list[dict[str, object]] = []

    async def parse(self, **kwargs: object) -> _FakeRawResponse:
        index = len(self.calls)
        self.calls.append(kwargs)
        outcome = self._responses[index]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeRawResponse:
    status_code = 200

    def __init__(
        self,
        *,
        parsed: AnswerPayload | None = None,
        output_text: str | None = None,
        raw_text: str = '{"raw":true}',
    ) -> None:
        self._parsed = parsed
        self._output_text = output_text
        self.text = raw_text

    def parse(self) -> _FakeParsedResponse:
        return _FakeParsedResponse(parsed=self._parsed, output_text=self._output_text)


class _FakeParsedResponse:
    def __init__(
        self,
        *,
        parsed: AnswerPayload | None = None,
        output_text: str | None = None,
    ) -> None:
        self.output_parsed = parsed
        self.output_text = output_text
        if parsed is not None:
            self.usage: dict[str, object] = {
                "input_tokens": 10,
                "input_tokens_details": {"cached_tokens": 2},
                "output_tokens": 20,
                "total_tokens": 30,
            }
        else:
            self.usage = {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            }


class _ResponseWithWarningDump:
    def __init__(self) -> None:
        self.model_dump_called = False
        self.usage = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 20,
            "total_tokens": 30,
        }

    def model_dump(self, *, mode: str) -> dict[str, object]:
        self.model_dump_called = True
        warnings.warn("Pydantic serializer warnings:", UserWarning, stacklevel=2)
        return {"usage": self.usage}
