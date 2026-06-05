import asyncio
import warnings

from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.core.models import AnswerRequest
from ocs_llm_answerer.llm.answer_format import format_answers_for_ocs, normalize_answer_for_ocs
from ocs_llm_answerer.llm.openai_responses import (
    OpenAIAnswerPayload,
    OpenAIResponsesProvider,
    _extract_usage,
)
from ocs_llm_answerer.llm.prompting import SYSTEM_PROMPT, build_user_input


def test_system_prompt_describes_ocs_answer_format():
    assert SYSTEM_PROMPT.startswith("你是一名全能的网课题目解答员")
    assert 'multiple：answers 写多个选项编号，按选项顺序排列，例如 ["A", "C"]' in SYSTEM_PROMPT
    assert "completion：answers 写需要填入的文本；多个空分别写多个元素" in SYSTEM_PROMPT
    assert '不要写 "A#C" 或 "A,C"' in SYSTEM_PROMPT
    assert '"answers": ["A"]' in SYSTEM_PROMPT
    assert "后端会把答案列表转换" not in SYSTEM_PROMPT
    assert "不要使用 markdown 或代码块" in SYSTEM_PROMPT


def test_build_user_input_uses_delimited_sections():
    request = AnswerRequest(
        title="1+1=?",
        type="single",
        options=["A. 2", "B. 3"],
    )

    user_input = build_user_input(request)

    assert "<question>" in user_input
    assert "题型：single" in user_input
    assert "<options>\nA. 2\nB. 3\n</options>" in user_input


def test_normalize_multiple_choice_answer_uses_ocs_separator():
    request = AnswerRequest(title="选择正确项", type="multiple")

    assert normalize_answer_for_ocs(request, "A,C") == "A#C"
    assert normalize_answer_for_ocs(request, "AC") == "A#C"
    assert normalize_answer_for_ocs(request, '["A", "C"]') == "A#C"


def test_normalize_completion_json_array_uses_ocs_separator():
    request = AnswerRequest(title="填空", type="completion")

    assert normalize_answer_for_ocs(request, '["张三", "李四"]') == "张三#李四"


def test_normalize_does_not_rewrite_completion_text():
    request = AnswerRequest(title="填空", type="completion")

    assert normalize_answer_for_ocs(request, "张三，李四") == "张三，李四"


def test_normalize_structured_single_answer_uses_first_item():
    request = AnswerRequest(title="1+1=?", type="single")

    assert format_answers_for_ocs(request, ["A"]) == "A"


def test_normalize_structured_multiple_answer_uses_ocs_separator():
    request = AnswerRequest(title="选择正确项", type="multiple")

    assert format_answers_for_ocs(request, ["A", "C"]) == "A#C"
    assert format_answers_for_ocs(request, ["A,C"]) == "A#C"
    assert format_answers_for_ocs(request, ["AC"]) == "A#C"


def test_normalize_structured_judgement_answer_is_boolean_string():
    request = AnswerRequest(title="太阳从东方升起", type="judgement")

    assert format_answers_for_ocs(request, ["True"]) == "true"
    assert format_answers_for_ocs(request, ["错误"]) == "false"


def test_normalize_structured_completion_answer_uses_ocs_separator():
    request = AnswerRequest(title="填空", type="completion")

    assert format_answers_for_ocs(request, ["张三", "李四"]) == "张三#李四"


def test_openai_answer_payload_strips_empty_answers():
    payload = OpenAIAnswerPayload(
        answers=[" A ", ""],
        explanation="依据",
        confidence=0.8,
    )

    assert payload.answers == ["A"]


def test_openai_answer_payload_schema_guides_model_output():
    schema = OpenAIAnswerPayload.model_json_schema()
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


def test_openai_responses_provider_uses_pydantic_text_format():
    config = ProviderConfig(
        adapter="openai_responses",
        base_url="https://fake.local/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
    )
    provider = OpenAIResponsesProvider("fake", config, "fake-key")
    fake_client = _FakeOpenAIClient(
        OpenAIAnswerPayload(
            answers=["A", "C"],
            explanation="list 和 dict 是可变类型。",
            confidence=0.95,
        )
    )
    provider._client = fake_client

    result = asyncio.run(
        provider.answer(
            AnswerRequest(
                title="以下哪些是 Python 可变数据类型？",
                type="multiple",
                options=["A. list", "B. tuple", "C. dict", "D. str"],
            )
        )
    )

    kwargs = fake_client.responses.with_raw_response.kwargs
    assert kwargs["text_format"] is OpenAIAnswerPayload
    assert kwargs["store"] is False
    assert "text" not in kwargs
    assert result.answer.answer == "A#C"
    assert result.response_body_raw == '{"raw":true}'
    assert result.http_status == 200
    assert result.usage.total_tokens == 30


def test_openai_responses_provider_falls_back_to_output_text_payload():
    config = ProviderConfig(
        adapter="openai_responses",
        base_url="https://fake.local/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
    )
    provider = OpenAIResponsesProvider("fake", config, "fake-key")
    fake_client = _FakeOpenAIClientWithOutputText(
        '{"answers":["A"],"explanation":"依据","confidence":0.8}'
    )
    provider._client = fake_client

    result = asyncio.run(provider.answer(AnswerRequest(title="1+1=?", type="single")))

    assert result.answer.answer == "A"
    assert result.answer.explanation == "依据"
    assert result.answer.confidence == 0.8


def test_openai_responses_provider_attaches_raw_body_to_parse_errors():
    config = ProviderConfig(
        adapter="openai_responses",
        base_url="https://fake.local/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
    )
    provider = OpenAIResponsesProvider("fake", config, "fake-key")
    provider._client = _FakeOpenAIClientWithOutputText("not-json")

    try:
        asyncio.run(provider.answer(AnswerRequest(title="1+1=?", type="single")))
    except RuntimeError as exc:
        assert exc.raw_body == '{"raw":true}'
        assert "ValidationError" in str(exc)
    else:
        raise AssertionError("provider.answer() should raise on malformed structured output")


def test_extract_usage_reads_responses_token_details():
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


def test_extract_usage_does_not_serialize_parsed_response():
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


class _FakeOpenAIClient:
    def __init__(self, payload: OpenAIAnswerPayload) -> None:
        self.responses = _FakeResponses(payload)


class _FakeOpenAIClientWithOutputText:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponsesWithOutputText(output_text)


class _FakeResponses:
    def __init__(self, payload: OpenAIAnswerPayload) -> None:
        self.with_raw_response = _FakeWithRawResponse(payload)


class _FakeResponsesWithOutputText:
    def __init__(self, output_text: str) -> None:
        self.with_raw_response = _FakeWithRawResponseOutputText(output_text)


class _FakeWithRawResponse:
    def __init__(self, payload: OpenAIAnswerPayload) -> None:
        self._payload = payload
        self.kwargs: dict[str, object] = {}

    async def parse(self, **kwargs: object) -> _FakeRawResponse:
        self.kwargs = kwargs
        return _FakeRawResponse(self._payload)


class _FakeWithRawResponseOutputText:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    async def parse(self, **kwargs: object) -> _FakeRawResponseOutputText:
        return _FakeRawResponseOutputText(self._output_text)


class _FakeRawResponse:
    status_code = 200
    text = '{"raw":true}'

    def __init__(self, payload: OpenAIAnswerPayload) -> None:
        self._payload = payload

    def parse(self) -> _FakeParsedResponse:
        return _FakeParsedResponse(self._payload)


class _FakeRawResponseOutputText:
    status_code = 200
    text = '{"raw":true}'

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def parse(self) -> _FakeParsedResponseOutputText:
        return _FakeParsedResponseOutputText(self._output_text)


class _FakeParsedResponse:
    def __init__(self, payload: OpenAIAnswerPayload) -> None:
        self.output_parsed = payload
        self.usage = {
            "input_tokens": 10,
            "input_tokens_details": {
                "cached_tokens": 2,
            },
            "output_tokens": 20,
            "total_tokens": 30,
        }


class _FakeParsedResponseOutputText:
    def __init__(self, output_text: str) -> None:
        self.output_parsed = None
        self.output_text = output_text
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
            "input_tokens_details": {
                "cached_tokens": 2,
            },
            "output_tokens": 20,
            "total_tokens": 30,
        }

    def model_dump(self, *, mode: str) -> dict[str, object]:
        self.model_dump_called = True
        warnings.warn("Pydantic serializer warnings:", UserWarning, stacklevel=2)
        return {"usage": self.usage}
