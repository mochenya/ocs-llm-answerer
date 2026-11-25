"""只使用内存替身验证答题编排、单请求会话与纠正重试循环，不依赖 HTTP、SQLite 或网络。"""

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from pydantic import ValidationError

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.formatting import InvalidAnswerError
from ocs_llm_answerer.answer.models import (
    AnswerPayload,
    CachedAnswer,
    LLMCallAudit,
    OCSQuestionType,
    Question,
    RequestAudit,
)
from ocs_llm_answerer.answer.normalization import build_question_hash, normalize_question
from ocs_llm_answerer.answer.ports import AnswerCachePort, LLMRequestPort
from ocs_llm_answerer.answer.prompting import SYSTEM_PROMPT, build_question_text
from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.llm.errors import (
    ProviderOutputError,
    ProviderResponseError,
    ProviderTimeoutError,
)
from ocs_llm_answerer.llm.models import LLMCallResult, LLMUsage, TextPart
from tests.fakes import StubLLMClient


@dataclass
class ServiceHarness:
    """具有真实协议签名的内存仓储及可观察通用 LLM 客户端。"""

    cache: MagicMock
    requests: MagicMock
    llm_client: StubLLMClient
    service: AnswerService


def _make_harness(
    llm_client: StubLLMClient | None = None,
    *,
    max_attempts: int = 3,
) -> ServiceHarness:
    cache = create_autospec(AnswerCachePort, instance=True)
    requests = create_autospec(LLMRequestPort, instance=True)
    cache.get.return_value = None
    requests.record.return_value = 17
    client = llm_client or StubLLMClient()
    return ServiceHarness(
        cache,
        requests,
        client,
        AnswerService(
            cast(AnswerCachePort, cache),
            client,
            cast(LLMRequestPort, requests),
            max_attempts=max_attempts,
        ),
    )


@pytest.fixture
def harness() -> ServiceHarness:
    """组装不执行 I/O 的服务测试环境。"""
    return _make_harness()


def test_success_normalizes_once_for_llm_and_preserves_audit(harness: ServiceHarness) -> None:
    """模型输入与哈希一致，原始载荷只送入仓储，先记录成功调用再写缓存。"""
    question = Question(
        title="  代码：\r\n    a  b  ",
        question_type=OCSQuestionType.SINGLE,
        options=[" A. 1 ", "B. 2"],
    )
    audit = RequestAudit('{ "title":"原始载荷", "extra":true }')
    order = MagicMock()
    order.attach_mock(harness.requests.record, "record")
    order.attach_mock(harness.cache.set, "set")

    result = asyncio.run(harness.service.answer(question, audit=audit))

    assert not result.cache_hit
    assert not hasattr(result, "code")
    assert question.title.startswith("  ")
    assert len(harness.llm_client.sessions) == 1
    session = harness.llm_client.sessions[0]
    assert session.instructions == SYSTEM_PROMPT
    assert session.response_models == [AnswerPayload]
    expected_question = normalize_question(question)
    assert session.received_inputs[0].parts == (TextPart(build_question_text(expected_question)),)
    assert result.question == "代码：\n    a  b"
    assert harness.cache.get.call_args.args[0] == build_question_hash(expected_question)[0]
    assert harness.cache.get.call_args.kwargs["audit"] is audit
    recorded: LLMCallAudit = harness.requests.record.call_args.args[0]
    assert recorded.status == "SUCCESS"
    assert recorded.audit is audit
    assert harness.cache.set.call_args.kwargs["audit"] is audit
    assert harness.cache.set.call_args.args[0].llm_request_id == 17
    assert [call[0] for call in order.mock_calls] == ["record", "set"]


def test_cache_hit_preserves_original_provider_without_calling_model(
    harness: ServiceHarness,
) -> None:
    """命中时复用答案来源元数据，不创建会话或新增调用流水。"""
    harness.cache.get.return_value = CachedAnswer(
        question_hash="hash",
        llm_request_id=9,
        question="题目",
        question_type=None,
        options=None,
        answer="答案",
        explanation="依据",
        confidence=0.8,
        provider="old-provider",
        model="old-model",
    )
    result = asyncio.run(harness.service.answer(Question(title="题目")))
    assert result.cache_hit
    assert (result.provider, result.model) == ("old-provider", "old-model")
    assert harness.llm_client.sessions == []
    harness.cache.set.assert_not_awaited()
    harness.requests.record.assert_not_awaited()


def test_invalid_answer_retries_in_same_session_and_succeeds() -> None:
    """题型校验失败时在同一会话追加错误反馈重试，每次尝试独立审计，最终缓存成功主键。"""
    invalid_result = LLMCallResult(
        parsed=AnswerPayload(answers=["A", "B"], explanation="猜了两个", confidence=0.6),
        response_body_raw='{"attempt":1}',
        http_status=200,
        usage=LLMUsage(input_tokens=10, output_tokens=4, total_tokens=14),
    )
    valid_result = LLMCallResult(
        parsed=AnswerPayload(answers=["A"], explanation="唯一正解", confidence=0.95),
        response_body_raw='{"attempt":2}',
        http_status=200,
        usage=LLMUsage(input_tokens=20, output_tokens=6, total_tokens=26),
    )
    harness = _make_harness(StubLLMClient(script=[invalid_result, valid_result]), max_attempts=3)
    harness.requests.record.side_effect = [101, 102]

    result = asyncio.run(
        harness.service.answer(
            Question(title="题目", question_type=OCSQuestionType.SINGLE, options=["A. 1", "B. 2"])
        )
    )

    assert result.answer == "A"
    assert not result.cache_hit
    assert len(harness.llm_client.sessions) == 1
    session = harness.llm_client.sessions[0]
    assert len(session.received_inputs) == 2
    correction_part = session.received_inputs[1].parts[0]
    assert isinstance(correction_part, TextPart)
    assert "single requires exactly one answer" in correction_part.text

    audits = [call.args[0] for call in harness.requests.record.call_args_list]
    assert [entry.status for entry in audits] == ["FAILURE", "SUCCESS"]
    assert audits[0].error_type == "InvalidAnswerError"
    assert audits[0].response_body_raw == '{"attempt":1}'
    assert audits[0].usage.total_tokens == 14
    assert audits[1].response_body_raw == '{"attempt":2}'
    assert audits[1].usage.total_tokens == 26
    assert harness.cache.set.call_args.args[0].llm_request_id == 102


def test_output_parse_error_retries_in_same_session_and_audits_usage() -> None:
    """结构解析失败抛出 ProviderOutputError 时，记录该次用量并在同一会话追发纠正提示。"""
    parse_error = ProviderOutputError(
        "ValidationError: confidence must be <= 1",
        raw_body='{"attempt":1,"bad":true}',
        status_code=200,
        usage=LLMUsage(input_tokens=12, output_tokens=8, total_tokens=20),
        raw_output_text='{"answers":["A"],"explanation":"x","confidence":2}',
    )
    valid_result = LLMCallResult(
        parsed=AnswerPayload(answers=["A"], explanation="修正后", confidence=0.9),
        response_body_raw='{"attempt":2}',
        http_status=200,
        usage=LLMUsage(input_tokens=25, output_tokens=7, total_tokens=32),
    )
    harness = _make_harness(StubLLMClient(script=[parse_error, valid_result]), max_attempts=3)
    harness.requests.record.side_effect = [201, 202]

    result = asyncio.run(
        harness.service.answer(Question(title="题目", question_type=OCSQuestionType.SINGLE))
    )

    assert result.answer == "A"
    assert len(harness.llm_client.sessions) == 1
    session = harness.llm_client.sessions[0]
    assert len(session.received_inputs) == 2
    correction_part = session.received_inputs[1].parts[0]
    assert isinstance(correction_part, TextPart)
    assert "confidence must be <= 1" in correction_part.text

    audits = [call.args[0] for call in harness.requests.record.call_args_list]
    assert [entry.status for entry in audits] == ["FAILURE", "SUCCESS"]
    assert audits[0].error_type == "ProviderOutputError"
    assert audits[0].response_body_raw == '{"attempt":1,"bad":true}'
    assert audits[0].http_status == 200
    assert audits[0].usage.total_tokens == 20
    assert harness.cache.set.call_args.args[0].llm_request_id == 202


def test_invalid_answer_exhausts_max_attempts_without_caching(harness: ServiceHarness) -> None:
    """连续不合规达到最大尝试次数后逐次审计，不写入成功流水或缓存。"""
    harness.llm_client.result.parsed.answers = ["A", "B"]
    with pytest.raises(InvalidAnswerError):
        asyncio.run(
            harness.service.answer(Question(title="题目", question_type=OCSQuestionType.SINGLE))
        )
    assert len(harness.llm_client.sessions) == 1
    assert len(harness.llm_client.sessions[0].received_inputs) == 3
    audits = [call.args[0] for call in harness.requests.record.call_args_list]
    assert len(audits) == 3
    assert all(entry.status == "FAILURE" for entry in audits)
    assert all(entry.response_body_raw == '{"upstream":"original"}' for entry in audits)
    harness.cache.set.assert_not_awaited()


@pytest.mark.parametrize(
    "error",
    [
        ProviderTimeoutError("timeout"),
        ProviderResponseError("bad http", raw_body="<html>500</html>", status_code=500),
        ValueError("programming error"),
    ],
)
def test_non_retryable_errors_stop_after_first_attempt(error: Exception) -> None:
    """超时、上游 HTTP 故障和未知编程错误不进入纠正重试循环。"""
    harness = _make_harness(StubLLMClient(error=error), max_attempts=3)
    with pytest.raises(type(error)) as raised:
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert raised.value is error
    assert len(harness.llm_client.received) == 1
    assert harness.requests.record.await_count == 1
    recorded: LLMCallAudit = harness.requests.record.call_args.args[0]
    assert recorded.status == "FAILURE"
    harness.cache.set.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [ProviderTimeoutError("timeout"), ValueError("programming error")]
)
def test_failure_audit_does_not_mask_original_error(
    harness: ServiceHarness,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    """二次审计故障必须留下日志，原异常实例仍然向上传播。"""
    harness.llm_client.error = error
    harness.requests.record.side_effect = RepositoryError("audit unavailable")
    with pytest.raises(type(error)) as raised:
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert raised.value is error
    assert "Failed to persist failure audit" in caplog.text
    harness.cache.set.assert_not_awaited()


def test_provider_error_passes_raw_response_to_audit(harness: ServiceHarness) -> None:
    """模型异常携带的非 JSON 原文不被重新编码。"""
    error = ProviderResponseError("bad response", raw_body="<html>failed</html>", status_code=502)
    harness.llm_client.error = error
    with pytest.raises(ProviderResponseError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    recorded: LLMCallAudit = harness.requests.record.call_args.args[0]
    assert recorded.response_body_raw == error.raw_body
    assert recorded.http_status == 502


def test_cache_read_failure_stops_before_model_call(harness: ServiceHarness) -> None:
    """缓存读取不可用时不继续产生模型调用费用。"""
    harness.cache.get.side_effect = RepositoryError("cache unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert harness.llm_client.received == []


def test_success_audit_failure_prevents_cache_write(harness: ServiceHarness) -> None:
    """没有可追溯的成功流水主键时，不保存答案缓存。"""
    harness.requests.record.side_effect = RepositoryError("audit unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    harness.cache.set.assert_not_awaited()
    assert harness.requests.record.await_count == 1


def test_cache_write_failure_does_not_reclassify_model_success(harness: ServiceHarness) -> None:
    """后续缓存失败不能把已经成功的模型调用重新记录成模型失败。"""
    harness.cache.set.side_effect = RepositoryError("cache unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert harness.requests.record.await_count == 1
    assert harness.requests.record.call_args.args[0].status == "SUCCESS"


def test_cancellation_is_not_converted_into_failure_audit(
    harness: ServiceHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任务取消保持取消语义，不被普通 Exception 捕获。"""
    session = harness.llm_client.create_session(instructions="ignored")
    monkeypatch.setattr(session, "send_structured", AsyncMock(side_effect=asyncio.CancelledError))
    monkeypatch.setattr(harness.llm_client, "create_session", MagicMock(return_value=session))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    harness.cache.set.assert_not_awaited()
    harness.requests.record.assert_not_awaited()


def test_answer_service_rejects_non_positive_max_attempts() -> None:
    """最大尝试次数必须至少为 1。"""
    cache = create_autospec(AnswerCachePort, instance=True)
    requests = create_autospec(LLMRequestPort, instance=True)
    with pytest.raises(ValueError, match="max_attempts"):
        AnswerService(
            cast(AnswerCachePort, cache),
            StubLLMClient(),
            cast(LLMRequestPort, requests),
            max_attempts=0,
        )


def test_internal_question_rejects_blank_title() -> None:
    """内部调用不依赖 HTTP schema 也能拒绝空白题干。"""
    with pytest.raises(ValidationError, match="non-whitespace"):
        Question(title=" \n\t")
