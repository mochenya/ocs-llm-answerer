"""只使用内存替身验证答题编排，不依赖 HTTP、SQLite 或真实模型。"""

import asyncio
from dataclasses import dataclass
from typing import cast
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest
from pydantic import ValidationError

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.formatting import InvalidAnswerError
from ocs_llm_answerer.answer.models import CachedAnswer, OCSQuestionType, Question, RequestAudit
from ocs_llm_answerer.answer.normalization import build_question_hash
from ocs_llm_answerer.answer.ports import AnswerCachePort, LLMRequestPort
from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.llm.errors import ProviderResponseError, ProviderTimeoutError
from tests.fakes import StubProvider


@dataclass
class ServiceHarness:
    """具有真实协议签名的内存仓储及可观察 Provider。"""

    cache: MagicMock
    requests: MagicMock
    provider: StubProvider
    service: AnswerService


@pytest.fixture
def harness() -> ServiceHarness:
    """组装不执行 I/O 的服务测试环境。

    Returns:
        缓存默认未命中、成功审计返回主键 17 的测试环境。
    """
    cache = create_autospec(AnswerCachePort, instance=True)
    requests = create_autospec(LLMRequestPort, instance=True)
    cache.get.return_value = None
    requests.record_success.return_value = 17
    provider = StubProvider()
    return ServiceHarness(
        cache,
        requests,
        provider,
        AnswerService(cast(AnswerCachePort, cache), provider, cast(LLMRequestPort, requests)),
    )


def test_success_normalizes_once_for_provider_and_preserves_audit(harness: ServiceHarness) -> None:
    """模型输入与哈希一致，原始载荷只送入仓储，先记录成功调用再写缓存。"""
    question = Question(
        title="  代码：\r\n    a  b  ",
        question_type=OCSQuestionType.SINGLE,
        options=[" A. 1 ", "B. 2"],
    )
    audit = RequestAudit('{ "title":"原始载荷", "extra":true }')
    order = MagicMock()
    order.attach_mock(harness.requests.record_success, "record_success")
    order.attach_mock(harness.cache.set, "set")

    result = asyncio.run(harness.service.answer(question, audit=audit))

    assert not result.cache_hit
    assert not hasattr(result, "code")
    assert question.title.startswith("  ")
    received = harness.provider.received[0]
    assert received.title == result.question == "代码：\n    a  b"
    assert received.options == ["A. 1", "B. 2"]
    assert not hasattr(received, "raw_payload_json")
    assert harness.cache.get.call_args.args[0] == build_question_hash(received)[0]
    assert harness.requests.record_success.call_args.kwargs["audit"] is audit
    assert harness.cache.set.call_args.kwargs["audit"] is audit
    assert harness.cache.set.call_args.args[0].llm_request_id == 17
    assert [call[0] for call in order.mock_calls] == ["record_success", "set"]
    harness.requests.record_failure.assert_not_awaited()


def test_cache_hit_preserves_original_provider_without_calling_model(
    harness: ServiceHarness,
) -> None:
    """命中时复用答案来源元数据，不调用 Provider 或新增调用流水。"""
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
    assert harness.provider.received == []
    harness.cache.set.assert_not_awaited()
    harness.requests.record_success.assert_not_awaited()
    harness.requests.record_failure.assert_not_awaited()


def test_invalid_answer_keeps_result_for_audit_without_caching(harness: ServiceHarness) -> None:
    """非法答案的原文和用量可供审计，但绝不写入成功流水或缓存。"""
    harness.provider.result.answer.answers = ["A", "B"]
    with pytest.raises(InvalidAnswerError):
        asyncio.run(
            harness.service.answer(Question(title="题目", question_type=OCSQuestionType.SINGLE))
        )
    assert harness.requests.record_failure.call_args.kwargs["result"] is harness.provider.result
    harness.cache.set.assert_not_awaited()
    harness.requests.record_success.assert_not_awaited()


@pytest.mark.parametrize(
    "error", [ProviderTimeoutError("timeout"), ValueError("programming error")]
)
def test_failure_audit_does_not_mask_original_error(
    harness: ServiceHarness,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
) -> None:
    """二次审计故障必须留下日志，原异常实例仍然向上传播。"""
    harness.provider.error = error
    harness.requests.record_failure.side_effect = RepositoryError("audit unavailable")
    with pytest.raises(type(error)) as raised:
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert raised.value is error
    assert "Failed to persist failure audit" in caplog.text
    harness.cache.set.assert_not_awaited()


def test_provider_error_passes_raw_response_to_audit(harness: ServiceHarness) -> None:
    """模型异常携带的非 JSON 原文不被重新编码。"""
    error = ProviderResponseError("bad response", raw_body="<html>failed</html>", status_code=502)
    harness.provider.error = error
    with pytest.raises(ProviderResponseError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert harness.requests.record_failure.call_args.kwargs["response_body_raw"] == error.raw_body


def test_cache_read_failure_stops_before_model_call(harness: ServiceHarness) -> None:
    """缓存读取不可用时不继续产生模型调用费用。"""
    harness.cache.get.side_effect = RepositoryError("cache unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    assert harness.provider.received == []


def test_success_audit_failure_prevents_cache_write(harness: ServiceHarness) -> None:
    """没有可追溯的成功流水主键时，不保存答案缓存。"""
    harness.requests.record_success.side_effect = RepositoryError("audit unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    harness.cache.set.assert_not_awaited()
    harness.requests.record_failure.assert_not_awaited()


def test_cache_write_failure_does_not_reclassify_model_success(harness: ServiceHarness) -> None:
    """后续缓存失败不能把已经成功的模型调用重新记录成模型失败。"""
    harness.cache.set.side_effect = RepositoryError("cache unavailable")
    with pytest.raises(RepositoryError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    harness.requests.record_success.assert_awaited_once()
    harness.requests.record_failure.assert_not_awaited()


def test_cancellation_is_not_converted_into_failure_audit(
    harness: ServiceHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """任务取消保持取消语义，不被普通 Exception 捕获。"""
    monkeypatch.setattr(harness.provider, "answer", AsyncMock(side_effect=asyncio.CancelledError))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(harness.service.answer(Question(title="题目")))
    harness.cache.set.assert_not_awaited()
    harness.requests.record_success.assert_not_awaited()
    harness.requests.record_failure.assert_not_awaited()


def test_internal_question_rejects_blank_title() -> None:
    """内部调用不依赖 HTTP schema 也能拒绝空白题干。"""
    with pytest.raises(ValidationError, match="non-whitespace"):
        Question(title=" \n\t")
