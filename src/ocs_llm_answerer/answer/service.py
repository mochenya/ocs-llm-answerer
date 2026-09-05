from __future__ import annotations

import logging
import time

from ocs_llm_answerer.answer.formatting import validate_and_format_answer
from ocs_llm_answerer.answer.models import (
    AnswerResult,
    CachedAnswer,
    Question,
    RequestAudit,
)
from ocs_llm_answerer.answer.normalization import build_normalized_question, normalize_question
from ocs_llm_answerer.answer.ports import AnswerCachePort, LLMRequestPort
from ocs_llm_answerer.llm.errors import ProviderError
from ocs_llm_answerer.llm.models import LLMCallResult
from ocs_llm_answerer.llm.provider import LLMProvider

_LOGGER = logging.getLogger(__name__)


class AnswerService:
    """协调缓存查询、模型调用、答案校验和仓储持久化。"""

    def __init__(
        self,
        cache_repository: AnswerCachePort,
        provider: LLMProvider,
        request_repository: LLMRequestPort,
    ) -> None:
        """注入用例需要的持久化组件和模型适配器。

        Args:
            cache_repository: 保存已校验答案的缓存仓储。
            provider: 返回结构化答案的模型适配器。
            request_repository: 独立提交模型调用流水的审计仓储。
        """
        self._cache_repository = cache_repository
        self._provider = provider
        self._request_repository = request_repository

    async def answer(self, request: Question, *, audit: RequestAudit | None = None) -> AnswerResult:
        """使用相同的标准化输入查询缓存或调用模型。

        Args:
            request: 不含 HTTP 或数据库编码细节的内部题目。
            audit: 与题目分离的原始请求快照。

        Returns:
            带有答案来源和缓存命中标志的业务结果。

        Raises:
            InvalidAnswerError: 模型答案不符合题型规则，不写入成功缓存。
            ProviderError: Provider 调用失败；失败审计异常不覆盖此错误。
            RepositoryError: 缓存读写或成功流水写入失败。
        """
        seen_at_ns = time.time_ns()
        request = normalize_question(request)
        normalized_question = build_normalized_question(request)
        cached = await self._cache_repository.get(
            normalized_question.question_hash,
            question_raw_json=audit.raw_payload_json if audit is not None else None,
            seen_at_ns=seen_at_ns,
        )
        if cached is not None:
            return _result_from_record(cached, cache_hit=True)

        request_started_at_ns = time.time_ns()
        started_at = time.perf_counter()
        llm_result: LLMCallResult | None = None
        try:
            llm_result = await self._provider.answer(request)
            answer = validate_and_format_answer(request, llm_result.answer.answers)
        except Exception as exc:
            request_completed_at_ns = time.time_ns()
            raw = exc.raw_body if isinstance(exc, ProviderError) else None
            try:
                await self._request_repository.record_failure(
                    normalized_question,
                    self._provider.request_metadata,
                    exc,
                    seen_at_ns,
                    request_started_at_ns,
                    request_completed_at_ns,
                    _elapsed_ms(started_at),
                    response_body_raw=raw,
                    result=llm_result,
                    audit=audit,
                )
            except Exception:
                _LOGGER.exception(
                    "Failed to persist failure audit for question %s (original error: %s)",
                    normalized_question.question_hash,
                    type(exc).__name__,
                )
            raise

        request_completed_at_ns = time.time_ns()
        llm_request_id = await self._request_repository.record_success(
            normalized_question,
            self._provider.request_metadata,
            llm_result,
            seen_at_ns,
            request_started_at_ns,
            request_completed_at_ns,
            _elapsed_ms(started_at),
            audit=audit,
        )
        llm_answer = llm_result.answer
        record = CachedAnswer(
            question_hash=normalized_question.question_hash,
            llm_request_id=llm_request_id,
            question=normalized_question.question,
            question_type=normalized_question.question_type,
            options=normalized_question.options,
            answer=answer,
            explanation=llm_answer.explanation,
            confidence=llm_answer.confidence,
            provider=self._provider.name,
            model=self._provider.model,
        )
        await self._cache_repository.set(record, seen_at_ns=seen_at_ns, audit=audit)

        return _result_from_record(record, cache_hit=False)


def _result_from_record(record: CachedAnswer, *, cache_hit: bool) -> AnswerResult:
    """构造业务结果，保留实际生成答案的 Provider 和模型信息。

    Args:
        record: 来自缓存或本次模型调用的答案记录。
        cache_hit: 是否复用了已有缓存。

    Returns:
        不包含 HTTP 成功标识的答题结果。
    """
    return AnswerResult(
        question=record.question,
        answer=record.answer,
        explanation=record.explanation,
        confidence=record.confidence,
        provider=record.provider,
        model=record.model,
        cache_hit=cache_hit,
    )


def _elapsed_ms(started_at: float) -> int:
    """计算非负的调用耗时。

    Args:
        started_at: perf_counter 返回的单调时钟起点。

    Returns:
        四舍五入后的毫秒数。
    """
    return max(0, round((time.perf_counter() - started_at) * 1000))
