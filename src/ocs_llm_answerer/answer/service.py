from __future__ import annotations

import time

from ocs_llm_answerer.answer.formatting import validate_and_format_answer
from ocs_llm_answerer.answer.normalization import build_normalized_question, normalize_request
from ocs_llm_answerer.core.models import (
    AnswerRequest,
    AnswerResponse,
    CachedAnswer,
    LLMCallResult,
)
from ocs_llm_answerer.database.cache import AnswerCache
from ocs_llm_answerer.database.llm_requests import LLMRequestLog
from ocs_llm_answerer.llm.provider import LLMProvider


class AnswerService:
    """协调缓存查询、provider 回退和缓存持久化。"""

    def __init__(
        self, cache: AnswerCache, provider: LLMProvider, request_log: LLMRequestLog
    ) -> None:
        self._cache = cache
        self._provider = provider
        self._request_log = request_log

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        """使用相同的标准化输入查询缓存或调用模型。

        Args:
            request: 已校验的题目及可选原始请求快照。

        Returns:
            带有答案来源和缓存命中标志的 OCS 响应。

        Raises:
            InvalidAnswerError: 模型答案不符合题型规则，不写入成功缓存。
            Exception: Provider 调用或持久化失败时向调用方传播错误。
        """
        seen_at_ns = time.time_ns()
        request = normalize_request(request)
        normalized_question = build_normalized_question(request)
        cached = await self._cache.get(
            normalized_question.question_hash,
            question_raw_json=normalized_question.question_raw_json,
            seen_at_ns=seen_at_ns,
        )
        if cached is not None:
            return _response_from_cached(cached)

        request_started_at_ns = time.time_ns()
        started_at = time.perf_counter()
        llm_result: LLMCallResult | None = None
        try:
            llm_result = await self._provider.answer(request)
            answer = validate_and_format_answer(request, llm_result.answer.answers)
        except Exception as exc:
            request_completed_at_ns = time.time_ns()
            raw = getattr(exc, "raw_body", None)
            await self._request_log.record_failure(
                normalized_question,
                self._provider.request_metadata,
                exc,
                seen_at_ns,
                request_started_at_ns,
                request_completed_at_ns,
                _elapsed_ms(started_at),
                response_body_raw=raw,
                result=llm_result,
            )
            raise

        request_completed_at_ns = time.time_ns()
        llm_request_id = await self._request_log.record_success(
            normalized_question,
            self._provider.request_metadata,
            llm_result,
            seen_at_ns,
            request_started_at_ns,
            request_completed_at_ns,
            _elapsed_ms(started_at),
        )
        llm_answer = llm_result.answer
        record = CachedAnswer(
            question_hash=normalized_question.question_hash,
            llm_request_id=llm_request_id,
            question=normalized_question.question,
            question_type=normalized_question.question_type,
            options_json=normalized_question.options_json,
            question_raw_json=normalized_question.question_raw_json,
            answer=answer,
            explanation=llm_answer.explanation,
            confidence=llm_answer.confidence,
            provider=self._provider.name,
            model=self._provider.model,
        )
        await self._cache.set(record, seen_at_ns=seen_at_ns)

        return _response_from_llm(record)


def _response_from_cached(cached: CachedAnswer) -> AnswerResponse:
    # 保留缓存记录里的 provider/model 元数据，方便追溯。
    return AnswerResponse(
        question=cached.question,
        answer=cached.answer,
        explanation=cached.explanation,
        confidence=cached.confidence,
        provider=cached.provider,
        model=cached.model,
        cache_hit=True,
    )


def _response_from_llm(record: CachedAnswer) -> AnswerResponse:
    return AnswerResponse(
        question=record.question,
        answer=record.answer,
        explanation=record.explanation,
        confidence=record.confidence,
        provider=record.provider,
        model=record.model,
        cache_hit=False,
    )


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((time.perf_counter() - started_at) * 1000))
