from __future__ import annotations

import logging
import time

from ocs_llm_answerer.answer.formatting import InvalidAnswerError, validate_and_format_answer
from ocs_llm_answerer.answer.models import (
    AnswerPayload,
    AnswerResult,
    CachedAnswer,
    LLMCallAudit,
    NormalizedQuestion,
    Question,
    RequestAudit,
)
from ocs_llm_answerer.answer.normalization import build_normalized_question, normalize_question
from ocs_llm_answerer.answer.ports import AnswerCachePort, LLMRequestPort
from ocs_llm_answerer.answer.prompting import (
    SYSTEM_PROMPT,
    build_correction_input,
    build_question_input,
)
from ocs_llm_answerer.llm.client import LLMClient
from ocs_llm_answerer.llm.errors import ProviderError, ProviderOutputError
from ocs_llm_answerer.llm.models import LLMCallResult, LLMUsage

_LOGGER = logging.getLogger(__name__)


class AnswerService:
    """协调缓存查询、单请求模型会话、纠正重试、答案校验和仓储持久化。"""

    def __init__(
        self,
        cache_repository: AnswerCachePort,
        llm_client: LLMClient,
        request_repository: LLMRequestPort,
        *,
        max_attempts: int = 3,
    ) -> None:
        """注入用例需要的持久化组件和通用 LLM 客户端。

        Args:
            cache_repository: 保存已校验答案的缓存仓储。
            llm_client: 创建单请求多轮会话的通用模型客户端。
            request_repository: 独立提交每次模型调用流水的审计仓储。
            max_attempts: 单次答题允许的最大模型尝试次数（含初次及解析/校验失败后的纠正重试）。

        Raises:
            ValueError: max_attempts 小于 1。
        """
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._cache_repository = cache_repository
        self._llm_client = llm_client
        self._request_repository = request_repository
        self._max_attempts = max_attempts

    async def answer(self, request: Question, *, audit: RequestAudit | None = None) -> AnswerResult:
        """使用相同的标准化输入查询缓存或在独立会话中完成答题与纠正重试。

        Args:
            request: 不含 HTTP 或数据库编码细节的内部题目。
            audit: 与题目分离的原始请求快照。

        Returns:
            带有答案来源和缓存命中标志的业务结果。

        Raises:
            InvalidAnswerError: 达到最大尝试次数后模型答案仍违反题型规则，不写入成功缓存。
            ProviderError: 模型调用失败或达到最大尝试次数后仍无法解析；失败审计异常不覆盖此错误。
            RepositoryError: 缓存读写或成功流水写入失败。
        """
        seen_at_ns = time.time_ns()
        request = normalize_question(request)
        normalized_question = build_normalized_question(request)
        cached = await self._cache_repository.get(
            normalized_question.question_hash,
            audit=audit,
            seen_at_ns=seen_at_ns,
        )
        if cached is not None:
            return _result_from_record(cached, cache_hit=True)

        session = self._llm_client.create_session(instructions=SYSTEM_PROMPT)
        user_input = build_question_input(request)

        for attempt in range(1, self._max_attempts + 1):
            request_started_at_ns = time.time_ns()
            started_at = time.perf_counter()
            llm_result: LLMCallResult[AnswerPayload] | None = None
            try:
                llm_result = await session.send_structured(user_input, AnswerPayload)
                answer = validate_and_format_answer(request, llm_result.parsed.answers)
            except Exception as exc:
                request_completed_at_ns = time.time_ns()
                await self._persist_failure_audit(
                    normalized_question=normalized_question,
                    error=exc,
                    seen_at_ns=seen_at_ns,
                    request_started_at_ns=request_started_at_ns,
                    request_completed_at_ns=request_completed_at_ns,
                    latency_ms=_elapsed_ms(started_at),
                    llm_result=llm_result,
                    audit=audit,
                )
                if isinstance(exc, (ProviderOutputError, InvalidAnswerError)) and (
                    attempt < self._max_attempts
                ):
                    user_input = build_correction_input(exc)
                    continue
                raise

            request_completed_at_ns = time.time_ns()
            llm_request_id = await self._request_repository.record(
                LLMCallAudit(
                    question=normalized_question,
                    metadata=self._llm_client.request_metadata,
                    status="SUCCESS",
                    seen_at_ns=seen_at_ns,
                    request_started_at_ns=request_started_at_ns,
                    request_completed_at_ns=request_completed_at_ns,
                    latency_ms=_elapsed_ms(started_at),
                    response_body_raw=llm_result.response_body_raw,
                    http_status=llm_result.http_status,
                    usage=llm_result.usage,
                    audit=audit,
                )
            )
            payload = llm_result.parsed
            record = CachedAnswer(
                question_hash=normalized_question.question_hash,
                llm_request_id=llm_request_id,
                question=normalized_question.question,
                question_type=normalized_question.question_type,
                options=normalized_question.options,
                answer=answer,
                explanation=payload.explanation,
                confidence=payload.confidence,
                provider=self._llm_client.name,
                model=self._llm_client.model,
            )
            await self._cache_repository.set(record, seen_at_ns=seen_at_ns, audit=audit)
            return _result_from_record(record, cache_hit=False)

        raise AssertionError("Unreachable: max_attempts is guaranteed to be >= 1")

    async def _persist_failure_audit(
        self,
        *,
        normalized_question: NormalizedQuestion,
        error: Exception,
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        latency_ms: int,
        llm_result: LLMCallResult[AnswerPayload] | None,
        audit: RequestAudit | None,
    ) -> None:
        """记录失败审计并吞掉二次存储异常，避免掩盖原始答题故障。"""
        if llm_result is not None:
            response_body_raw = llm_result.response_body_raw
            http_status = llm_result.http_status
            usage = llm_result.usage
        elif isinstance(error, ProviderOutputError):
            response_body_raw = error.raw_body
            http_status = error.status_code
            usage = error.usage
        elif isinstance(error, ProviderError):
            response_body_raw = error.raw_body
            http_status = error.status_code
            usage = LLMUsage()
        else:
            response_body_raw = None
            http_status = None
            usage = LLMUsage()

        try:
            await self._request_repository.record(
                LLMCallAudit(
                    question=normalized_question,
                    metadata=self._llm_client.request_metadata,
                    status="FAILURE",
                    seen_at_ns=seen_at_ns,
                    request_started_at_ns=request_started_at_ns,
                    request_completed_at_ns=request_completed_at_ns,
                    latency_ms=latency_ms,
                    response_body_raw=response_body_raw,
                    http_status=http_status,
                    usage=usage,
                    error_type=type(error).__name__,
                    error_message=str(error),
                    audit=audit,
                )
            )
        except Exception:
            _LOGGER.exception(
                "Failed to persist failure audit for question %s (original error: %s)",
                normalized_question.question_hash,
                type(error).__name__,
            )


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
