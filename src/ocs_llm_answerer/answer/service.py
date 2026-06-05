from __future__ import annotations

import hashlib
import json
import re
import time

from ocs_llm_answerer.core.models import (
    AnswerRequest,
    AnswerResponse,
    CachedAnswer,
    NormalizedQuestion,
)
from ocs_llm_answerer.database.cache import AnswerCache
from ocs_llm_answerer.database.llm_requests import LLMRequestLog
from ocs_llm_answerer.llm.provider import LLMProvider

_WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def normalize_options(options: list[str] | str | None) -> list[str] | None:
    """把 OCS 选项载荷标准化为可比较的列表。"""
    if options is None:
        return None
    if isinstance(options, str):
        normalized = [normalize_text(line) for line in options.splitlines() if line.strip()]
        return normalized or None
    normalized = [normalize_text(option) for option in options if option.strip()]
    return normalized or None


def build_question_hash(request: AnswerRequest) -> tuple[str, list[str] | None]:
    """基于题目语义字段而不是原始 JSON 形状生成哈希。"""
    normalized_options = normalize_options(request.options)
    payload = {
        "title": normalize_text(request.title),
        "type": normalize_text(request.question_type or ""),
        "options": normalized_options,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), normalized_options


def build_normalized_question(request: AnswerRequest) -> NormalizedQuestion:
    question_hash, normalized_options = build_question_hash(request)
    options_json = (
        json.dumps(normalized_options, ensure_ascii=False)
        if normalized_options is not None
        else None
    )
    return NormalizedQuestion(
        question_hash=question_hash,
        question=normalize_text(request.title),
        question_type=normalize_text(request.question_type or "") or None,
        options_json=options_json,
        question_raw_json=request.raw_payload_json,
    )


class AnswerService:
    """协调缓存查询、provider 回退和缓存持久化。"""

    def __init__(
        self, cache: AnswerCache, provider: LLMProvider, request_log: LLMRequestLog
    ) -> None:
        self._cache = cache
        self._provider = provider
        self._request_log = request_log

    async def answer(self, request: AnswerRequest) -> AnswerResponse:
        seen_at_ns = time.time_ns()
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
        try:
            llm_result = await self._provider.answer(request)
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
            )
            raise

        request_completed_at_ns = time.time_ns()
        call_hash = await self._request_log.record_success(
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
            call_hash=call_hash,
            question=normalized_question.question,
            question_type=normalized_question.question_type,
            options_json=normalized_question.options_json,
            question_raw_json=normalized_question.question_raw_json,
            answer=llm_answer.answer,
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
