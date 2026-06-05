from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import aiosqlite

from ocs_llm_answerer.core.models import LLMCallResult, LLMProviderMetadata, NormalizedQuestion
from ocs_llm_answerer.database.questions import question_values_from_normalized, upsert_question


class LLMRequestLog:
    """基于 SQLite 的 LLM provider 调用审计日志。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def record_success(
        self,
        question: NormalizedQuestion,
        metadata: LLMProviderMetadata,
        result: LLMCallResult,
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        latency_ms: int,
    ) -> str:
        return await self._insert(
            question=question,
            metadata=metadata,
            request_status="SUCCESS",
            seen_at_ns=seen_at_ns,
            request_started_at_ns=request_started_at_ns,
            request_completed_at_ns=request_completed_at_ns,
            response_body_raw=result.response_body_raw,
            error_type=None,
            error_message=None,
            http_status=result.http_status,
            latency_ms=latency_ms,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
            cached_tokens=result.usage.cached_tokens,
        )

    async def record_failure(
        self,
        question: NormalizedQuestion,
        metadata: LLMProviderMetadata,
        error: Exception,
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        latency_ms: int,
        response_body_raw: str | None = None,
    ) -> str:
        return await self._insert(
            question=question,
            metadata=metadata,
            request_status="FAILURE",
            seen_at_ns=seen_at_ns,
            request_started_at_ns=request_started_at_ns,
            request_completed_at_ns=request_completed_at_ns,
            response_body_raw=response_body_raw,
            error_type=type(error).__name__,
            error_message=str(error),
            http_status=_extract_http_status(error),
            latency_ms=latency_ms,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cached_tokens=None,
        )

    async def _insert(
        self,
        question: NormalizedQuestion,
        metadata: LLMProviderMetadata,
        request_status: Literal["SUCCESS", "FAILURE"],
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        response_body_raw: str | None,
        error_type: str | None,
        error_message: str | None,
        http_status: int | None,
        latency_ms: int,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        cached_tokens: int | None,
    ) -> str:
        call_hash = _build_call_hash(question, metadata, request_started_at_ns)
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            question_raw_json = await upsert_question(
                db, *question_values_from_normalized(question), seen_at_ns=seen_at_ns
            )
            await db.execute(
                """
                INSERT INTO llm_requests (
                    call_hash, request_started_at_ns, request_completed_at_ns,
                    question_hash, question_raw_json,
                    adapter, base_url, api_key_env, model,
                    timeout_seconds, max_retries, extra_body,
                    request_status, response_body_raw,
                    error_type, error_message, http_status, latency_ms,
                    input_tokens, output_tokens, total_tokens, cached_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    call_hash,
                    request_started_at_ns,
                    request_completed_at_ns,
                    question.question_hash,
                    question_raw_json,
                    metadata.adapter,
                    metadata.base_url,
                    metadata.api_key_env,
                    metadata.model,
                    metadata.timeout_seconds,
                    metadata.max_retries,
                    _dump_optional_json(metadata.extra_body),
                    request_status,
                    response_body_raw,
                    error_type,
                    error_message,
                    http_status,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cached_tokens,
                ),
            )
            await db.commit()
        return call_hash


def _dump_optional_json(value: object | None) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _extract_http_status(error: Exception) -> int | None:
    status_code = getattr(error, "status_code", None)
    return status_code if isinstance(status_code, int) else None


def _build_call_hash(
    question: NormalizedQuestion,
    metadata: LLMProviderMetadata,
    request_started_at_ns: int,
) -> str:
    payload = {
        "question_hash": question.question_hash,
        "request_started_at_ns": request_started_at_ns,
        "adapter": metadata.adapter,
        "base_url": metadata.base_url,
        "api_key_env": metadata.api_key_env,
        "model": metadata.model,
        "timeout_seconds": metadata.timeout_seconds,
        "max_retries": metadata.max_retries,
        "extra_body": metadata.extra_body,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
