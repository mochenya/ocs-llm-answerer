from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.models import NormalizedQuestion, RequestAudit
from ocs_llm_answerer.database.connection import connect_database
from ocs_llm_answerer.database.questions import question_values, upsert_question
from ocs_llm_answerer.llm.errors import ProviderError
from ocs_llm_answerer.llm.models import LLMCallResult, LLMProviderMetadata, LLMUsage


class LLMRequestRepository:
    """基于 SQLite 的 LLM 调用流水仓储，负责成功和失败调用的审计持久化。"""

    def __init__(self, database_path: Path) -> None:
        """记录独立审计事务使用的数据库路径。

        Args:
            database_path: 已初始化的 SQLite 数据库文件路径。
        """
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
        *,
        audit: RequestAudit | None = None,
    ) -> int:
        """保存成功调用并返回可供缓存引用的数据库主键。

        Args:
            question: 具有稳定身份的标准化题目。
            metadata: 本次 Provider 配置快照。
            result: 已通过答题层校验的模型结果。
            seen_at_ns: 请求进入服务的 Unix 纳秒时间。
            request_started_at_ns: 模型调用开始的 Unix 纳秒时间。
            request_completed_at_ns: 模型调用结束的 Unix 纳秒时间。
            latency_ms: 单调时钟测得的调用耗时。
            audit: 独立于标准化题目的原始请求快照。

        Returns:
            已提交的 llm_requests.id 整数主键。

        Raises:
            RepositoryError: 题目或流水写入失败。
        """
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
            audit=audit,
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
        result: LLMCallResult | None = None,
        *,
        audit: RequestAudit | None = None,
    ) -> int:
        """记录调用或答案校验失败，并保留已经收到的模型响应。

        Args:
            question: 具有稳定身份的标准化题目。
            metadata: 本次 Provider 配置快照。
            error: 导致答题失败的原始异常。
            seen_at_ns: 请求进入服务的时间，单位为 Unix 纳秒。
            request_started_at_ns: 模型调用开始时间，单位为 Unix 纳秒。
            request_completed_at_ns: 模型调用结束时间，单位为 Unix 纳秒。
            latency_ms: 单调时钟测得的调用耗时。
            response_body_raw: 解析失败等异常携带的原始响应文本。
            result: 已取得但未通过题型校验的结果，用于保留响应和用量。
            audit: 独立于标准化题目的原始请求快照。

        Returns:
            已提交的 llm_requests.id 整数主键。

        Raises:
            RepositoryError: 题目或流水写入失败。
        """
        usage = result.usage if result is not None else LLMUsage()
        return await self._insert(
            question=question,
            metadata=metadata,
            request_status="FAILURE",
            seen_at_ns=seen_at_ns,
            request_started_at_ns=request_started_at_ns,
            request_completed_at_ns=request_completed_at_ns,
            response_body_raw=result.response_body_raw if result is not None else response_body_raw,
            error_type=type(error).__name__,
            error_message=str(error),
            http_status=(
                result.http_status
                if result is not None
                else error.status_code
                if isinstance(error, ProviderError)
                else None
            ),
            latency_ms=latency_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            cached_tokens=usage.cached_tokens,
            audit=audit,
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
        audit: RequestAudit | None,
    ) -> int:
        """原子写入题目和调用流水，使用数据库生成的唯一 ID。

        Args:
            question: 具有稳定身份的标准化题目。
            metadata: Provider 配置快照。
            request_status: 调用或答案校验的最终状态。
            seen_at_ns: 请求进入服务的 Unix 纳秒时间。
            request_started_at_ns: 模型调用开始的 Unix 纳秒时间。
            request_completed_at_ns: 模型调用结束的 Unix 纳秒时间。
            response_body_raw: 可为空或非 JSON 的原始响应文本。
            error_type: 原始错误类型，成功时为 None。
            error_message: 原始错误说明，成功时为 None。
            http_status: 上游 HTTP 状态码，尚未收到响应时可为 None。
            latency_ms: 单调时钟测得的耗时。
            input_tokens: 输入 token 数，未取得时为 None。
            output_tokens: 输出 token 数，未取得时为 None。
            total_tokens: 总 token 数，未取得时为 None。
            cached_tokens: 上游缓存命中的 token 数，未取得时为 None。
            audit: 独立于标准化题目的原始请求快照。

        Returns:
            本次插入产生的 llm_requests.id，不依赖时间或配置哈希。

        Raises:
            RepositoryError: 数据库写入失败或未返回主键，事务不提交。
        """
        async with connect_database(self._database_path) as db:
            question_raw_json = await upsert_question(
                db, *question_values(question, audit), seen_at_ns=seen_at_ns
            )
            cursor = await db.execute(
                """
                INSERT INTO llm_requests (
                    request_started_at_ns, request_completed_at_ns,
                    question_hash, question_raw_json,
                    adapter, base_url, api_key_env, model,
                    timeout_seconds, max_retries, extra_body,
                    request_status, response_body_raw,
                    error_type, error_message, http_status, latency_ms,
                    input_tokens, output_tokens, total_tokens, cached_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
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
            llm_request_id = cursor.lastrowid
            await cursor.close()
            if llm_request_id is None:
                raise RepositoryError("SQLite did not return an LLM request ID")
            await db.commit()
        return llm_request_id


def _dump_optional_json(value: object | None) -> str | None:
    """稳定编码可选配置，保留既有数据库存储格式。

    Args:
        value: 可由 JSON 编码的配置值或 None。

    Returns:
        排序且无额外空格的 JSON，None 原样返回。
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
