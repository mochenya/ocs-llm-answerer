from __future__ import annotations

import json
from pathlib import Path

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.models import LLMCallAudit
from ocs_llm_answerer.database.connection import connect_database
from ocs_llm_answerer.database.questions import question_values, upsert_question


class LLMRequestRepository:
    """基于 SQLite 的 LLM 调用流水仓储，负责成功和失败调用的审计持久化。"""

    def __init__(self, database_path: Path) -> None:
        """记录独立审计事务使用的数据库路径。

        Args:
            database_path: 已初始化的 SQLite 数据库文件路径。
        """
        self._database_path = database_path

    async def record(self, entry: LLMCallAudit) -> int:
        """原子写入题目快照与单次模型调用流水，返回数据库生成的唯一主键。

        Args:
            entry: 完整的调用配置、状态、耗时、原文、错误与用量审计对象。

        Returns:
            本次插入产生的 llm_requests.id 整数主键。

        Raises:
            RepositoryError: 数据库写入失败或未返回主键，事务不提交。
        """
        metadata = entry.metadata
        usage = entry.usage
        async with connect_database(self._database_path) as db:
            question_raw_json = await upsert_question(
                db,
                *question_values(entry.question, entry.audit),
                seen_at_ns=entry.seen_at_ns,
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
                    entry.request_started_at_ns,
                    entry.request_completed_at_ns,
                    entry.question.question_hash,
                    question_raw_json,
                    metadata.adapter,
                    metadata.base_url,
                    metadata.api_key_env,
                    metadata.model,
                    metadata.timeout_seconds,
                    metadata.max_retries,
                    _dump_optional_json(metadata.extra_body),
                    entry.status,
                    entry.response_body_raw,
                    entry.error_type,
                    entry.error_message,
                    entry.http_status,
                    entry.latency_ms,
                    usage.input_tokens,
                    usage.output_tokens,
                    usage.total_tokens,
                    usage.cached_tokens,
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
