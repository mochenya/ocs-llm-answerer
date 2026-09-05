from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import ValidationError

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.models import CachedAnswer, RequestAudit
from ocs_llm_answerer.database.connection import connect_database
from ocs_llm_answerer.database.questions import question_values, upsert_question


class AnswerCacheRepository:
    """基于 SQLite 的答案缓存仓储，负责缓存读写和命中统计。"""

    def __init__(self, database_path: Path) -> None:
        """记录连接目标，不在构造时执行数据库操作。

        Args:
            database_path: 已初始化的 SQLite 数据库文件路径。
        """
        self._database_path = database_path

    async def get(
        self,
        question_hash: str,
        question_raw_json: str | None = None,
        seen_at_ns: int | None = None,
    ) -> CachedAnswer | None:
        """读取当前答案及其来源调用，并更新命中统计。

        Args:
            question_hash: 标准化题目的稳定标识。
            question_raw_json: 本次请求原文，提供时更新题目的最近载荷。
            seen_at_ns: 本次请求时间；缺省时使用当前 Unix 纳秒时间。

        Returns:
            带有来源调用 ID 的缓存记录；未命中时返回 None。

        Raises:
            RepositoryError: 查询或统计写入失败。
        """
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with connect_database(self._database_path) as db:
            cursor = await db.execute(
                """
                SELECT q.question_hash, c.llm_request_id, q.question, q.question_type,
                       q.options_json, c.answer, c.explanation, c.confidence, c.provider, c.model
                FROM answer_cache AS c
                JOIN questions AS q ON q.question_hash = c.question_hash
                WHERE c.question_hash = ?
                """,
                (question_hash,),
            )
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            values = dict(row)
            options_json = values.pop("options_json")
            try:
                values["options"] = json.loads(options_json) if options_json is not None else None
                record = CachedAnswer.model_validate(values)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise RepositoryError("Stored answer is invalid") from exc
            await db.execute(
                """
                UPDATE answer_cache
                SET hit_count = hit_count + 1,
                    last_hit_at_ns = ?
                WHERE question_hash = ?
                """,
                (seen_at_ns, question_hash),
            )
            if question_raw_json is not None:
                await db.execute(
                    """
                    UPDATE questions
                    SET question_raw_json = ?,
                        last_seen_at_ns = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE question_hash = ?
                    """,
                    (question_raw_json, seen_at_ns, question_hash),
                )
            else:
                await db.execute(
                    """
                    UPDATE questions
                    SET last_seen_at_ns = ?
                    WHERE question_hash = ?
                    """,
                    (seen_at_ns, question_hash),
                )
            await db.commit()
        return record

    async def set(
        self,
        record: CachedAnswer,
        seen_at_ns: int | None = None,
        *,
        audit: RequestAudit | None = None,
    ) -> None:
        """在同一短事务中写入题目与当前答案，并验证调用外键。

        调用方应先保存成功流水，再传入其整数 ID。写入失败时，题目更新和
        缓存更新一并回滚，避免留下部分修改。

        Args:
            record: 已通过题型校验、引用已存在调用的答案。
            seen_at_ns: 本次请求时间；缺省时使用当前 Unix 纳秒时间。
            audit: 与答案内容分离的原始请求快照。

        Raises:
            RepositoryError: 调用不存在、约束被违反或数据库写入失败。
        """
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with connect_database(self._database_path) as db:
            await upsert_question(db, *question_values(record, audit), seen_at_ns=seen_at_ns)
            await db.execute(
                """
                INSERT INTO answer_cache (
                    question_hash, llm_request_id, answer, explanation, confidence, provider, model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_hash) DO UPDATE SET
                    llm_request_id = excluded.llm_request_id,
                    answer = excluded.answer,
                    explanation = excluded.explanation,
                    confidence = excluded.confidence,
                    provider = excluded.provider,
                    model = excluded.model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record.question_hash,
                    record.llm_request_id,
                    record.answer,
                    record.explanation,
                    record.confidence,
                    record.provider,
                    record.model,
                ),
            )
            await db.commit()
