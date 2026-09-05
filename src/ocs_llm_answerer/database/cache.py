from __future__ import annotations

import time
from importlib.resources import files
from pathlib import Path

import aiosqlite

from ocs_llm_answerer.core.models import CachedAnswer
from ocs_llm_answerer.database.questions import question_values_from_cached, upsert_question

_SCHEMA_FILE = "schema.sql"


async def init_sqlite(database_path: Path, schema_path: Path | None = None) -> None:
    """创建当前数据库结构，可对相同结构重复初始化。

    早期项目直接维护当前 schema，不迁移或识别历史结构。结构改变后使用
    新数据库文件，避免自动改写已有题目和审计数据。

    Args:
        database_path: 数据库文件路径，父目录不存在时会创建。
        schema_path: 可选建表脚本；默认使用包内 schema.sql。

    Raises:
        aiosqlite.Error: 数据库连接或建表失败。
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = _read_schema_sql(schema_path)
    async with aiosqlite.connect(database_path) as db:
        await db.execute("PRAGMA foreign_keys = ON")
        await db.executescript(schema_sql)
        await db.commit()


def _read_schema_sql(schema_path: Path | None) -> str:
    if schema_path is not None:
        return schema_path.read_text(encoding="utf-8")
    return files(__package__).joinpath(_SCHEMA_FILE).read_text(encoding="utf-8")


class AnswerCacheRepository:
    """基于 SQLite 的答案缓存仓储，负责缓存读写和命中统计。"""

    def __init__(self, database_path: Path) -> None:
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
            aiosqlite.Error: 查询或统计写入失败。
        """
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
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
        return CachedAnswer.model_validate(dict(row))

    async def set(self, record: CachedAnswer, seen_at_ns: int | None = None) -> None:
        """在同一短事务中写入题目与当前答案，并验证调用外键。

        调用方应先保存成功流水，再传入其整数 ID。写入失败时，题目更新和
        缓存更新一并回滚，避免留下部分修改。

        Args:
            record: 已通过题型校验、引用已存在调用的答案。
            seen_at_ns: 本次请求时间；缺省时使用当前 Unix 纳秒时间。

        Raises:
            aiosqlite.IntegrityError: 调用不存在或其他数据库约束被违反。
            aiosqlite.Error: 数据库写入失败。
        """
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await upsert_question(db, *question_values_from_cached(record), seen_at_ns=seen_at_ns)
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
