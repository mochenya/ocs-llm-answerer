from __future__ import annotations

import time
from importlib.resources import files
from pathlib import Path

import aiosqlite

from ocs_llm_answerer.core.models import CachedAnswer
from ocs_llm_answerer.database.questions import question_values_from_cached, upsert_question

_SCHEMA_FILE = "schema.sql"


async def init_sqlite(database_path: Path, schema_path: Path | None = None) -> None:
    """启动时为全新 checkout 创建数据库表。"""
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


class AnswerCache:
    """基于 SQLite 的小型缓存，用于确定性题目查询。"""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def get(
        self,
        question_hash: str,
        question_raw_json: str | None = None,
        seen_at_ns: int | None = None,
    ) -> CachedAnswer | None:
        """按稳定的题目哈希返回一条缓存答案。"""
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT q.question_hash, c.call_hash, q.question, q.question_type, q.options_json,
                       c.answer, c.explanation, c.confidence, c.provider, c.model
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
        """按题目哈希插入或更新最新的 provider 答案。"""
        if seen_at_ns is None:
            seen_at_ns = time.time_ns()
        async with aiosqlite.connect(self._database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            await upsert_question(db, *question_values_from_cached(record), seen_at_ns=seen_at_ns)
            await db.execute(
                """
                INSERT INTO answer_cache (
                    question_hash, call_hash, answer, explanation, confidence, provider, model
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_hash) DO UPDATE SET
                    call_hash = excluded.call_hash,
                    answer = excluded.answer,
                    explanation = excluded.explanation,
                    confidence = excluded.confidence,
                    provider = excluded.provider,
                    model = excluded.model,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    record.question_hash,
                    record.call_hash,
                    record.answer,
                    record.explanation,
                    record.confidence,
                    record.provider,
                    record.model,
                ),
            )
            await db.commit()
