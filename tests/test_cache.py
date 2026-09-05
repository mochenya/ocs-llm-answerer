import asyncio
import sqlite3
from pathlib import Path

import pytest

from ocs_llm_answerer.core.models import (
    CachedAnswer,
    LLMAnswer,
    LLMCallResult,
    LLMProviderMetadata,
    NormalizedQuestion,
)
from ocs_llm_answerer.database.cache import AnswerCacheRepository, init_sqlite
from ocs_llm_answerer.database.llm_requests import LLMRequestRepository


async def save_call(database_path: Path, record: CachedAnswer) -> int:
    """为测试答案创建真实调用流水，不绕过数据库关联约束。

    Args:
        database_path: 已初始化的临时数据库路径。
        record: 提供题目和答案内容的测试记录。

    Returns:
        已提交的调用流水整数主键。
    """
    return await LLMRequestRepository(database_path).record_success(
        NormalizedQuestion(
            question_hash=record.question_hash,
            question=record.question,
            question_type=record.question_type,
            options_json=record.options_json,
        ),
        LLMProviderMetadata(
            adapter="fake",
            base_url="https://test.invalid/v1",
            api_key_env="FAKE_API_KEY",
            model=record.model,
            timeout_seconds=1,
            max_retries=0,
        ),
        LLMCallResult(
            answer=LLMAnswer(
                answers=[record.answer],
                explanation=record.explanation,
                confidence=record.confidence,
            )
        ),
        seen_at_ns=1,
        request_started_at_ns=2,
        request_completed_at_ns=3,
        latency_ms=1,
    )


@pytest.fixture
def persisted_answer(tmp_path):
    """准备有真实来源调用、但尚未进入缓存的答案。"""
    database_path = tmp_path / "cache.sqlite3"
    record = CachedAnswer(
        question_hash="hash-1",
        llm_request_id=1,
        question="1+1=?",
        question_type="single",
        options_json='["A. 2", "B. 3"]',
        answer="A",
        explanation="解析",
        confidence=0.9,
        provider="fake",
        model="fake-model",
    )

    async def prepare():
        """初始化数据库并将实际调用主键写回测试记录。"""
        await init_sqlite(database_path)
        record.llm_request_id = await save_call(database_path, record)

    asyncio.run(prepare())
    return database_path, record


def test_answer_cache_uses_normalized_schema(persisted_answer):
    """缓存答案保留调用主键和原有命中统计行为。"""
    database_path, record = persisted_answer

    async def exercise_cache():
        """依次验证未命中、写入和两次命中。"""
        cache = AnswerCacheRepository(database_path)
        first_hit = await cache.get(record.question_hash)
        await cache.set(record)
        cached = await cache.get(record.question_hash)
        await cache.get(record.question_hash)
        return first_hit, cached

    first_hit, cached = asyncio.run(exercise_cache())

    assert first_hit is None
    assert cached == record

    with sqlite3.connect(database_path) as db:
        question_row = db.execute(
            """
            SELECT question, question_type, options_json, last_seen_at_ns
            FROM questions
            WHERE question_hash = ?
            """,
            (record.question_hash,),
        ).fetchone()
        answer_row = db.execute(
            """
            SELECT llm_request_id, answer, explanation, confidence, provider, model,
                   hit_count, last_hit_at_ns
            FROM answer_cache
            WHERE question_hash = ?
            """,
            (record.question_hash,),
        ).fetchone()

    assert question_row[:3] == ("1+1=?", "single", '["A. 2", "B. 3"]')
    assert isinstance(question_row[3], int)
    assert question_row[3] > 0
    assert answer_row[:7] == (record.llm_request_id, "A", "解析", 0.9, "fake", "fake-model", 2)
    assert isinstance(answer_row[7], int)
    assert answer_row[7] > 0


@pytest.mark.parametrize("existing_cache", [False, True])
def test_missing_call_is_rejected_and_question_changes_roll_back(persisted_answer, existing_cache):
    """不存在的调用 ID 不能用于新增或更新缓存，先前的题目修改也会回滚。"""
    database_path, record = persisted_answer
    invalid = record.model_copy(
        update={"llm_request_id": record.llm_request_id + 100, "question": "不应写入的题干"}
    )

    async def exercise():
        """在实际缓存写入入口触发外键约束。"""
        cache = AnswerCacheRepository(database_path)
        if existing_cache:
            await cache.set(record)
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            await cache.set(invalid)
        return await cache.get(record.question_hash)

    cached = asyncio.run(exercise())

    assert cached == (record if existing_cache else None)
    with sqlite3.connect(database_path) as db:
        assert db.execute("SELECT question FROM questions").fetchone()[0] == record.question
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_cache_can_reference_a_new_call_without_removing_history(persisted_answer):
    """相同题目、时间和配置的多次调用仍有独立 ID，缓存可以切换来源。"""
    database_path, record = persisted_answer

    async def exercise():
        """写入两次调用并更新当前缓存来源。"""
        cache = AnswerCacheRepository(database_path)
        await cache.set(record)
        second_id = await save_call(database_path, record)
        updated = record.model_copy(update={"llm_request_id": second_id})
        await cache.set(updated)
        return second_id, await cache.get(record.question_hash)

    second_id, cached = asyncio.run(exercise())

    assert isinstance(second_id, int)
    assert second_id != record.llm_request_id
    assert cached.llm_request_id == second_id
    with sqlite3.connect(database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM llm_requests").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 1
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []


def test_referenced_call_cannot_be_deleted_but_cache_can_be_cleared(persisted_answer):
    """来源流水受外键保护，清理缓存不会删除调用历史。"""
    database_path, record = persisted_answer
    asyncio.run(AnswerCacheRepository(database_path).set(record))

    with sqlite3.connect(database_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            db.execute("DELETE FROM llm_requests WHERE id = ?", (record.llm_request_id,))
        db.execute("DELETE FROM answer_cache WHERE question_hash = ?", (record.question_hash,))
        assert db.execute("SELECT COUNT(*) FROM llm_requests").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0


def test_deleting_question_keeps_existing_cascade_semantics(persisted_answer):
    """显式删除题目仍同时删除关联缓存与调用，不留下悬空引用。"""
    database_path, record = persisted_answer
    asyncio.run(AnswerCacheRepository(database_path).set(record))

    with sqlite3.connect(database_path) as db:
        db.execute("PRAGMA foreign_keys = ON")
        db.execute("DELETE FROM questions WHERE question_hash = ?", (record.question_hash,))
        assert db.execute("SELECT COUNT(*) FROM questions").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM llm_requests").fetchone()[0] == 0
        assert db.execute("PRAGMA foreign_key_check").fetchall() == []
