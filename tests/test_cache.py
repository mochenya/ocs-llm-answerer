import asyncio
import sqlite3

from ocs_llm_answerer.core.models import CachedAnswer
from ocs_llm_answerer.database.cache import AnswerCache, init_sqlite


def test_answer_cache_uses_normalized_schema(tmp_path):
    database_path = tmp_path / "cache.sqlite3"
    record = CachedAnswer(
        question_hash="hash-1",
        call_hash="call-hash-1",
        question="1+1=?",
        question_type="single",
        options_json='["A. 2", "B. 3"]',
        answer="A",
        explanation="解析",
        confidence=0.9,
        provider="fake",
        model="fake-model",
    )

    async def exercise_cache():
        await init_sqlite(database_path)
        cache = AnswerCache(database_path)
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
            SELECT call_hash, answer, explanation, confidence, provider, model,
                   hit_count, last_hit_at_ns
            FROM answer_cache
            WHERE question_hash = ?
            """,
            (record.question_hash,),
        ).fetchone()

    assert question_row[:3] == ("1+1=?", "single", '["A. 2", "B. 3"]')
    assert isinstance(question_row[3], int)
    assert question_row[3] > 0
    assert answer_row[:7] == ("call-hash-1", "A", "解析", 0.9, "fake", "fake-model", 2)
    assert isinstance(answer_row[7], int)
    assert answer_row[7] > 0
