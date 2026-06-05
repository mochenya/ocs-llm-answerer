from __future__ import annotations

import json
import time

import aiosqlite

from ocs_llm_answerer.core.models import CachedAnswer, NormalizedQuestion

_QUESTION_UPSERT_SQL = """
INSERT INTO questions (
    question_hash, question, question_type, options_json, question_raw_json, last_seen_at_ns
)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(question_hash) DO UPDATE SET
    question = excluded.question,
    question_type = excluded.question_type,
    options_json = excluded.options_json,
    question_raw_json = excluded.question_raw_json,
    last_seen_at_ns = excluded.last_seen_at_ns,
    updated_at = CURRENT_TIMESTAMP
"""


async def upsert_question(
    db: aiosqlite.Connection,
    question_hash: str,
    question: str,
    question_type: str | None,
    options_json: str | None,
    question_raw_json: str | None = None,
    seen_at_ns: int | None = None,
) -> str:
    if question_raw_json is None:
        question_raw_json = build_question_raw_json(question, question_type, options_json)
    if seen_at_ns is None:
        seen_at_ns = time.time_ns()
    await db.execute(
        _QUESTION_UPSERT_SQL,
        (question_hash, question, question_type, options_json, question_raw_json, seen_at_ns),
    )
    return question_raw_json


def build_question_raw_json(
    question: str,
    question_type: str | None,
    options_json: str | None,
) -> str:
    payload: dict[str, object] = {"title": question}
    if question_type is not None:
        payload["type"] = question_type
    if options_json is not None:
        payload["options"] = json.loads(options_json)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def question_values_from_cached(
    record: CachedAnswer,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        record.question_hash,
        record.question,
        record.question_type,
        record.options_json,
        record.question_raw_json,
    )


def question_values_from_normalized(
    question: NormalizedQuestion,
) -> tuple[str, str, str | None, str | None, str | None]:
    return (
        question.question_hash,
        question.question,
        question.question_type,
        question.options_json,
        question.question_raw_json,
    )
