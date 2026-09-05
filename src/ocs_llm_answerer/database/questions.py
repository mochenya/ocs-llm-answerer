from __future__ import annotations

import json
import time

import aiosqlite

from ocs_llm_answerer.answer.models import CachedAnswer, NormalizedQuestion, RequestAudit

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
    """在调用方事务中更新题目及最近请求快照，不自行提交。

    Args:
        db: 已启用外键的数据库连接。
        question_hash: 题目的稳定身份。
        question: 标准化题干。
        question_type: 题型或 None。
        options_json: 已编码的有序选项。
        question_raw_json: 原始请求；缺省时构造内容等价的 JSON。
        seen_at_ns: 请求进入时间，缺省时读取当前 Unix 纳秒时间。

    Returns:
        实际保存的请求快照，可用于写入独立审计流水。

    Raises:
        aiosqlite.Error: SQL 执行失败，由外层连接边界转换为仓储错误。
    """
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
    """为非 HTTP 调用构造可审计的题目 JSON。

    Args:
        question: 标准化题干。
        question_type: 题型或 None。
        options_json: 已编码的有序选项或 None。

    Returns:
        使用既有 OCS 字段名的 JSON 文本。
    """
    payload: dict[str, object] = {"title": question}
    if question_type is not None:
        payload["type"] = question_type
    if options_json is not None:
        payload["options"] = json.loads(options_json)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def question_values(
    record: CachedAnswer | NormalizedQuestion,
    audit: RequestAudit | None = None,
) -> tuple[str, str, str | None, str | None, str | None]:
    """将结构化题目转换为数据库参数，原始载荷不参与业务字段编码。

    Args:
        record: 具有稳定身份的题目或缓存记录。
        audit: 与业务数据分离的原始请求快照。

    Returns:
        哈希、题干、题型、选项 JSON 和可选原始快照。
    """
    return (
        record.question_hash,
        record.question,
        record.question_type,
        json.dumps(record.options, ensure_ascii=False) if record.options is not None else None,
        audit.raw_payload_json if audit is not None else None,
    )
