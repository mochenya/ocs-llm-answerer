import json

import pytest

from ocs_llm_answerer.answer import normalization
from ocs_llm_answerer.answer.normalization import (
    build_normalized_question,
    build_question_hash,
    normalize_options,
    normalize_request,
    normalize_text,
)
from ocs_llm_answerer.core.models import AnswerRequest


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ('len("a  b") 的结果', 'len("a b") 的结果'),
        ("if ready:\n    work()\ndone()", "if ready:\n    work()\n    done()"),
        ("第一行\n第二行", "第一行 第二行"),
        ("a\tb", "a b"),
    ],
)
def test_semantic_whitespace_produces_distinct_question_keys(first, second):
    """字符串内容、缩进、换行和制表符不能被折叠成同一道题。"""
    assert (
        build_question_hash(AnswerRequest(title=first))[0]
        != build_question_hash(AnswerRequest(title=second))[0]
    )


def test_transport_variants_share_a_question_key():
    """外围空白、换行编码和选项容器差异不影响同一道题的身份。"""
    first = AnswerRequest(title="  第一行\r\n第二行  ", type="single", options=" A. x\r\nB. y\n")
    second = AnswerRequest(title="第一行\n第二行", type="single", options=["A. x", " B. y "])

    assert build_question_hash(first) == build_question_hash(second)


def test_option_contents_and_order_are_part_of_question_identity():
    """保留选项内部空白及顺序，避免把答案编号用于另一组选项。"""
    requests = [
        AnswerRequest(title="选择字符串", options=['A. "a  b"', 'B. "a b"']),
        AnswerRequest(title="选择字符串", options=['A. "a b"', 'B. "a b"']),
        AnswerRequest(title="选择字符串", options=['B. "a b"', 'A. "a  b"']),
    ]

    assert len({build_question_hash(request)[0] for request in requests}) == 3


@pytest.mark.parametrize("options", [None, "", " \n\t", [], [" ", "\n"]])
def test_empty_options_are_equivalent(options):
    """没有有效选项的载荷统一表示为 None。"""
    assert normalize_options(options) is None


def test_array_option_preserves_embedded_line_breaks_and_indentation():
    """数组中的一个多行选项保持为一个选项。"""
    assert normalize_options([" if ready:\r\n    work() "]) == ["if ready:\n    work()"]


def test_normalization_is_idempotent_and_preserves_raw_payload():
    """重复标准化不会改变身份，模型输入与存储字段一致，原始快照不变。"""
    request = AnswerRequest(title=" 代码：\r\n    x = 'a  b' ", options=" A. 1\r\n B. 2 ")
    raw = '{"title":"原始审计快照","extra":true}'
    request.set_raw_payload_json(raw)

    normalized = normalize_request(request)
    stored = build_normalized_question(normalized)

    assert normalized == normalize_request(normalized)
    assert normalized.title == stored.question == "代码：\n    x = 'a  b'"
    assert normalized.options == json.loads(stored.options_json)
    assert stored.question_raw_json == normalized.raw_payload_json == raw
    assert request.title.startswith(" ")
    assert isinstance(request.options, str)


def test_question_key_version_isolates_normalization_rules(monkeypatch):
    """提升规则版本会产生独立的缓存身份，不回退查询旧键。"""
    request = AnswerRequest(title="1+1=?")
    old_key = build_question_hash(request)[0]

    monkeypatch.setattr(normalization, "_QUESTION_HASH_VERSION", 2)

    assert build_question_hash(request)[0] != old_key


def test_normalize_text_preserves_internal_whitespace():
    """外围清理不影响正文中的连续空格和缩进。"""
    assert normalize_text("\n  题目：\r    a  b\n") == "题目：\n    a  b"
