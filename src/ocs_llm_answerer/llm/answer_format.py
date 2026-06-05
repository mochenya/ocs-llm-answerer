from __future__ import annotations

import json
import re

from ocs_llm_answerer.core.models import AnswerRequest, OCSQuestionType

_OCS_MULTI_SEPARATOR = "#"
_ANSWER_SEPARATOR = re.compile(r"\s*(?:===|###|---|[#|,，;；、/])\s*")
_OPTION_LETTERS = re.compile(r"[A-Za-z]{2,}")
_OPTION_LETTER = re.compile(r"[A-Za-z]")


def normalize_answer_for_ocs(request: AnswerRequest, answer: str) -> str:
    """把少量答案格式偏差标准化为 OCS 友好的字符串。"""
    value = answer.strip()
    if request.question_type not in {OCSQuestionType.MULTIPLE, OCSQuestionType.COMPLETION}:
        return value

    json_array = _parse_answer_json_array(value)
    if json_array is not None:
        return _OCS_MULTI_SEPARATOR.join(json_array)

    if request.question_type != OCSQuestionType.MULTIPLE:
        return value

    parts = _split_multiple_choice_answer(value)
    return _OCS_MULTI_SEPARATOR.join(parts) if parts is not None else value


def format_answers_for_ocs(request: AnswerRequest, answers: list[str]) -> str:
    """把结构化答案片段转换为 OCS 约定的答案字符串。"""
    parts = [answer.strip() for answer in answers if answer.strip()]
    if not parts:
        raise ValueError("answers must not be empty")

    if request.question_type == OCSQuestionType.MULTIPLE:
        return _OCS_MULTI_SEPARATOR.join(_normalize_multiple_choice_parts(parts))

    if request.question_type == OCSQuestionType.COMPLETION:
        if len(parts) == 1:
            return normalize_answer_for_ocs(request, parts[0])
        return _OCS_MULTI_SEPARATOR.join(parts)

    if request.question_type == OCSQuestionType.JUDGEMENT:
        return _normalize_judgement_answer(parts[0])

    if len(parts) == 1:
        return parts[0]
    return _OCS_MULTI_SEPARATOR.join(parts)


def _normalize_multiple_choice_parts(parts: list[str]) -> list[str]:
    normalized: list[str] = []
    for part in parts:
        json_array = _parse_answer_json_array(part)
        if json_array is not None:
            normalized.extend(_normalize_multiple_choice_parts(json_array))
            continue

        split = _split_multiple_choice_answer(part)
        if split is not None:
            normalized.extend(split)
            continue

        normalized.append(part.strip().upper().strip(".。:："))
    return normalized


def _normalize_judgement_answer(answer: str) -> str:
    value = answer.strip().lower()
    if value in {"true", "t", "yes", "y", "正确", "对", "是"}:
        return "true"
    if value in {"false", "f", "no", "n", "错误", "错", "否"}:
        return "false"
    return value


def _parse_answer_json_array(value: str) -> list[str] | None:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return None
    parts = [item.strip() for item in parsed if item.strip()]
    return parts or None


def _split_multiple_choice_answer(value: str) -> list[str] | None:
    if _OPTION_LETTERS.fullmatch(value):
        return [letter.upper() for letter in value]

    parts = [part.strip().upper().strip(".。:：") for part in _ANSWER_SEPARATOR.split(value)]
    if len(parts) < 2 or any(_OPTION_LETTER.fullmatch(part) is None for part in parts):
        return None
    return parts
