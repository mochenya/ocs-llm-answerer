"""在缓存边界统一校验题型语义，并生成 OCS 答案字符串。"""

from __future__ import annotations

import re

from ocs_llm_answerer.answer.models import OCSQuestionType, Question
from ocs_llm_answerer.answer.normalization import normalize_options

_LETTER = re.compile(r"([A-Za-z])[.。:：]?")
_LETTERS = re.compile(r"[A-Za-z]{2,}")
_SEPARATOR = re.compile(r"\s*(?:===|###|---|[#|,，;；、/])\s*")
_OPTION_PREFIX = re.compile(r"^[A-Za-z][.．、:：)）]\s*")
_JUDGEMENT_VALUES = {
    **dict.fromkeys(("true", "t", "yes", "y", "正确", "对", "是"), "true"),
    **dict.fromkeys(("false", "f", "no", "n", "错误", "错", "否"), "false"),
}


class InvalidAnswerError(ValueError):
    """模型答案违反题型规则，不能作为成功答案返回或缓存。"""


def validate_and_format_answer(request: Question, answers: list[str]) -> str:
    """校验完整答案列表，再生成 OCS 使用的分隔字符串。

    单选和判断题必须恰好有一项；选择题允许编号或与选项精确匹配的文本。
    多选结果去重并按编号排序。填空项只去除外围空白，不拆分其标点或 JSON
    文本，也不忽略空项，避免多空答案错位。未知题型仅检查答案非空。

    Args:
        request: 当前题目的类型和有序选项。
        answers: Provider 返回的结构化答案项，尚未做 OCS 格式转换。

    Returns:
        已通过题型校验的答案；多项使用 # 拼接。

    Raises:
        InvalidAnswerError: 答案为空、数量错误、判断值非法或选择项无效。
    """
    parts = [answer.strip() for answer in answers]
    if not parts or any(not part for part in parts):
        raise InvalidAnswerError("answer items must not be empty")

    question_type = request.question_type
    if question_type in {OCSQuestionType.SINGLE, OCSQuestionType.JUDGEMENT} and len(parts) != 1:
        raise InvalidAnswerError(f"{question_type} requires exactly one answer")

    if question_type == OCSQuestionType.JUDGEMENT:
        value = _JUDGEMENT_VALUES.get(parts[0].lower())
        if value is None:
            raise InvalidAnswerError("judgement answer must be true or false")
        return value

    options = normalize_options(request.options)
    if question_type == OCSQuestionType.SINGLE:
        return _choice_letter(parts[0], options)

    if question_type == OCSQuestionType.MULTIPLE:
        letters = {
            _choice_letter(item, options)
            for part in parts
            for item in _multiple_choice_items(part, options)
        }
        return "#".join(sorted(letters))

    return "#".join(parts)


def _choice_letter(answer: str, options: list[str] | None) -> str:
    """将一个选择项校验并转换为大写字母编号。

    Args:
        answer: 一个编号或完整选项文本。
        options: 按 A、B 等编号顺序排列的选项；缺省时无法验证编号上界。

    Returns:
        通过校验的大写选项编号。

    Raises:
        InvalidAnswerError: 编号越界，或文本无法唯一对应到一个选项。
    """
    match = _LETTER.fullmatch(answer)
    if match:
        letter = match[1].upper()
        if options is not None and ord(letter) - ord("A") >= len(options):
            raise InvalidAnswerError(f"option {letter} is outside the available options")
        return letter

    matches = _matching_option_indices(answer, options)
    if len(matches) == 1 and matches[0] < 26:
        return chr(ord("A") + matches[0])
    raise InvalidAnswerError("choice answer must be a valid letter or unambiguous option text")


def _matching_option_indices(answer: str, options: list[str] | None) -> list[int]:
    """查找与答案精确匹配的完整选项或去除编号后的选项文本。

    Args:
        answer: 已清理外围空白的答案文本。
        options: 标准化的选项列表。

    Returns:
        所有匹配位置；保留多重匹配以便调用方拒绝歧义。
    """
    return [
        index
        for index, option in enumerate(options or [])
        if answer == option or answer == _OPTION_PREFIX.sub("", option)
    ]


def _multiple_choice_items(answer: str, options: list[str] | None) -> list[str]:
    """拆分常见的合并编号，同时保留匹配选项的完整文本。

    Args:
        answer: 一个结构化答案项，可能包含 AC 或 A,C 等合并编号。
        options: 用于避免误拆完整选项文本的有序选项。

    Returns:
        待逐项校验的编号或选项文本；空项不会被静默丢弃。
    """
    if _matching_option_indices(answer, options):
        return [answer]
    if _LETTERS.fullmatch(answer):
        return list(answer)
    return _SEPARATOR.split(answer)
