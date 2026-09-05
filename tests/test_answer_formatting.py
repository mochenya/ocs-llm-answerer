import pytest

from ocs_llm_answerer.answer.formatting import InvalidAnswerError, validate_and_format_answer
from ocs_llm_answerer.core.models import AnswerRequest


@pytest.mark.parametrize(
    ("question_type", "answers", "expected"),
    [
        ("single", [" a. "], "A"),
        ("single", ["B"], "B"),
        ("single", ["苹果"], "A"),
        ("single", ["B. 香蕉"], "B"),
        ("multiple", ["B", "A", "A"], "A#B"),
        ("multiple", ["BA"], "A#B"),
        ("multiple", ["b,a"], "A#B"),
        ("multiple", ["A#B", "B."], "A#B"),
        ("multiple", ["苹果", "香蕉"], "A#B"),
        ("judgement", ["True"], "true"),
        ("judgement", ["正确"], "true"),
        ("judgement", ["错误"], "false"),
        ("judgement", ["No"], "false"),
        ("completion", [" 张三 ", "李四"], "张三#李四"),
        ("completion", ["张三，李四"], "张三，李四"),
        ("completion", ['["A", "B"]'], '["A", "B"]'),
        ("completion", ['len("a  b")'], 'len("a  b")'),
        (None, ["按题意作答", "补充信息"], "按题意作答#补充信息"),
    ],
)
def test_valid_answers_follow_question_type_rules(question_type, answers, expected):
    """题型校验允许可确定的格式偏差，且不改写填空内容。"""
    request = AnswerRequest(title="题目", type=question_type, options=["A. 苹果", "B. 香蕉"])

    assert validate_and_format_answer(request, answers) == expected


@pytest.mark.parametrize(
    ("question_type", "answers"),
    [
        ("single", []),
        ("single", [" "]),
        ("single", ["A", "B"]),
        ("single", ["AB"]),
        ("single", ["C"]),
        ("single", ["不存在的选项"]),
        ("multiple", []),
        ("multiple", ["ABC"]),
        ("multiple", ["A", "Z"]),
        ("multiple", ["A", ""]),
        ("multiple", ["A,"]),
        ("multiple", ["不存在的选项"]),
        ("judgement", ["maybe"]),
        ("judgement", ["true", "false"]),
        ("judgement", []),
        ("completion", []),
        ("completion", ["第一空", "", "第三空"]),
        (None, [""]),
    ],
)
def test_invalid_answers_are_rejected(question_type, answers):
    """数量、编号范围、判断值和空项错误必须在进入缓存前被拒绝。"""
    request = AnswerRequest(title="题目", type=question_type, options=["A. 苹果", "B. 香蕉"])

    with pytest.raises(InvalidAnswerError):
        validate_and_format_answer(request, answers)


def test_multiple_choice_preserves_full_option_text_before_splitting():
    """完整英文单词和带分隔符的选项文本不能被误拆成一组编号。"""
    request = AnswerRequest(title="选择", type="multiple", options=["A. list", "B. A,B"])

    assert validate_and_format_answer(request, ["list", "A,B"]) == "A#B"


def test_ambiguous_option_text_is_rejected():
    """两个选项具有相同文本时，要求模型给出确定的编号。"""
    request = AnswerRequest(title="选择", type="single", options=["A. 相同", "B. 相同"])

    with pytest.raises(InvalidAnswerError):
        validate_and_format_answer(request, ["相同"])


def test_missing_options_only_allow_letter_validation():
    """缺少选项列表时允许编号，但不能把无法核实的文本映射为编号。"""
    request = AnswerRequest(title="选择", type="single")

    assert validate_and_format_answer(request, ["Z"]) == "Z"
    with pytest.raises(InvalidAnswerError):
        validate_and_format_answer(request, ["选项内容"])
