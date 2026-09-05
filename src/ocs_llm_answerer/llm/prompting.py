from __future__ import annotations

from importlib.resources import files

from ocs_llm_answerer.answer.models import Question

SYSTEM_PROMPT = files(__package__).joinpath("prompts/answer_system.txt").read_text(encoding="utf-8")


def build_user_input(request: Question) -> str:
    """将内部题目转换为模型输入，不读取原始 HTTP 载荷。

    Args:
        request: 选项已转换为有序列表的题目。

    Returns:
        保留原有提示词分段和题目内容的文本。
    """
    option_text = "\n".join(request.options or [])

    return "\n".join(
        [
            "<question>",
            f"题干：{request.title}",
            f"题型：{request.question_type or '未知'}",
            "</question>",
            "<options>",
            option_text or "无",
            "</options>",
        ]
    )
