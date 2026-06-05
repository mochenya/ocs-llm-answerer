from __future__ import annotations

from importlib.resources import files

from ocs_llm_answerer.core.models import AnswerRequest

SYSTEM_PROMPT = files(__package__).joinpath("prompts/answer_system.txt").read_text(encoding="utf-8")


def build_user_input(request: AnswerRequest) -> str:
    options = request.options
    option_text = "\n".join(options) if isinstance(options, list) else options or ""

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
