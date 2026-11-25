"""将 OCS 标准化题目和校验失败信息转换为厂商无关的 LLM 多模态用户输入。"""

from __future__ import annotations

from importlib.resources import files

from ocs_llm_answerer.answer.models import Question
from ocs_llm_answerer.llm.models import ContentPart, ImagePart, TextPart, UserInput

SYSTEM_PROMPT = files(__package__).joinpath("prompts/answer_system.txt").read_text(encoding="utf-8")


def build_question_text(request: Question) -> str:
    """将内部题目格式化为带分段标签的提示文本，不读取原始 HTTP 载荷。

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


def build_question_input(request: Question) -> UserInput:
    """根据题目是否含图片，构造业务无关的多模态用户输入。

    Args:
        request: 已标准化的内部题目。

    Returns:
        纯文本或按 OCS `[图片N]` 占位符编号排列的多模态 UserInput。
    """
    text = build_question_text(request)
    if not request.images:
        return UserInput.from_text(text)

    parts: list[ContentPart] = [TextPart(text=text)]
    for number, image in enumerate(request.images, start=1):
        parts.append(TextPart(text=f"[图片{number}]"))
        parts.append(ImagePart(media_type=image.media_type, base64_data=image.base64_data))
    return UserInput(parts=tuple(parts))


def build_correction_input(error: Exception) -> UserInput:
    """将结构解析或题型校验报错整理为下一轮纠正提示。

    Args:
        error: 上一轮模型输出触发的解析或业务规则异常。

    Returns:
        追加到当前会话的纯文本纠正输入。
    """
    return UserInput.from_text(
        "\n".join(
            [
                "你上一轮的输出未通过后端解析或题型规则校验，请根据以下错误重新作答：",
                f"错误原因：{error}",
                "请严格遵守系统提示词中的题型规则，只返回合法的 JSON 对象，"
                "不要使用 markdown 代码块或额外说明。",
            ]
        )
    )
