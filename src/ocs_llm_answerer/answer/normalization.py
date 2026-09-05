"""统一题目身份和模型输入，保留可能影响答案的空白与选项顺序。"""

from __future__ import annotations

import hashlib
import json

from ocs_llm_answerer.answer.models import NormalizedQuestion, Question

_QUESTION_HASH_VERSION = 1


def normalize_text(value: str) -> str:
    """统一换行并去除外围空白，保留内部空格、缩进和换行。

    Args:
        value: 题干或单个选项的原始文本。

    Returns:
        适合参与题目身份计算和模型输入的文本。
    """
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def normalize_options(options: list[str] | str | None) -> list[str] | None:
    """将 OCS 换行选项或数组转换为有序列表。

    空白选项会被忽略；有效选项的内部内容和相对顺序保持不变。
    数组元素内的换行属于选项内容，不会继续拆分。

    Args:
        options: OCS 提供的选项数组、换行字符串或缺省值。

    Returns:
        标准化的非空选项列表；没有有效选项时返回 None。
    """
    if options is None:
        return None
    values = normalize_text(options).split("\n") if isinstance(options, str) else options
    return [normalize_text(option) for option in values if option.strip()] or None


def normalize_question(request: Question) -> Question:
    """创建模型输入与缓存共用的题目副本。

    Args:
        request: 已完成传输层转换的内部题目。

    Returns:
        标准化后的副本，不修改输入对象。
    """
    return request.model_copy(
        update={
            "title": normalize_text(request.title),
            "options": normalize_options(request.options),
        }
    )


def build_question_hash(request: Question) -> tuple[str, list[str] | None]:
    """按版本化的标准化题目内容生成稳定标识。

    版本是哈希输入的一部分。修改标准化规则时提升版本即可隔离不同规则，
    不需要读取旧格式的缓存键。

    Args:
        request: 已校验的题目请求，允许尚未标准化。

    Returns:
        SHA-256 十六进制标识和参与标识计算的选项列表。
    """
    normalized_options = normalize_options(request.options)
    payload = {
        "version": _QUESTION_HASH_VERSION,
        "title": normalize_text(request.title),
        "type": request.question_type or "",
        "options": normalized_options,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest(), normalized_options


def build_normalized_question(request: Question) -> NormalizedQuestion:
    """构造缓存和调用日志共用的题目记录。

    Args:
        request: 内部题目，不包含数据库编码或请求快照。

    Returns:
        包含题目身份和标准化字段的内部记录。
    """
    question_hash, normalized_options = build_question_hash(request)
    return NormalizedQuestion(
        question_hash=question_hash,
        question=normalize_text(request.title),
        question_type=request.question_type,
        options=normalized_options,
    )
