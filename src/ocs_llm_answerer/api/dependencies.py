from __future__ import annotations

import json
from dataclasses import dataclass

from fastapi import Header, HTTPException, Request, status
from pydantic import ValidationError

from ocs_llm_answerer.answer.models import Question, RequestAudit
from ocs_llm_answerer.answer.normalization import normalize_options
from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.schemas import AnswerRequest


@dataclass(frozen=True)
class ParsedAnswerRequest:
    """HTTP 边界解析后的业务输入与原始审计快照。"""

    question: Question
    audit: RequestAudit


def get_answer_service(request: Request) -> AnswerService:
    """读取启动阶段组装的答题服务。

    Args:
        request: 当前 HTTP 请求。

    Returns:
        已配置的答题服务实例。

    Raises:
        RuntimeError: 应用生命周期没有正确初始化服务。
    """
    service = getattr(request.app.state, "answer_service", None)
    if not isinstance(service, AnswerService):
        raise RuntimeError("Answer service is not initialized")
    return service


def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """只在配置了访问密钥时启用鉴权。

    Args:
        request: 用于读取应用设置的当前请求。
        x_api_key: 请求头提供的可选密钥。

    Raises:
        HTTPException: 配置密钥后，缺失或错误密钥返回 401。
    """
    expected = getattr(request.app.state, "app_api_key", None)
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def parse_answer_request(request: Request) -> ParsedAnswerRequest:
    """解析标准 JSON 或 OCS 使用 text/plain 发送的 JSON 载荷。

    Args:
        request: 待读取的 HTTP 请求。

    Returns:
        已校验的内部题目及独立的原始 UTF-8 JSON 快照。

    Raises:
        HTTPException: 编码、JSON 语法或字段校验失败时返回 422。
    """
    try:
        # OCS 可能用 Content-Type: text/plain;charset=UTF-8 发送 JSON 字符串。
        # 如果不手动解析，FastAPI 会把整个请求体当成普通文本，直接返回 422。
        raw_payload_json = (await request.body()).decode("utf-8")
        raw_payload = json.loads(raw_payload_json)
        answer_request = AnswerRequest.model_validate(raw_payload)
        return ParsedAnswerRequest(
            question=Question(
                title=answer_request.title,
                question_type=answer_request.question_type,
                options=normalize_options(answer_request.options),
            ),
            audit=RequestAudit(raw_payload_json=raw_payload_json),
        )
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Request body must be valid JSON",
        ) from exc
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Request body must be UTF-8 encoded",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=exc.errors(include_context=False),
        ) from exc
