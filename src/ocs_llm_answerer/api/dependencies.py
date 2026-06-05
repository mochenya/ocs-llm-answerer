from __future__ import annotations

import json

from fastapi import Header, HTTPException, Request, status
from pydantic import ValidationError

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.core.models import AnswerRequest


def get_answer_service(request: Request) -> AnswerService:
    """读取 FastAPI 启动时组装好的全局服务。"""
    return request.app.state.answer_service


def verify_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """除非显式配置 OCS_LLM_ANSWERER_API_KEY，否则保持本地开发开放。"""
    expected = getattr(request.app.state, "app_api_key", None)
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def parse_answer_request(request: Request) -> AnswerRequest:
    """兼容标准 JSON，以及 OCS 用 text/plain 包起来的 JSON 请求体。"""
    try:
        # OCS 可能用 Content-Type: text/plain;charset=UTF-8 发送 JSON 字符串。
        # 如果不手动解析，FastAPI 会把整个请求体当成普通文本，直接返回 422。
        raw_payload_json = (await request.body()).decode("utf-8")
        raw_payload = json.loads(raw_payload_json)
        answer_request = AnswerRequest.model_validate(raw_payload)
        answer_request.set_raw_payload_json(raw_payload_json)
        return answer_request
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
            detail=exc.errors(),
        ) from exc
