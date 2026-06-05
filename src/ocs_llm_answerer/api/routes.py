from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.dependencies import (
    get_answer_service,
    parse_answer_request,
    verify_api_key,
)
from ocs_llm_answerer.core.models import AnswerRequest, AnswerResponse

router = APIRouter()


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ocs-answerer.json")
async def ocs_answerer_config(request: Request) -> list[dict[str, object]]:
    """返回 OCS AnswererWrapper 订阅配置。"""
    headers: dict[str, str] = {"Content-Type": "application/json"}
    api_key = getattr(request.app.state, "app_api_key", None)
    if api_key:
        headers["X-API-Key"] = api_key

    config: dict[str, object] = {
        "name": "OCS LLM Answerer",
        "homepage": str(request.base_url).rstrip("/"),
        "url": str(request.url_for("answer_question")),
        "method": "post",
        "contentType": "json",
        "headers": headers,
        "data": {
            "title": "${title}",
            "type": "${type}",
            "options": "${options}",
        },
        "handler": "return (res)=> res.code === 1 ? [res.question, res.answer] : undefined",
    }
    request_type = getattr(request.app.state, "app_ocs_answerer_request_type", "fetch")
    if request_type == "GM_xmlhttpRequest":
        config["type"] = request_type

    # OCS 会把订阅结果当数组展开；这里必须返回 list，不能返回单个配置对象。
    return [config]


@router.post(
    "/api/v1/answer",
    response_model=AnswerResponse,
    dependencies=[Depends(verify_api_key)],
)
async def answer_question(
    payload: AnswerRequest = Depends(parse_answer_request),
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    return await service.answer(payload)
