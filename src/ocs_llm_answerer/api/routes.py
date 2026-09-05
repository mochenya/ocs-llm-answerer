from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.dependencies import (
    get_answer_service,
    parse_answer_request,
    verify_api_key,
)
from ocs_llm_answerer.core.models import AnswerRequest, AnswerResponse

router = APIRouter()


@router.head("/")
async def root_status_probe() -> Response:
    return Response(status_code=200)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ocs-answerer.json")
async def ocs_answerer_config(request: Request) -> list[dict[str, object]]:
    """返回不含访问密钥的公开 OCS 订阅模板。

    订阅允许匿名读取，因此不能回传应用密钥。启用鉴权时，用户需要在
    OCS 自定义题库配置中手动填写 X-API-Key。

    Args:
        request: 用于生成服务地址和读取 OCS 请求模式的 HTTP 请求。

    Returns:
        OCS 可导入的配置数组，其中仅包含公开信息。
    """

    config: dict[str, object] = {
        "name": "OCS LLM Answerer",
        "homepage": str(request.base_url).rstrip("/"),
        "url": str(request.url_for("answer_question")),
        "method": "post",
        "contentType": "json",
        "headers": {"Content-Type": "application/json"},
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
