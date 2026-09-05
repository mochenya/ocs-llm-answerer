from __future__ import annotations

from fastapi import APIRouter, Depends, Request, Response

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.dependencies import (
    ParsedAnswerRequest,
    get_answer_service,
    parse_answer_request,
    verify_api_key,
)
from ocs_llm_answerer.api.schemas import AnswerResponse, ErrorResponse

router = APIRouter()


@router.head("/")
async def root_status_probe() -> Response:
    """返回 OCS 状态探针使用的空成功响应。

    Returns:
        不含正文的 HTTP 200 响应。
    """
    return Response(status_code=200)


@router.get("/health")
async def health() -> dict[str, str]:
    """报告应用进程存活状态，不执行上游连通性检查。

    Returns:
        固定的存活状态。
    """
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
    responses={
        401: {"model": ErrorResponse, "description": "访问密钥缺失或无效"},
        422: {"model": ErrorResponse, "description": "请求编码、JSON 或字段校验失败"},
        502: {"model": ErrorResponse, "description": "上游响应或模型答案无效"},
        503: {"model": ErrorResponse, "description": "上游暂时不可用或持久化失败"},
        504: {"model": ErrorResponse, "description": "模型调用超时"},
    },
    openapi_extra={
        "requestBody": {
            "required": True,
            "content": {
                "application/json": {"schema": {"$ref": "#/components/schemas/AnswerRequest"}},
                "text/plain": {
                    "schema": {
                        "type": "string",
                        "description": "UTF-8 编码的 AnswerRequest JSON 文本",
                    },
                    "example": '{"title":"1+1=?","type":"single","options":["A. 2","B. 3"]}',
                },
            },
        }
    },
)
async def answer_question(
    payload: ParsedAnswerRequest = Depends(parse_answer_request),
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    """返回可供 OCS 使用的答案，并明确报告模型答案不合法的情况。

    Args:
        payload: 已解析和校验的 OCS 请求。
        service: 应用启动时组装的答题服务。

    Returns:
        来自缓存或本次模型调用的合法答案，保持原有 OCS 字段。
    """
    result = await service.answer(payload.question, audit=payload.audit)
    return AnswerResponse.model_validate(result.model_dump())
