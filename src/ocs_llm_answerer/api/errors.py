"""将应用错误映射为稳定且不泄露上游诊断信息的 HTTP 响应。"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.answer.formatting import InvalidAnswerError
from ocs_llm_answerer.llm.errors import (
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

_LOGGER = logging.getLogger(__name__)


async def handle_application_error(request: Request, exc: Exception) -> JSONResponse:
    """转换已注册的应用异常，保留既有 detail 响应形状。

    Args:
        request: 触发错误的 HTTP 请求。
        exc: 已注册的答题、Provider 或仓储异常。

    Returns:
        带有固定状态码及公开错误说明的 JSON 响应。
    """
    if isinstance(exc, InvalidAnswerError):
        status_code, detail = 502, f"Invalid model answer: {exc}"
    elif isinstance(exc, ProviderTimeoutError):
        status_code, detail = 504, "Model provider timed out"
    elif isinstance(exc, ProviderUnavailableError):
        status_code, detail = 503, "Model provider is unavailable"
    elif isinstance(exc, ProviderError):
        status_code, detail = 502, "Invalid model provider response"
    elif isinstance(exc, RepositoryError):
        status_code, detail = 503, "Answer storage is unavailable"
        _LOGGER.error(
            "Answer persistence failed for %s",
            request.url.path,
            exc_info=(type(exc), exc, exc.__traceback__),
        )
    else:
        raise exc
    return JSONResponse(status_code=status_code, content={"detail": detail})


def register_error_handlers(app: FastAPI) -> None:
    """注册可预期错误，不将未知编程错误伪装为上游故障。

    Args:
        app: 待配置的应用实例。
    """
    for error_type in (InvalidAnswerError, ProviderError, RepositoryError):
        app.add_exception_handler(error_type, handle_application_error)
