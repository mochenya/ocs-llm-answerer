from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.errors import register_error_handlers
from ocs_llm_answerer.api.openapi import configure_openapi
from ocs_llm_answerer.api.routes import router
from ocs_llm_answerer.core.config import Settings, get_settings, load_providers_config
from ocs_llm_answerer.database.cache import AnswerCacheRepository
from ocs_llm_answerer.database.connection import init_sqlite
from ocs_llm_answerer.database.llm_requests import LLMRequestRepository
from ocs_llm_answerer.llm.factory import create_provider
from ocs_llm_answerer.llm.provider import LLMProvider


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    """创建应用，只管理应用内部创建的 Provider 的生命周期。

    Args:
        settings: 应用设置；缺省时读取环境配置。
        provider: 可选外部实例，关闭责任始终由注入方承担。

    Returns:
        尚未启动生命周期的 FastAPI 应用。
    """
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """组装运行时依赖，并在退出或后续初始化失败时释放自有资源。

        Args:
            app: 当前应用实例。

        Yields:
            依赖就绪后的应用运行阶段。
        """
        # 启动时统一构建运行时依赖，让路由处理函数保持简单。
        await init_sqlite(app_settings.app_database_path)
        llm_provider = (
            provider
            if provider is not None
            else create_provider(load_providers_config(app_settings.app_providers_config_path))
        )
        try:
            cache_repository = AnswerCacheRepository(app_settings.app_database_path)
            request_repository = LLMRequestRepository(app_settings.app_database_path)
            app.state.answer_service = AnswerService(
                cache_repository=cache_repository,
                provider=llm_provider,
                request_repository=request_repository,
            )
            app.state.app_api_key = app_settings.app_api_key
            app.state.app_ocs_answerer_request_type = app_settings.app_ocs_answerer_request_type
            yield
        finally:
            if provider is None:
                await llm_provider.aclose()

    app = FastAPI(title="OCS LLM Answerer Backend", lifespan=lifespan)
    register_error_handlers(app)
    # OCS 页面运行在 chaoxing.com，但会从浏览器里调用本机 localhost 服务。
    # 如果没有 CORS，后端日志可能已经是 200 OK，页面仍会因为浏览器拦截响应而报
    # TypeError: Failed to fetch。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        # Chrome 可能把公网 HTTPS 页面访问 127.0.0.1 视为 Private Network Access，
        # 因此需要显式允许这类预检请求。
        allow_private_network=True,
    )
    app.include_router(router)
    configure_openapi(app)
    return app


app = create_app()
