from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.api.routes import router
from ocs_llm_answerer.core.config import Settings, get_settings, load_providers_config
from ocs_llm_answerer.database.cache import AnswerCacheRepository, init_sqlite
from ocs_llm_answerer.database.llm_requests import LLMRequestRepository
from ocs_llm_answerer.llm.provider import LLMProvider, create_provider


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    """创建 ASGI 应用，并允许测试注入设置和 provider。"""
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动时统一构建运行时依赖，让路由处理函数保持简单。
        await init_sqlite(app_settings.app_database_path)
        llm_provider = provider or create_provider(
            load_providers_config(app_settings.app_providers_config_path)
        )
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

    app = FastAPI(title="OCS LLM Answerer Backend", lifespan=lifespan)
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
    return app


app = create_app()
