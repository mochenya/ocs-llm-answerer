from __future__ import annotations

import os
from typing import Protocol

from ocs_llm_answerer.core.config import (
    ProviderAdapter,
    ProviderConfig,
    ProvidersConfig,
    load_environment,
)
from ocs_llm_answerer.core.models import AnswerRequest, LLMCallResult, LLMProviderMetadata
from ocs_llm_answerer.llm.openai_responses import OpenAIResponsesProvider


class LLMProvider(Protocol):
    """AnswerService 和测试使用的最小 provider 契约。"""

    name: str
    model: str
    request_metadata: LLMProviderMetadata

    async def answer(self, request: AnswerRequest) -> LLMCallResult: ...


_PROVIDER_FACTORIES = {
    # 保持显式注册：新增 provider 时添加一个 adapter 文件，再在这里加一行，
    # 不引入动态插件系统。
    ProviderAdapter.OPENAI_RESPONSES: OpenAIResponsesProvider,
}


def create_provider(config: ProvidersConfig) -> LLMProvider:
    """创建当前启用的 provider，避免把 API key 存进 providers.json。"""
    load_environment()
    provider_name = config.active_provider
    provider_config = config.active
    api_key = os.getenv(provider_config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key environment variable for provider '{provider_name}': "
            f"{provider_config.api_key_env}"
        )

    return create_provider_from_config(provider_name, provider_config, api_key)


def create_provider_from_config(
    provider_name: str,
    provider_config: ProviderConfig,
    api_key: str,
) -> LLMProvider:
    try:
        factory = _PROVIDER_FACTORIES[provider_config.adapter]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported provider adapter: {provider_config.adapter}") from exc
    return factory(provider_name, provider_config, api_key)
