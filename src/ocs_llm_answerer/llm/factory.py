"""从显式配置创建具体 LLMClient；业务代码只依赖 llm/client.py 的协议。"""

from __future__ import annotations

import os
from collections.abc import Callable

from ocs_llm_answerer.core.config import (
    ProviderAdapter,
    ProviderConfig,
    ProvidersConfig,
    load_environment,
)
from ocs_llm_answerer.llm.adapters.openai_responses import OpenAIResponsesClient
from ocs_llm_answerer.llm.client import LLMClient
from ocs_llm_answerer.llm.models import LLMProviderMetadata

_CLIENT_FACTORIES: dict[ProviderAdapter, Callable[[str, ProviderConfig, str], LLMClient]] = {
    ProviderAdapter.OPENAI_RESPONSES: OpenAIResponsesClient,
}


def provider_metadata_from_config(config: ProviderConfig) -> LLMProviderMetadata:
    """将运行配置转换为与 SDK 无关的审计快照，不读取或保存密钥值。

    Args:
        config: 已校验的单个 Provider 运行时配置。

    Returns:
        可供调用审计持久化的不可变配置快照。
    """
    return LLMProviderMetadata(
        adapter=config.adapter,
        base_url=config.base_url,
        api_key_env=config.api_key_env,
        model=config.model,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        extra_body=config.extra_body,
    )


def create_llm_client(config: ProvidersConfig) -> LLMClient:
    """读取环境密钥并创建当前启用的 LLMClient。

    Args:
        config: 已校验且包含活动 Provider 的配置。

    Returns:
        由调用方负责关闭的 LLMClient 实例。

    Raises:
        RuntimeError: 密钥环境变量缺失或适配器未注册。
    """
    load_environment()
    provider_config = config.active
    api_key = os.getenv(provider_config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key environment variable for provider '{config.active_provider}': "
            f"{provider_config.api_key_env}"
        )
    return create_llm_client_from_config(config.active_provider, provider_config, api_key)


def create_llm_client_from_config(
    provider_name: str, provider_config: ProviderConfig, api_key: str
) -> LLMClient:
    """通过显式注册表构建客户端适配器，不引入动态插件加载。

    Args:
        provider_name: 配置中的实例名称。
        provider_config: 模型和连接配置。
        api_key: 不进入持久化配置的访问密钥。

    Returns:
        满足 LLMClient 协议的适配器实例。

    Raises:
        RuntimeError: 适配器未注册。
    """
    try:
        factory = _CLIENT_FACTORIES[provider_config.adapter]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported provider adapter: {provider_config.adapter}") from exc
    return factory(provider_name, provider_config, api_key)
