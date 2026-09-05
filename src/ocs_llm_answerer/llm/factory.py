"""从显式配置创建具体 Provider；业务代码只依赖 provider 模块的契约。"""

import os
from collections.abc import Callable

from ocs_llm_answerer.core.config import (
    ProviderAdapter,
    ProviderConfig,
    ProvidersConfig,
    load_environment,
)
from ocs_llm_answerer.llm.openai_responses import OpenAIResponsesProvider
from ocs_llm_answerer.llm.provider import LLMProvider

_PROVIDER_FACTORIES: dict[ProviderAdapter, Callable[[str, ProviderConfig, str], LLMProvider]] = {
    ProviderAdapter.OPENAI_RESPONSES: OpenAIResponsesProvider,
}


def create_provider(config: ProvidersConfig) -> LLMProvider:
    """读取环境密钥并创建当前启用的 Provider。

    Args:
        config: 已校验且包含活动 Provider 的配置。

    Returns:
        由调用方负责关闭的 Provider 实例。

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
    return create_provider_from_config(config.active_provider, provider_config, api_key)


def create_provider_from_config(
    provider_name: str, provider_config: ProviderConfig, api_key: str
) -> LLMProvider:
    """通过显式注册表构建适配器，不引入动态插件加载。

    Args:
        provider_name: 配置中的实例名称。
        provider_config: 模型和连接配置。
        api_key: 不进入持久化配置的访问密钥。

    Returns:
        与通用 Provider 契约兼容的实例。

    Raises:
        RuntimeError: 适配器未注册。
    """
    try:
        factory = _PROVIDER_FACTORIES[provider_config.adapter]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported provider adapter: {provider_config.adapter}") from exc
    return factory(provider_name, provider_config, api_key)
