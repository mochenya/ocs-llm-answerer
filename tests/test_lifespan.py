"""验证应用与外部注入方各自的资源所有权。"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ocs_llm_answerer import main
from ocs_llm_answerer.core.config import ProviderConfig, Settings
from ocs_llm_answerer.llm.openai_responses import OpenAIResponsesProvider
from tests.fakes import StubProvider


@pytest.fixture
def owned_provider(monkeypatch: pytest.MonkeyPatch) -> StubProvider:
    """替换工厂和配置加载，不读取实际密钥。

    Args:
        monkeypatch: 自动恢复的测试替换工具。

    Returns:
        应用内部工厂返回的、可观察关闭次数的 Provider。
    """
    provider = StubProvider()
    monkeypatch.setattr(main, "load_providers_config", MagicMock())
    monkeypatch.setattr(main, "create_provider", MagicMock(return_value=provider))
    return provider


def test_app_closes_owned_provider(tmp_path: Path, owned_provider: StubProvider) -> None:
    """正常退出时只关闭自有 Provider 一次。"""
    app = main.create_app(Settings(app_database_path=tmp_path / "cache.db"))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert owned_provider.close_calls == 0
    assert owned_provider.close_calls == 1


def test_app_does_not_close_injected_provider(tmp_path: Path) -> None:
    """外部实例可被多个应用生命周期复用，应用不接管其释放责任。"""
    provider = StubProvider()
    app = main.create_app(Settings(app_database_path=tmp_path / "cache.db"), provider)
    for _ in range(2):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
    assert provider.close_calls == 0


def test_app_closes_owned_provider_after_initialization_failure(
    tmp_path: Path,
    owned_provider: StubProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider 创建后依赖组装失败，也必须释放已经取得的资源。"""
    monkeypatch.setattr(
        main, "AnswerService", MagicMock(side_effect=RuntimeError("assembly failed"))
    )
    app = main.create_app(Settings(app_database_path=tmp_path / "cache.db"))
    with pytest.raises(RuntimeError, match="assembly failed"), TestClient(app):
        pass
    assert owned_provider.close_calls == 1


def test_app_closes_owned_provider_when_lifespan_body_raises(
    tmp_path: Path,
    owned_provider: StubProvider,
) -> None:
    """运行阶段抛出异常时 finally 仍然执行，不仅覆盖正常退出路径。"""
    app = main.create_app(Settings(app_database_path=tmp_path / "cache.db"))

    async def exercise() -> None:
        """在已进入的生命周期中注入失败。"""
        async with app.router.lifespan_context(app):
            raise ValueError("runtime failed")

    with pytest.raises(ValueError, match="runtime failed"):
        asyncio.run(exercise())
    assert owned_provider.close_calls == 1


def test_provider_aclose_closes_real_sdk_client() -> None:
    """只创建并关闭客户端，不发送网络请求。"""
    config = ProviderConfig.model_validate(
        {
            "adapter": "openai_responses",
            "base_url": "https://test.invalid/v1",
            "api_key_env": "STUB_KEY",
            "model": "stub-model",
        }
    )
    provider = OpenAIResponsesProvider("stub", config, "stub-key")
    assert not provider._client.is_closed()
    asyncio.run(provider.aclose())
    assert provider._client.is_closed()
