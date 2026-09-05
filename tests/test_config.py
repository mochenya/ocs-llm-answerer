import json
import os

import pytest
from pydantic import ValidationError

from ocs_llm_answerer.core.config import (
    ProviderAdapter,
    ProviderConfig,
    ProvidersConfig,
    get_settings,
    load_providers_config,
)
from ocs_llm_answerer.llm import factory as provider_module


def test_load_providers_config(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "active_provider": "openai",
                "providers": {
                    "openai": {
                        "adapter": "openai_responses",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "OPENAI_API_KEY",
                        "model": "gpt-4o-mini",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_providers_config(path)

    assert config.active_provider == "openai"
    assert config.active.model == "gpt-4o-mini"


def test_get_settings_loads_dotenv_into_process_environment(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OCS_LLM_ANSWERER_API_KEY=local-app-key",
                "OCS_LLM_ANSWERER_DATABASE_PATH=runtime/cache.sqlite3",
                "OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH=runtime/providers.json",
                "OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE=GM_xmlhttpRequest",
                "OPENAI_API_KEY=local-provider-key",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCS_LLM_ANSWERER_API_KEY", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_DATABASE_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()

        assert settings.app_api_key == "local-app-key"
        assert settings.app_database_path.as_posix() == "runtime/cache.sqlite3"
        assert settings.app_providers_config_path.as_posix() == "runtime/providers.json"
        assert settings.app_ocs_answerer_request_type == "GM_xmlhttpRequest"
        assert os.getenv("OPENAI_API_KEY") == "local-provider-key"
    finally:
        get_settings.cache_clear()


def test_get_settings_accepts_legacy_question_env_names(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OCS_LLM_QUESTION_API_KEY=legacy-app-key",
                "OCS_LLM_QUESTION_DATABASE_PATH=legacy/cache.sqlite3",
                "OCS_LLM_QUESTION_PROVIDERS_CONFIG_PATH=legacy/providers.json",
                "OCS_LLM_QUESTION_OCS_ANSWERER_REQUEST_TYPE=GM_xmlhttpRequest",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OCS_LLM_ANSWERER_API_KEY", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_DATABASE_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE", raising=False)
    monkeypatch.delenv("OCS_LLM_QUESTION_API_KEY", raising=False)
    monkeypatch.delenv("OCS_LLM_QUESTION_DATABASE_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_QUESTION_PROVIDERS_CONFIG_PATH", raising=False)
    monkeypatch.delenv("OCS_LLM_QUESTION_OCS_ANSWERER_REQUEST_TYPE", raising=False)
    get_settings.cache_clear()

    try:
        settings = get_settings()

        assert settings.app_api_key == "legacy-app-key"
        assert settings.app_database_path.as_posix() == "legacy/cache.sqlite3"
        assert settings.app_providers_config_path.as_posix() == "legacy/providers.json"
        assert settings.app_ocs_answerer_request_type == "GM_xmlhttpRequest"
    finally:
        get_settings.cache_clear()


def test_create_provider_loads_custom_api_key_env_from_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("CUSTOM_LLM_API_KEY=custom-provider-key", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CUSTOM_LLM_API_KEY", raising=False)
    get_settings.cache_clear()

    class FakeProvider:
        def __init__(self, name: str, config: ProviderConfig, api_key: str) -> None:
            self.name = name
            self.model = config.model
            self.api_key = api_key
            self.request_metadata = config.to_metadata()

    monkeypatch.setitem(
        provider_module._PROVIDER_FACTORIES,
        ProviderAdapter.OPENAI_RESPONSES,
        FakeProvider,
    )
    config = ProvidersConfig(
        active_provider="custom",
        providers={
            "custom": ProviderConfig(
                adapter=ProviderAdapter.OPENAI_RESPONSES,
                base_url="https://api.example.test/v1",
                api_key_env="CUSTOM_LLM_API_KEY",
                model="custom-model",
            )
        },
    )

    try:
        provider = provider_module.create_provider(config)

        assert provider.api_key == "custom-provider-key"
    finally:
        get_settings.cache_clear()


def test_create_provider_rejects_missing_api_key_env(monkeypatch):
    monkeypatch.delenv("MISSING_PROVIDER_API_KEY", raising=False)
    config = ProvidersConfig(
        active_provider="custom",
        providers={
            "custom": ProviderConfig(
                adapter=ProviderAdapter.OPENAI_RESPONSES,
                base_url="https://api.example.test/v1",
                api_key_env="MISSING_PROVIDER_API_KEY",
                model="custom-model",
            )
        },
    )

    with pytest.raises(RuntimeError, match="Missing API key environment variable"):
        provider_module.create_provider(config)


def test_load_providers_config_rejects_missing_active_provider(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(
        json.dumps(
            {
                "active_provider": "missing",
                "providers": {
                    "openai": {
                        "adapter": "openai_responses",
                        "base_url": "https://api.openai.com/v1",
                        "api_key_env": "OPENAI_API_KEY",
                        "model": "gpt-4o-mini",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError):
        load_providers_config(path)


def test_providers_config_rejects_empty_providers():
    with pytest.raises(ValidationError):
        ProvidersConfig.model_validate({"active_provider": "openai", "providers": {}})


def test_provider_config_rejects_non_http_base_url():
    with pytest.raises(ValidationError, match="base_url"):
        ProviderConfig(
            adapter="openai_responses",
            base_url="not-a-url",
            api_key_env="OPENAI_API_KEY",
            model="gpt-4o-mini",
        )


def test_provider_config_rejects_invalid_api_key_env_name():
    with pytest.raises(ValidationError, match="api_key_env"):
        ProviderConfig(
            adapter="openai_responses",
            base_url="https://api.openai.com/v1",
            api_key_env="OPENAI API KEY",
            model="gpt-4o-mini",
        )
