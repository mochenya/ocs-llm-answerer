from __future__ import annotations

import json
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from ocs_llm_answerer.llm.models import LLMProviderMetadata

_API_KEY_ENV = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ENV_FILE = Path(".env")


class Settings(BaseSettings):
    """由环境变量驱动的应用设置。"""

    app_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OCS_LLM_ANSWERER_API_KEY", "OCS_LLM_QUESTION_API_KEY"),
    )
    app_database_path: Path = Field(
        default=Path("data/cache.sqlite3"),
        validation_alias=AliasChoices(
            "OCS_LLM_ANSWERER_DATABASE_PATH",
            "OCS_LLM_QUESTION_DATABASE_PATH",
        ),
    )
    app_providers_config_path: Path = Field(
        default=Path("config/providers.json"),
        validation_alias=AliasChoices(
            "OCS_LLM_ANSWERER_PROVIDERS_CONFIG_PATH",
            "OCS_LLM_QUESTION_PROVIDERS_CONFIG_PATH",
        ),
    )
    app_ocs_answerer_request_type: Literal["fetch", "GM_xmlhttpRequest"] = Field(
        default="fetch",
        validation_alias=AliasChoices(
            "OCS_LLM_ANSWERER_OCS_ANSWERER_REQUEST_TYPE",
            "OCS_LLM_QUESTION_OCS_ANSWERER_REQUEST_TYPE",
        ),
    )

    # Settings only models application config. Provider API keys may also live in .env;
    # extra="ignore" lets those keys coexist, while load_environment() exports them
    # to os.environ so providers can read the api_key_env configured in providers.json.
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


class ProviderAdapter(StrEnum):
    """providers.json 支持的 provider adapter 标识。"""

    OPENAI_RESPONSES = "openai_responses"


class ProviderConfig(BaseModel):
    """单个 LLM provider 条目的运行时配置。"""

    adapter: ProviderAdapter
    base_url: str = Field(min_length=1)
    api_key_env: str = Field(min_length=1)
    model: str = Field(min_length=1)
    timeout_seconds: float = Field(default=30, gt=0)
    max_retries: int = Field(default=2, ge=0)
    extra_body: dict[str, Any] | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    @field_validator("base_url")
    @classmethod
    def base_url_must_be_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an http or https URL")
        return value

    @field_validator("api_key_env")
    @classmethod
    def api_key_env_must_be_valid_env_name(cls, value: str) -> str:
        if _API_KEY_ENV.fullmatch(value) is None:
            raise ValueError("api_key_env must be a valid environment variable name")
        return value

    def to_metadata(self) -> LLMProviderMetadata:
        """提取与 SDK 无关的审计配置，不读取密钥值。

        Returns:
            当前模型和连接配置的快照。
        """
        return LLMProviderMetadata(
            adapter=self.adapter,
            base_url=self.base_url,
            api_key_env=self.api_key_env,
            model=self.model,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            extra_body=self.extra_body,
        )


class ProvidersConfig(BaseModel):
    """已校验的 providers.json 内容。"""

    active_provider: str = Field(min_length=1)
    providers: dict[str, ProviderConfig] = Field(min_length=1)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def active_provider_exists(self) -> ProvidersConfig:
        if self.active_provider not in self.providers:
            raise ValueError(f"active_provider '{self.active_provider}' is not configured")
        return self

    @property
    def active(self) -> ProviderConfig:
        return self.providers[self.active_provider]


@lru_cache
def get_settings() -> Settings:
    load_environment()
    return Settings()


def load_environment() -> None:
    """Load every key from .env into process environment without overriding real env vars."""
    load_dotenv(dotenv_path=_ENV_FILE, override=False)


def load_providers_config(path: Path) -> ProvidersConfig:
    """加载 provider 配置，并用适合启动阶段的错误快速失败。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return ProvidersConfig.model_validate(raw)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Provider config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Provider config file is not valid JSON: {path}") from exc
    except ValidationError as exc:
        raise RuntimeError(f"Provider config file is invalid: {path}") from exc
