import asyncio
import sqlite3
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI

from ocs_llm_answerer.answer.models import OCSQuestionType, Question
from ocs_llm_answerer.answer.normalization import build_normalized_question
from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.database.cache import AnswerCacheRepository
from ocs_llm_answerer.database.connection import init_sqlite
from ocs_llm_answerer.database.llm_requests import LLMRequestRepository
from ocs_llm_answerer.llm.errors import ProviderTimeoutError, ProviderUnavailableError
from ocs_llm_answerer.llm.models import LLMAnswer, LLMCallResult
from ocs_llm_answerer.llm.openai_responses import OpenAIResponsesProvider


@pytest.mark.parametrize(
    ("status_code", "raw_body"),
    [
        (200, "<html>unexpected gateway page</html>"),
        (200, '{"output":'),
        (502, "<html>bad gateway</html>"),
        (429, "请稍后重试"),
        (401, "invalid upstream credential"),
        (503, "temporarily unavailable"),
        (502, ""),
    ],
)
def test_real_sdk_errors_keep_raw_response_in_audit(tmp_path, status_code, raw_body):
    """通过模拟 HTTP 传输验证 SDK、服务和 SQLite 全链路保留非 JSON 失败现场。"""
    database_path = tmp_path / "audit.db"
    config = ProviderConfig(
        adapter="openai_responses",
        base_url="https://test.invalid/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
        max_retries=0,
    )

    def respond(request):
        """返回固定 HTTP 响应，禁止测试访问真实模型服务。"""
        return httpx.Response(status_code, text=raw_body, headers={"content-type": "text/plain"})

    async def exercise():
        """使用真实 SDK 调用路径，将异常写入临时数据库。"""
        await init_sqlite(database_path)
        provider = OpenAIResponsesProvider("fake", config, "fake-key")
        await provider._client.close()
        provider._client = AsyncOpenAI(
            api_key="fake-key",
            base_url=config.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        service = AnswerService(
            cache_repository=AnswerCacheRepository(database_path),
            provider=provider,
            request_repository=LLMRequestRepository(database_path),
        )
        try:
            with pytest.raises(RuntimeError) as error:
                await service.answer(
                    Question(title="失败响应测试", question_type=OCSQuestionType.SINGLE)
                )
            assert error.value.raw_body == raw_body
            assert error.value.status_code == status_code
            assert error.value.__cause__ is not None
        finally:
            await provider._client.close()

    asyncio.run(exercise())

    with sqlite3.connect(database_path) as db:
        rows = db.execute(
            "SELECT request_status, response_body_raw, http_status, error_type FROM llm_requests"
        ).fetchall()
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
    error_type = (
        "ProviderUnavailableError" if status_code in {429, 503} else "ProviderResponseError"
    )
    assert rows == [("FAILURE", raw_body, status_code, error_type)]


def test_raw_text_is_allowed_while_application_json_is_validated(tmp_path):
    """原文可以自由存储，应用生成的 JSON 字段仍由数据库保证合法性。"""
    database_path = tmp_path / "audit.db"
    metadata = ProviderConfig(
        adapter="openai_responses",
        base_url="https://test.invalid/v1",
        api_key_env="FAKE_API_KEY",
        model="fake-model",
    ).to_metadata()

    async def exercise():
        """写入带有非 JSON 原文的成功调用记录。"""
        await init_sqlite(database_path)
        await LLMRequestRepository(database_path).record_success(
            build_normalized_question(Question(title="测试")),
            metadata,
            LLMCallResult(
                answer=LLMAnswer(answers=["A"], confidence=0.9),
                response_body_raw="原始文本\n第二行",
                http_status=200,
            ),
            seen_at_ns=1,
            request_started_at_ns=2,
            request_completed_at_ns=3,
            latency_ms=1,
        )

    asyncio.run(exercise())

    with sqlite3.connect(database_path) as db:
        assert db.execute("SELECT response_body_raw FROM llm_requests").fetchone()[0] == (
            "原始文本\n第二行"
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE llm_requests SET extra_body = ?", ("not-json",))
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("UPDATE llm_requests SET question_raw_json = ?", ("not-json",))


@pytest.mark.parametrize(
    ("transport_error", "provider_error"),
    [(httpx.ReadTimeout, ProviderTimeoutError), (httpx.ConnectError, ProviderUnavailableError)],
)
def test_sdk_transport_errors_use_provider_contract(
    tmp_path: Path,
    transport_error: type[httpx.TransportError],
    provider_error: type[Exception],
) -> None:
    """真实 SDK 在模拟传输异常后返回稳定错误类型，并审计无响应的调用。"""
    path = tmp_path / "cache.db"
    config = ProviderConfig.model_validate(
        {
            "adapter": "openai_responses",
            "base_url": "https://test.invalid/v1",
            "api_key_env": "FAKE_KEY",
            "model": "fake-model",
            "max_retries": 0,
        }
    )

    def respond(request: httpx.Request) -> httpx.Response:
        """模拟网络故障，禁止发出真实请求。

        Args:
            request: SDK 构造的请求。

        Raises:
            httpx.TransportError: 参数化的网络故障。
        """
        raise transport_error("injected transport failure", request=request)

    async def exercise() -> None:
        """运行 SDK、服务和临时数据库调用链，并始终关闭客户端。"""
        await init_sqlite(path)
        provider = OpenAIResponsesProvider("fake", config, "fake-key")
        await provider.aclose()
        provider._client = AsyncOpenAI(
            api_key="fake-key",
            base_url=config.base_url,
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )
        try:
            service = AnswerService(
                AnswerCacheRepository(path), provider, LLMRequestRepository(path)
            )
            with pytest.raises(provider_error) as raised:
                await service.answer(Question(title="题目"))
            assert raised.value.__cause__ is not None
        finally:
            await provider.aclose()

    asyncio.run(exercise())
    with sqlite3.connect(path) as db:
        rows = db.execute(
            "SELECT error_type, http_status, response_body_raw FROM llm_requests"
        ).fetchall()
    assert rows == [(provider_error.__name__, None, None)]
