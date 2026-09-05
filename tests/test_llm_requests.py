import asyncio
import sqlite3

import httpx
import pytest
from openai import AsyncOpenAI

from ocs_llm_answerer.answer.normalization import build_normalized_question
from ocs_llm_answerer.answer.service import AnswerService
from ocs_llm_answerer.core.config import ProviderConfig
from ocs_llm_answerer.core.models import AnswerRequest, LLMAnswer, LLMCallResult
from ocs_llm_answerer.database.cache import AnswerCache, init_sqlite
from ocs_llm_answerer.database.llm_requests import LLMRequestLog
from ocs_llm_answerer.llm.openai_responses import OpenAIResponsesProvider


@pytest.mark.parametrize(
    ("status_code", "raw_body"),
    [
        (200, "<html>unexpected gateway page</html>"),
        (200, '{"output":'),
        (502, "<html>bad gateway</html>"),
        (429, "请稍后重试"),
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
        service = AnswerService(AnswerCache(database_path), provider, LLMRequestLog(database_path))
        try:
            with pytest.raises(RuntimeError) as error:
                await service.answer(AnswerRequest(title="失败响应测试", type="single"))
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
    assert rows == [("FAILURE", raw_body, status_code, "_RawResponseError")]


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
        await LLMRequestLog(database_path).record_success(
            build_normalized_question(AnswerRequest(title="测试")),
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
