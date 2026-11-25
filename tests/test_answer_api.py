import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from ocs_llm_answerer.answer.models import AnswerPayload
from ocs_llm_answerer.core.config import Settings
from ocs_llm_answerer.llm.errors import ProviderUnavailableError
from ocs_llm_answerer.llm.models import (
    LLMCallResult,
    LLMProviderMetadata,
    LLMUsage,
    TextPart,
    UserInput,
)
from ocs_llm_answerer.main import create_app


class _FakeSession:
    """单请求会话替身，把调用委托回 FakeLLMClient。"""

    def __init__(self, client: FakeLLMClient, instructions: str) -> None:
        self._client = client
        self.instructions = instructions
        self.inputs: list[UserInput] = []

    async def send_structured(self, user_input, response_model):
        self.inputs.append(user_input)
        return await self._client._handle_call(user_input, response_model)


class FakeLLMClient:
    """用于测试路由、鉴权、缓存和审计的确定性 LLMClient。"""

    name = "fake"
    model = "fake-model"
    request_metadata = LLMProviderMetadata(
        adapter="fake_adapter",
        base_url="https://fake.local/v1",
        api_key_env="FAKE_API_KEY",
        model=model,
        timeout_seconds=30,
        max_retries=2,
        extra_body={"trace": True},
    )

    def __init__(self) -> None:
        self.calls = 0
        self.received_inputs: list[UserInput] = []

    def create_session(self, *, instructions: str) -> _FakeSession:
        return _FakeSession(self, instructions)

    async def aclose(self) -> None:
        """内存替身不持有需要释放的资源。"""

    async def _handle_call(self, user_input: UserInput, response_model):
        self.calls += 1
        self.received_inputs.append(user_input)
        first_text = next(
            (part.text for part in user_input.parts if isinstance(part, TextPart)), ""
        )
        answer = AnswerPayload(answers=["A"], explanation=f"解析：{first_text}", confidence=0.9)
        raw_response = {
            "id": "913943893d5c45018725f08b6598ec83",
            "object": "response",
            "status": "completed",
            "model": self.model,
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": answer.model_dump_json(),
                        }
                    ],
                }
            ],
            "usage": {
                "input_tokens": 265,
                "input_tokens_details": {
                    "cached_tokens": 192,
                },
                "output_tokens": 654,
                "total_tokens": 919,
            },
        }
        return LLMCallResult(
            parsed=answer,
            response_body_raw=json.dumps(raw_response, ensure_ascii=False, separators=(",", ":")),
            http_status=200,
            usage=LLMUsage(
                input_tokens=265,
                output_tokens=654,
                total_tokens=919,
                cached_tokens=192,
            ),
        )


class FailingLLMClient(FakeLLMClient):
    async def _handle_call(self, user_input: UserInput, response_model):
        """模拟上游限流。"""
        self.calls += 1
        self.received_inputs.append(user_input)
        raise ProviderUnavailableError("provider exploded", status_code=429)


def test_answer_requires_api_key_when_configured(tmp_path):
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "single"})

    assert response.status_code == 401


def test_answer_uses_provider_then_cache(tmp_path):
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=llm_client,
    )

    payload = {
        "title": "1+1=?",
        "type": "single",
        "options": ["A. 2", "B. 3"],
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json=payload, headers={"X-API-Key": "secret"})
        second = client.post("/api/v1/answer", json=payload, headers={"X-API-Key": "secret"})

    assert first.status_code == 200
    assert first.json()["answer"] == "A"
    assert first.json()["confidence"] == 0.9
    assert first.json()["cache_hit"] is False
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert llm_client.calls == 1


def test_answer_treats_blank_string_options_as_no_options_for_cache_key(tmp_path):
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/answer",
            json={"title": "1+1=?", "type": "single", "options": "   \n\t  "},
        )
        second = client.post(
            "/api/v1/answer",
            json={"title": " 1+1=? ", "type": "single", "options": []},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert llm_client.calls == 1


def test_answer_records_successful_llm_request(tmp_path):
    llm_client = FakeLLMClient()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=database_path),
        llm_client=llm_client,
    )

    payload = {
        "title": "1+1=?",
        "type": "single",
        "options": ["A. 2", "B. 3"],
    }

    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json=payload, headers={"X-API-Key": "secret"})
        second = client.post("/api/v1/answer", json=payload, headers={"X-API-Key": "secret"})

    assert first.status_code == 200
    assert second.status_code == 200

    with sqlite3.connect(database_path) as db:
        rows = db.execute(
            """
            SELECT id, request_started_at_ns, request_completed_at_ns, question_hash,
                   adapter, base_url, api_key_env, model,
                   timeout_seconds, max_retries, extra_body,
                   request_status, response_body_raw,
                   http_status, input_tokens, output_tokens, total_tokens, cached_tokens
            FROM llm_requests
            """
        ).fetchall()
        cache_row = db.execute(
            """
            SELECT question_hash, llm_request_id
            FROM answer_cache
            """
        ).fetchone()

    assert len(rows) == 1
    (
        llm_request_id,
        request_started_at_ns,
        request_completed_at_ns,
        question_hash,
        adapter,
        base_url,
        api_key_env,
        model,
        timeout_seconds,
        max_retries,
        extra_body,
        status,
        raw_response,
        http_status,
        *usage,
    ) = rows[0]
    assert (
        adapter,
        base_url,
        api_key_env,
        model,
        timeout_seconds,
        max_retries,
        extra_body,
        status,
    ) == (
        "fake_adapter",
        "https://fake.local/v1",
        "FAKE_API_KEY",
        "fake-model",
        30,
        2,
        '{"trace":true}',
        "SUCCESS",
    )
    assert isinstance(request_started_at_ns, int)
    assert isinstance(request_completed_at_ns, int)
    assert request_completed_at_ns >= request_started_at_ns
    assert isinstance(llm_request_id, int)
    assert llm_request_id > 0
    assert http_status == 200
    assert json.loads(raw_response)["usage"] == {
        "input_tokens": 265,
        "input_tokens_details": {"cached_tokens": 192},
        "output_tokens": 654,
        "total_tokens": 919,
    }
    assert usage == [265, 654, 919, 192]
    assert cache_row == (question_hash, llm_request_id)


def test_answer_records_failed_llm_request(tmp_path):
    llm_client = FailingLLMClient()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=database_path),
        llm_client=llm_client,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "single"})

    assert response.status_code == 503

    with sqlite3.connect(database_path) as db:
        row = db.execute(
            """
            SELECT request_status, error_type, error_message, http_status,
                   response_body_raw, request_started_at_ns, request_completed_at_ns
            FROM llm_requests
            """
        ).fetchone()

    assert row[:5] == ("FAILURE", "ProviderUnavailableError", "provider exploded", 429, None)
    assert isinstance(row[5], int)
    assert isinstance(row[6], int)
    assert row[6] >= row[5]


def test_answer_accepts_ocs_text_plain_json_body(tmp_path):
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=llm_client,
    )

    body = (
        '{"options":"《神圣家族》\\n《德意志意识形态》\\n《共产党宣言》\\n《德法年鉴》\\n《资本论》",'
        '"title":"马克思和恩格斯合写的(        ) 首次系统阐述了历史唯物主义的基本观点。",'
        '"type":"single"}'
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/answer",
            content=body.encode("utf-8"),
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "X-API-Key": "llmapp",
            },
        )

    assert response.status_code == 200
    assert response.json()["answer"] == "A"
    assert llm_client.calls == 1


def test_answer_rejects_invalid_json_body(tmp_path):
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/answer",
            content=b'{"title":',
            headers={"Content-Type": "text/plain;charset=UTF-8"},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "Request body must be valid JSON"


def test_answer_records_real_text_plain_payload(tmp_path):
    llm_client = FakeLLMClient()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=database_path),
        llm_client=llm_client,
    )

    body = '{"extra":"ignored","options":"A. 2\\nB. 3","title":"1+1=?","type":"single"}'

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/answer",
            content=body.encode("utf-8"),
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "X-API-Key": "llmapp",
            },
        )

    assert response.status_code == 200

    with sqlite3.connect(database_path) as db:
        question_raw_json = db.execute("SELECT question_raw_json FROM questions").fetchone()[0]
        request_raw_json = db.execute("SELECT question_raw_json FROM llm_requests").fetchone()[0]

    assert question_raw_json == body
    assert request_raw_json == body


def test_answer_updates_question_raw_payload_on_cache_hit(tmp_path):
    llm_client = FakeLLMClient()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=database_path),
        llm_client=llm_client,
    )

    first_body = '{"title":"1+1=?","type":"single","options":"A. 2\\nB. 3"}'
    second_body = '{"extra":"cache-hit","options":"A. 2\\nB. 3","title":"1+1=?","type":"single"}'

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/answer",
            content=first_body.encode("utf-8"),
            headers={"Content-Type": "text/plain;charset=UTF-8", "X-API-Key": "llmapp"},
        )
        second = client.post(
            "/api/v1/answer",
            content=second_body.encode("utf-8"),
            headers={"Content-Type": "text/plain;charset=UTF-8", "X-API-Key": "llmapp"},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert llm_client.calls == 1

    with sqlite3.connect(database_path) as db:
        question_raw_json = db.execute("SELECT question_raw_json FROM questions").fetchone()[0]
        llm_request_count = db.execute("SELECT COUNT(*) FROM llm_requests").fetchone()[0]

    assert question_raw_json == second_body
    assert llm_request_count == 1


def test_answer_cors_allows_ocs_origin_and_private_network_preflight(tmp_path):
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        preflight = client.options(
            "/api/v1/answer",
            headers={
                "Origin": "https://mooc1-api.chaoxing.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-api-key",
                "Access-Control-Request-Private-Network": "true",
            },
        )
        post = client.post(
            "/api/v1/answer",
            json={"title": "1+1=?", "type": "single"},
            headers={"Origin": "https://mooc1-api.chaoxing.com", "X-API-Key": "llmapp"},
        )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert preflight.headers["access-control-allow-private-network"] == "true"
    assert post.status_code == 200
    assert post.headers["access-control-allow-origin"] == "*"


def test_answer_accepts_blank_question_type(tmp_path):
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": ""})

    assert response.status_code == 200
    assert llm_client.calls == 1


def test_answer_rejects_unknown_question_type(tmp_path):
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "essay"})

    assert response.status_code == 422


def test_health_endpoint(tmp_path):
    app = create_app(
        settings=Settings(app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/", "/api/v1/answer"])
def test_head_endpoints_support_ocs_status_probe(tmp_path, path):
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.head(path, params={"t": "1780724765342"})

    assert response.status_code == 200
    assert response.content == b""


def test_ocs_answerer_config_is_array_and_maps_response(tmp_path):
    app = create_app(
        settings=Settings(
            app_api_key="secret",
            app_database_path=tmp_path / "cache.sqlite3",
            app_image_enabled=True,
        ),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.get("/ocs-answerer.json")

    assert response.status_code == 200
    config = response.json()
    assert isinstance(config, list)
    assert config[0]["url"] == "http://testserver/api/v1/answer"
    assert config[0]["headers"] == {"Content-Type": "application/json"}
    assert config[0]["data"] == {
        "title": "${title}",
        "type": "${type}",
        "options": "${options}",
        "images": {"handler": "return env => env.images"},
        "suggestion_title": {"handler": "return env => env.suggestion_title"},
        "suggestion_options": {"handler": "return env => env.suggestion_options"},
    }
    assert "type" not in config[0]
    assert config[0]["handler"] == (
        "return (res)=> res.code === 1 ? [res.question, res.answer] : undefined"
    )


def test_ocs_answerer_config_can_use_gm_xmlhttp_request(tmp_path):
    app = create_app(
        settings=Settings(
            app_api_key="secret",
            app_database_path=tmp_path / "cache.sqlite3",
            app_ocs_answerer_request_type="GM_xmlhttpRequest",
        ),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        response = client.get("/ocs-answerer.json")

    assert response.status_code == 200
    config = response.json()
    assert config[0]["type"] == "GM_xmlhttpRequest"


def test_public_subscription_cannot_bypass_answer_authentication(tmp_path):
    """匿名订阅不泄露密钥，使用模板请求头仍然不能访问受保护的答题接口。"""
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key="private-test-key", app_database_path=tmp_path / "cache.db"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        subscription = client.get(
            "/ocs-answerer.json", headers={"Origin": "https://untrusted.example"}
        )
        template_headers = subscription.json()[0]["headers"]
        denied = client.post("/api/v1/answer", json={"title": "1+1=?"}, headers=template_headers)
        allowed = client.post(
            "/api/v1/answer",
            json={"title": "1+1=?"},
            headers={"X-API-Key": "private-test-key"},
        )
        authenticated_subscription = client.get(
            "/ocs-answerer.json", headers={"X-API-Key": "private-test-key"}
        )

    assert subscription.status_code == 200
    assert "private-test-key" not in subscription.text
    assert "private-test-key" not in authenticated_subscription.text
    assert "X-API-Key" not in template_headers
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert llm_client.calls == 1


def test_public_subscription_works_without_configured_api_key(tmp_path):
    """未启用鉴权时，公开订阅中的请求头可以直接用于答题。"""
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        llm_client=FakeLLMClient(),
    )

    with TestClient(app) as client:
        config = client.get("/ocs-answerer.json").json()[0]
        response = client.post("/api/v1/answer", json={"title": "1+1=?"}, headers=config["headers"])

    assert response.status_code == 200


@pytest.mark.parametrize("title", ["", " ", "\t\r\n", "\u2003"])
def test_answer_rejects_blank_title_without_calling_provider(tmp_path, title):
    """纯空白题干返回可序列化的 422 JSON，并且不消耗模型调用。"""
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": title})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["title"]
    assert llm_client.calls == 0


def test_answer_does_not_reuse_cache_across_semantic_whitespace(tmp_path):
    """题干字符串内的不同空格触发独立答题，外围空白仍可复用缓存。"""
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        llm_client=llm_client,
    )

    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json={"title": 'len("a  b") 的结果'})
        second = client.post("/api/v1/answer", json={"title": 'len("a b") 的结果'})
        repeated = client.post("/api/v1/answer", json={"title": '  len("a  b") 的结果  '})

    assert not first.json()["cache_hit"]
    assert not second.json()["cache_hit"]
    assert repeated.json()["cache_hit"]
    assert llm_client.calls == 2


def test_llm_client_receives_normalized_prompt_without_audit_payload(tmp_path):
    """模型输入匹配缓存使用的题目，审计仍然保存未经修改的请求体。"""
    llm_client = FakeLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        llm_client=llm_client,
    )
    raw = json.dumps(
        {"title": "  代码：\r\n    x = 'a  b'  ", "options": " A. 1\r\nB. 2 ", "extra": "secret"}
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", content=raw)

    assert response.status_code == 200
    assert response.json()["question"] == "代码：\n    x = 'a  b'"
    sent_text = llm_client.received_inputs[0].parts[0].text
    assert "题干：代码：\n    x = 'a  b'" in sent_text
    assert "<options>\nA. 1\nB. 2\n</options>" in sent_text
    assert "secret" not in sent_text
    with sqlite3.connect(tmp_path / "cache.db") as db:
        assert db.execute("SELECT question_raw_json FROM llm_requests").fetchone()[0] == raw


@pytest.mark.parametrize(
    ("question_type", "answers"),
    [
        ("single", ["A", "B"]),
        ("judgement", ["maybe"]),
        ("multiple", ["ABC"]),
        ("completion", ["第一空", "", "第三空"]),
    ],
)
def test_invalid_model_answer_is_logged_without_entering_cache(tmp_path, question_type, answers):
    """连续违反题型规则时逐次重试并审计，失败保留响应和用量且不写入缓存。"""
    raw = json.dumps({"answers": answers}, ensure_ascii=False)

    class InvalidLLMClient(FakeLLMClient):
        async def _handle_call(self, user_input, response_model):
            result = await super()._handle_call(user_input, response_model)
            result.parsed.answers = answers
            result.response_body_raw = raw
            return result

    database_path = tmp_path / "cache.db"
    llm_client = InvalidLLMClient()
    app = create_app(
        settings=Settings(
            app_api_key=None,
            app_database_path=database_path,
            app_max_answer_attempts=3,
        ),
        llm_client=llm_client,
    )
    payload = {"title": "题目", "type": question_type, "options": ["A. 苹果", "B. 香蕉"]}

    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json=payload)
        second = client.post("/api/v1/answer", json=payload)

    assert first.status_code == second.status_code == 502
    assert first.json()["detail"].startswith("Invalid model answer:")
    assert llm_client.calls == 6
    with sqlite3.connect(database_path) as db:
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
        rows = db.execute(
            "SELECT request_status, error_type, response_body_raw, http_status, total_tokens "
            "FROM llm_requests"
        ).fetchall()
    assert rows == [("FAILURE", "InvalidAnswerError", raw, 200, 919)] * 6


def test_answer_recovers_on_correction_retry_and_caches_winning_request_id(tmp_path):
    """首次输出不合法、第二次纠正合法时返回 200，流水保留失败与成功，缓存指向成功调用。"""

    class RecoveringLLMClient(FakeLLMClient):
        async def _handle_call(self, user_input, response_model):
            result = await super()._handle_call(user_input, response_model)
            if self.calls == 1:
                result.parsed.answers = ["A", "B"]
            else:
                result.parsed.answers = ["B"]
            return result

    database_path = tmp_path / "cache.db"
    llm_client = RecoveringLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=database_path),
        llm_client=llm_client,
    )
    payload = {"title": "单选", "type": "single", "options": ["A. 苹果", "B. 香蕉"]}

    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json=payload)
        second = client.post("/api/v1/answer", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"] == "B"
    assert not first.json()["cache_hit"]
    assert second.json()["cache_hit"]
    assert llm_client.calls == 2
    with sqlite3.connect(database_path) as db:
        rows = db.execute("SELECT id, request_status, error_type FROM llm_requests").fetchall()
        cached_request_id = db.execute("SELECT llm_request_id FROM answer_cache").fetchone()[0]
    assert [(r[1], r[2]) for r in rows] == [("FAILURE", "InvalidAnswerError"), ("SUCCESS", None)]
    assert cached_request_id == rows[1][0]


def test_service_formats_structured_answers_before_caching(tmp_path):
    """OCS 格式化由服务完成，LLMClient 可直接返回未排序且重复的答案项。"""

    class MultipleLLMClient(FakeLLMClient):
        async def _handle_call(self, user_input, response_model):
            result = await super()._handle_call(user_input, response_model)
            result.parsed.answers = ["B", "A", "A"]
            return result

    llm_client = MultipleLLMClient()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        llm_client=llm_client,
    )
    payload = {"title": "多选", "type": "multiple", "options": ["A. 苹果", "B. 香蕉"]}
    with TestClient(app) as client:
        first = client.post("/api/v1/answer", json=payload)
        second = client.post("/api/v1/answer", json=payload)

    assert first.status_code == second.status_code == 200
    assert first.json()["answer"] == second.json()["answer"] == "A#B"
    assert second.json()["cache_hit"]
    assert llm_client.calls == 1
