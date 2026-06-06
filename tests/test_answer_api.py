import json
import sqlite3
from hashlib import sha256

from fastapi.testclient import TestClient

from ocs_llm_answerer.core.config import Settings
from ocs_llm_answerer.core.models import (
    AnswerRequest,
    LLMAnswer,
    LLMCallResult,
    LLMProviderMetadata,
    LLMUsage,
)
from ocs_llm_answerer.main import create_app


class FakeProvider:
    """用于测试路由、鉴权和缓存的确定性 provider。"""

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

    async def answer(self, request: AnswerRequest) -> LLMCallResult:
        self.calls += 1
        answer = LLMAnswer(answer="A", explanation=f"解析：{request.title}", confidence=0.9)
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
            answer=answer,
            response_body_raw=json.dumps(raw_response, ensure_ascii=False, separators=(",", ":")),
            http_status=200,
            usage=LLMUsage(
                input_tokens=265,
                output_tokens=654,
                total_tokens=919,
                cached_tokens=192,
            ),
        )


class ProviderHTTPError(RuntimeError):
    status_code = 429


class FailingProvider(FakeProvider):
    async def answer(self, request: AnswerRequest) -> LLMCallResult:
        self.calls += 1
        raise ProviderHTTPError("provider exploded")


def test_answer_requires_api_key_when_configured(tmp_path):
    provider = FakeProvider()
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        provider=provider,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "single"})

    assert response.status_code == 401


def test_answer_uses_provider_then_cache(tmp_path):
    provider = FakeProvider()
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        provider=provider,
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
    assert provider.calls == 1


def test_answer_treats_blank_string_options_as_no_options_for_cache_key(tmp_path):
    provider = FakeProvider()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        provider=provider,
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
    assert provider.calls == 1


def test_answer_records_successful_llm_request(tmp_path):
    provider = FakeProvider()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=database_path),
        provider=provider,
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
            SELECT call_hash, request_started_at_ns, request_completed_at_ns, question_hash,
                   adapter, base_url, api_key_env, model,
                   timeout_seconds, max_retries, extra_body,
                   request_status, response_body_raw,
                   http_status, input_tokens, output_tokens, total_tokens, cached_tokens
            FROM llm_requests
            """
        ).fetchall()
        cache_row = db.execute(
            """
            SELECT question_hash, call_hash
            FROM answer_cache
            """
        ).fetchone()

    assert len(rows) == 1
    (
        call_hash,
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
    expected_hash_payload = {
        "question_hash": question_hash,
        "request_started_at_ns": request_started_at_ns,
        "adapter": adapter,
        "base_url": base_url,
        "api_key_env": api_key_env,
        "model": model,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "extra_body": json.loads(extra_body),
    }
    expected_hash = sha256(
        json.dumps(
            expected_hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert call_hash == expected_hash
    assert http_status == 200
    assert json.loads(raw_response)["usage"] == {
        "input_tokens": 265,
        "input_tokens_details": {"cached_tokens": 192},
        "output_tokens": 654,
        "total_tokens": 919,
    }
    assert usage == [265, 654, 919, 192]
    assert cache_row == (question_hash, call_hash)


def test_answer_records_failed_llm_request(tmp_path):
    provider = FailingProvider()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=database_path),
        provider=provider,
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "single"})

    assert response.status_code == 500

    with sqlite3.connect(database_path) as db:
        row = db.execute(
            """
            SELECT request_status, error_type, error_message, http_status,
                   response_body_raw, request_started_at_ns, request_completed_at_ns
            FROM llm_requests
            """
        ).fetchone()

    assert row[:5] == ("FAILURE", "ProviderHTTPError", "provider exploded", 429, None)
    assert isinstance(row[5], int)
    assert isinstance(row[6], int)
    assert row[6] >= row[5]


def test_answer_accepts_ocs_text_plain_json_body(tmp_path):
    provider = FakeProvider()
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=tmp_path / "cache.sqlite3"),
        provider=provider,
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
    assert provider.calls == 1


def test_answer_rejects_invalid_json_body(tmp_path):
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
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
    provider = FakeProvider()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=database_path),
        provider=provider,
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
        question_raw_json = db.execute(
            """
            SELECT question_raw_json
            FROM questions
            """
        ).fetchone()[0]
        request_raw_json = db.execute(
            """
            SELECT question_raw_json
            FROM llm_requests
            """
        ).fetchone()[0]

    assert question_raw_json == body
    assert request_raw_json == body


def test_answer_updates_question_raw_payload_on_cache_hit(tmp_path):
    provider = FakeProvider()
    database_path = tmp_path / "cache.sqlite3"
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=database_path),
        provider=provider,
    )

    first_body = '{"title":"1+1=?","type":"single","options":"A. 2\\nB. 3"}'
    second_body = '{"extra":"cache-hit","options":"A. 2\\nB. 3","title":"1+1=?","type":"single"}'

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/answer",
            content=first_body.encode("utf-8"),
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "X-API-Key": "llmapp",
            },
        )
        second = client.post(
            "/api/v1/answer",
            content=second_body.encode("utf-8"),
            headers={
                "Content-Type": "text/plain;charset=UTF-8",
                "X-API-Key": "llmapp",
            },
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["cache_hit"] is True
    assert provider.calls == 1

    with sqlite3.connect(database_path) as db:
        question_raw_json = db.execute(
            """
            SELECT question_raw_json
            FROM questions
            """
        ).fetchone()[0]
        llm_request_count = db.execute(
            """
            SELECT COUNT(*)
            FROM llm_requests
            """
        ).fetchone()[0]

    assert question_raw_json == second_body
    assert llm_request_count == 1


def test_answer_cors_allows_ocs_origin_and_private_network_preflight(tmp_path):
    app = create_app(
        settings=Settings(app_api_key="llmapp", app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
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
            headers={
                "Origin": "https://mooc1-api.chaoxing.com",
                "X-API-Key": "llmapp",
            },
        )

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == "*"
    assert preflight.headers["access-control-allow-private-network"] == "true"
    assert post.status_code == 200
    assert post.headers["access-control-allow-origin"] == "*"


def test_answer_accepts_blank_question_type(tmp_path):
    provider = FakeProvider()
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        provider=provider,
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": ""})

    assert response.status_code == 200
    assert provider.calls == 1


def test_answer_rejects_unknown_question_type(tmp_path):
    app = create_app(
        settings=Settings(app_api_key=None, app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "1+1=?", "type": "essay"})

    assert response.status_code == 422


def test_health_endpoint(tmp_path):
    app = create_app(
        settings=Settings(app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_head_endpoint_supports_ocs_status_probe(tmp_path):
    app = create_app(
        settings=Settings(app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
    )

    with TestClient(app) as client:
        response = client.head("/?t=1780724765342")

    assert response.status_code == 200


def test_ocs_answerer_config_is_array_and_maps_response(tmp_path):
    app = create_app(
        settings=Settings(app_api_key="secret", app_database_path=tmp_path / "cache.sqlite3"),
        provider=FakeProvider(),
    )

    with TestClient(app) as client:
        response = client.get("/ocs-answerer.json")

    assert response.status_code == 200
    config = response.json()
    assert isinstance(config, list)
    assert config[0]["url"] == "http://testserver/api/v1/answer"
    assert config[0]["headers"] == {
        "Content-Type": "application/json",
        "X-API-Key": "secret",
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
        provider=FakeProvider(),
    )

    with TestClient(app) as client:
        response = client.get("/ocs-answerer.json")

    assert response.status_code == 200
    config = response.json()
    assert config[0]["type"] == "GM_xmlhttpRequest"
