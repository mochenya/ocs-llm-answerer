"""验证 HTTP 契约及真实 SQLite 上的审计、缓存故障语义。"""

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from ocs_llm_answerer.answer.errors import RepositoryError
from ocs_llm_answerer.api.schemas import AnswerRequest
from ocs_llm_answerer.core.config import Settings
from ocs_llm_answerer.database.cache import AnswerCacheRepository
from ocs_llm_answerer.llm.errors import (
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from ocs_llm_answerer.main import create_app
from tests.fakes import StubProvider


def assert_local_refs_resolve(value: object, document: dict[str, Any]) -> None:
    """遍历 OpenAPI 文档并断言每个本地 JSON Pointer 都存在。

    Args:
        value: 当前待遍历的节点。
        document: 用于解析本地引用的完整文档。
    """
    if isinstance(value, dict):
        reference = value.get("$ref")
        if isinstance(reference, str) and reference.startswith("#/"):
            target: Any = document
            for part in reference[2:].split("/"):
                target = target[part.replace("~1", "/").replace("~0", "~")]
        for child in value.values():
            assert_local_refs_resolve(child, document)
    elif isinstance(value, list):
        for child in value:
            assert_local_refs_resolve(child, document)


def test_openapi_documents_request_content_types_fields_and_errors(tmp_path: Path) -> None:
    """手动解析请求仍具有完整的文档契约，枚举引用指向组件而非悬空 $defs。"""
    app = create_app(
        Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"), StubProvider()
    )
    with TestClient(app) as client:
        response = client.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    operation = schema["paths"]["/api/v1/answer"]["post"]
    assert operation["requestBody"]["required"]
    content = operation["requestBody"]["content"]
    assert set(content) == {"application/json", "text/plain"}
    assert content["application/json"]["schema"] == {"$ref": "#/components/schemas/AnswerRequest"}
    assert content["text/plain"]["schema"]["type"] == "string"
    AnswerRequest.model_validate_json(content["text/plain"]["example"])
    request_schema = schema["components"]["schemas"]["AnswerRequest"]
    assert set(request_schema["properties"]) == {"title", "type", "options"}
    assert request_schema["required"] == ["title"]
    assert {"type": "string", "const": ""} in request_schema["properties"]["type"]["anyOf"]
    assert set(operation["responses"]) == {"200", "401", "422", "502", "503", "504"}
    assert_local_refs_resolve(schema, schema)
    assert app.openapi() is app.openapi()


@pytest.mark.parametrize("content_type", ["application/json", "text/plain;charset=UTF-8"])
def test_both_transports_preserve_exact_raw_audit(tmp_path: Path, content_type: str) -> None:
    """接口映射不改变成功字段、题目语义或原始载荷，包括被 schema 忽略的字段。"""
    path = tmp_path / "cache.db"
    provider = StubProvider()
    app = create_app(Settings(app_api_key=None, app_database_path=path), provider)
    raw = '{ "title":"  1+1=?  ", "type":"single", "options":" A. 2\\nB. 3 ", "extra":42 }'
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/answer", content=raw, headers={"Content-Type": content_type}
        )
    assert response.status_code == 200
    assert set(response.json()) == {
        "code",
        "question",
        "answer",
        "explanation",
        "confidence",
        "provider",
        "model",
        "cache_hit",
    }
    assert response.json()["code"] == 1
    assert provider.received[0].options == ["A. 2", "B. 3"]
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT question_raw_json FROM questions").fetchone()[0] == raw
        assert db.execute("SELECT question_raw_json FROM llm_requests").fetchone()[0] == raw


@pytest.mark.parametrize(
    ("body", "expected_detail"),
    [(b"\xff", "Request body must be UTF-8 encoded"), (b"{", "Request body must be valid JSON")],
)
def test_invalid_transport_is_rejected_before_provider(
    tmp_path: Path,
    body: bytes,
    expected_detail: str,
) -> None:
    """编码和 JSON 语法失败继续返回原有 detail 字符串。"""
    provider = StubProvider()
    app = create_app(Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"), provider)
    with TestClient(app) as client:
        response = client.post("/api/v1/answer", content=body)
    assert response.status_code == 422
    assert response.json() == {"detail": expected_detail}
    assert not provider.received


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (ProviderTimeoutError("private timeout"), 504, "Model provider timed out"),
        (
            ProviderUnavailableError("private 429", status_code=429),
            503,
            "Model provider is unavailable",
        ),
        (
            ProviderResponseError("private auth", raw_body="private body", status_code=401),
            502,
            "Invalid model provider response",
        ),
    ],
)
def test_provider_errors_map_without_leaking_upstream_data(
    tmp_path: Path,
    error: Exception,
    status: int,
    detail: str,
) -> None:
    """上游内部错误保留在审计，公开 HTTP 响应不暴露模型服务诊断文本。"""
    path = tmp_path / "cache.db"
    app = create_app(Settings(app_api_key=None, app_database_path=path), StubProvider(error=error))
    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "题目"})
    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert "private" not in response.text
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT error_message FROM llm_requests").fetchone()[0] == str(error)
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0


def test_unknown_programming_error_stays_500(tmp_path: Path) -> None:
    """未分类的编程异常不能被统一误报为模型服务故障。"""
    app = create_app(
        Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"),
        StubProvider(error=ValueError("unexpected internal bug")),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/v1/answer", json={"title": "题目"})
    assert response.status_code == 500
    assert "unexpected internal bug" not in response.text


@pytest.mark.parametrize("target", ["llm_requests", "answer_cache"])
def test_persistence_failure_preserves_independent_success_audit(
    tmp_path: Path, target: str
) -> None:
    """真实 SQLite 触发器制造写入故障，证明成功流水与缓存不是同一事务。"""
    path = tmp_path / "cache.db"
    provider = StubProvider()
    app = create_app(Settings(app_api_key=None, app_database_path=path), provider)
    raw = '{ "title":"题目", "trace":"exact original" }'
    with TestClient(app) as client:
        with sqlite3.connect(path) as db:
            db.execute(
                f"CREATE TRIGGER reject_insert BEFORE INSERT ON {target} "
                "BEGIN SELECT RAISE(ABORT, 'injected storage failure'); END"
            )
        response = client.post("/api/v1/answer", content=raw)
    assert response.status_code == 503
    assert response.json() == {"detail": "Answer storage is unavailable"}
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM answer_cache").fetchone()[0] == 0
        rows = db.execute("SELECT request_status, question_raw_json FROM llm_requests").fetchall()
    assert rows == ([("SUCCESS", raw)] if target == "answer_cache" else [])


def test_failed_audit_preserves_provider_http_mapping(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """失败审计无法写入时，原模型错误的 504 映射仍然有效。"""
    path = tmp_path / "cache.db"
    app = create_app(
        Settings(app_api_key=None, app_database_path=path),
        StubProvider(error=ProviderTimeoutError("timeout")),
    )
    with TestClient(app) as client:
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TRIGGER reject_audit BEFORE INSERT ON llm_requests "
                "BEGIN SELECT RAISE(ABORT, 'audit failure'); END"
            )
        response = client.post("/api/v1/answer", json={"title": "题目"})
    assert response.status_code == 504
    assert "Failed to persist failure audit" in caplog.text


def test_cache_read_failure_maps_to_503(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """缓存读取失败使用仓储错误契约，且不会继续调用模型。"""
    provider = StubProvider()
    app = create_app(Settings(app_api_key=None, app_database_path=tmp_path / "cache.db"), provider)
    monkeypatch.setattr(
        AnswerCacheRepository, "get", AsyncMock(side_effect=RepositoryError("failed"))
    )
    with TestClient(app) as client:
        response = client.post("/api/v1/answer", json={"title": "题目"})
    assert response.status_code == 503
    assert provider.received == []
