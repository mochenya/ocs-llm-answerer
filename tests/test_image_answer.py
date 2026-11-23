import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

from ocs_llm_answerer.api import dependencies, images
from ocs_llm_answerer.core.config import Settings
from ocs_llm_answerer.main import create_app
from tests.fakes import StubProvider

BLUE_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA"
    "FklEQVR4nGOImvefJMQwqmFUw/DVAADDH/cQsqxg4QAAAABJRU5ErkJggg=="
)
RED_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAA"
    "F0lEQVQoz2P8z0AaYCJR/aiGUQ1DSAMAQC4BHziaHckAAAAASUVORK5CYII="
)
BLUE_WEBP = (
    "data:image/webp;base64,UklGRjgAAABXRUJQVlA4ICwAAACQAQCdASoQABAAAgA0"
    "JaACdLoAA5gA/vmTb/+QH/+QH/+QH/8gP+IXeyAwAA=="
)


@pytest.fixture
def image_answer_api(tmp_path):
    path = tmp_path / "image-test.sqlite3"
    provider = StubProvider()
    settings = Settings(app_api_key=None, app_database_path=path, app_image_enabled=True)
    app = create_app(settings, provider)
    with TestClient(app) as client:
        yield client, provider, path


def test_image_answering_is_off_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("OCS_LLM_ANSWERER_IMAGE_ENABLED", raising=False)
    path = tmp_path / "text-only.sqlite3"
    provider = StubProvider()
    settings = Settings(_env_file=None, app_api_key=None, app_database_path=path)

    with TestClient(create_app(settings, provider)) as client:
        [template] = client.get("/ocs-answerer.json").json()
        disabled = client.post("/api/v1/answer", json={"title": "看图", "images": [BLUE_PNG]})
        text = client.post("/api/v1/answer", json={"title": "纯文字", "images": []})

    assert set(template["data"]) == {"title", "type", "options"}
    assert disabled.status_code == 422
    assert disabled.json() == {"detail": "Image answering is disabled"}
    assert text.status_code == 200
    assert [question.title for question in provider.received] == ["纯文字"]
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT question FROM questions").fetchall() == [("纯文字",)]


@pytest.mark.parametrize(
    ("payload", "title", "options"),
    [
        (
            {"suggestion_title": " [图片1] ", "type": "unknown"},
            "[图片1]",
            ["蓝色", "红色"],
        ),
        (
            {"suggestion_options": " [图片1] \n红色"},
            "看图作答",
            ["[图片1]", "红色"],
        ),
        (
            {"suggestion_title": "看 [图片1]", "suggestion_options": "蓝色\n [图片1] "},
            "看 [图片1]",
            ["蓝色", "[图片1]"],
        ),
        (
            {"images": [], "suggestion_title": "[图片9]", "suggestion_options": "[图片8]"},
            "看图作答",
            ["蓝色", "红色"],
        ),
    ],
)
def test_selects_effective_question_and_keeps_raw_audit(image_answer_api, payload, title, options):
    client, provider, path = image_answer_api
    raw = json.dumps(
        {"title": "看图作答", "options": "蓝色\n红色", "images": [BLUE_PNG], **payload},
        ensure_ascii=False,
    )
    response = client.post(
        "/api/v1/answer", content=raw, headers={"Content-Type": "text/plain;charset=UTF-8"}
    )
    assert response.status_code == 200
    assert response.json()["question"] == title
    request = provider.received[0]
    assert request.title == title
    assert request.options == options
    assert request.question_type is None
    assert [image.data_url for image in request.images] == (
        [] if payload.get("images") == [] else [BLUE_PNG]
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT question_raw_json FROM questions").fetchone() == (raw,)
        assert connection.execute("SELECT question_raw_json FROM llm_requests").fetchone() == (raw,)


def test_images_are_canonicalized_and_cached_by_ordered_content(image_answer_api):
    client, provider, _ = image_answer_api

    def answer(data_urls):
        response = client.post(
            "/api/v1/answer",
            json={
                "title": "看图",
                "suggestion_title": "比较 [图片1] 和 [图片2]",
                "images": data_urls,
            },
        )
        assert response.status_code == 200
        return response.json()["cache_hit"]

    assert not answer([BLUE_WEBP.replace("image/webp", "image/png"), RED_PNG])
    assert provider.received[0].images[0].data_url == BLUE_WEBP
    assert answer([BLUE_WEBP, RED_PNG])
    assert not answer([RED_PNG, BLUE_WEBP])
    assert not answer([BLUE_PNG, RED_PNG])
    assert len(provider.received) == 3


@pytest.mark.parametrize(
    "payload",
    [
        {"images": "not-an-array"},
        {"images": ["data:image/png;base64,%%%"]},
        {"images": ["data:image/png;base64,YWJj"]},
        {"images": ["https://example.org/picture.png"]},
        {"images": [BLUE_PNG] * 17},
        {"suggestion_title": "[图片0]"},
        {"suggestion_title": "[图片2]"},
        {"options": "第一项\n第二项", "suggestion_options": "[图片1]"},
    ],
)
def test_invalid_image_request_stops_before_model(image_answer_api, payload):
    client, provider, _ = image_answer_api
    request = {"title": "看图", "images": [BLUE_PNG], **payload}
    response = client.post("/api/v1/answer", json=request)
    assert response.status_code == 422
    assert BLUE_PNG not in response.text
    assert not provider.received


def test_image_and_http_request_size_limits(image_answer_api, monkeypatch):
    client, provider, _ = image_answer_api
    payload = {"title": "看图", "images": [BLUE_PNG, RED_PNG]}
    monkeypatch.setattr(images, "MAX_TOTAL_IMAGE_BYTES", 100)
    response = client.post("/api/v1/answer", json=payload)
    assert response.status_code == 422
    assert "combined decoded image limit" in response.json()["detail"]
    monkeypatch.setattr(images, "MAX_IMAGE_BYTES", 50)
    response = client.post("/api/v1/answer", json=payload)
    assert response.status_code == 422
    assert "decoded image limit" in response.json()["detail"]
    monkeypatch.setattr(dependencies, "MAX_ANSWER_BODY_BYTES", 10)
    response = client.post("/api/v1/answer", content=iter([b'{"title":', b'"figure"}']))
    assert response.status_code == 413
    assert not provider.received
