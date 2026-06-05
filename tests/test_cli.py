import pytest

from ocs_llm_answerer import __version__, cli


def capture_uvicorn_run(monkeypatch) -> dict[str, object]:
    call: dict[str, object] = {}

    def fake_run(app: str, **kwargs: object) -> None:
        call["app"] = app
        call.update(kwargs)

    monkeypatch.setattr(cli.uvicorn, "run", fake_run)
    return call


def test_cli_uses_safe_local_defaults(monkeypatch):
    call = capture_uvicorn_run(monkeypatch)

    cli.main([])

    assert call == {
        "app": "ocs_llm_answerer.main:app",
        "host": "127.0.0.1",
        "port": 8000,
        "reload": False,
        "log_level": "info",
    }


def test_cli_passes_uvicorn_arguments(monkeypatch):
    call = capture_uvicorn_run(monkeypatch)

    cli.main(["--host", "0.0.0.0", "--port", "9000", "--reload", "--log-level", "warning"])

    assert call == {
        "app": "ocs_llm_answerer.main:app",
        "host": "0.0.0.0",
        "port": 9000,
        "reload": True,
        "log_level": "warning",
    }


def test_cli_prints_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--version"])

    assert exc_info.value.code == 0
    assert capsys.readouterr().out == f"OCS LLM Answerer {__version__}\n"
