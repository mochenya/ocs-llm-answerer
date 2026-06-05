from __future__ import annotations

import argparse
from collections.abc import Sequence

import uvicorn

from ocs_llm_answerer import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ocs-llm-answerer",
        description="Run the OCS LLM Answerer FastAPI backend.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind. Defaults to 127.0.0.1.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind. Defaults to 8000.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload for local development.",
    )
    parser.add_argument("--log-level", default="info", help="Uvicorn log level. Defaults to info.")
    parser.add_argument(
        "--version",
        action="version",
        version=f"OCS LLM Answerer {__version__}",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    uvicorn.run(
        "ocs_llm_answerer.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
