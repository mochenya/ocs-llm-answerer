"""静态守护单向包依赖架构（DAG），防止跨层反向耦合回归。"""

from __future__ import annotations

import ast
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "ocs_llm_answerer"

FORBIDDEN_PACKAGE_IMPORTS: dict[str, set[str]] = {
    "core": {"answer", "api", "database", "llm", "main", "cli"},
    "llm": {"answer", "api", "database", "main", "cli"},
    "answer": {"api", "database", "main", "cli"},
    "database": {"api", "main", "cli"},
    "api": {"database", "main", "cli"},
}


def _imported_internal_packages(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    packages: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "ocs_llm_answerer":
                    packages.add(parts[1])
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("ocs_llm_answerer.")
        ):
            parts = node.module.split(".")
            if len(parts) >= 2:
                packages.add(parts[1])
    return packages


def test_package_dependencies_follow_unidirectional_dag() -> None:
    """核心边界必须满足 core <- llm <- answer <- (api, database) 单向无环依赖。"""
    violations: list[str] = []
    for package, forbidden in FORBIDDEN_PACKAGE_IMPORTS.items():
        package_dir = SRC_ROOT / package
        for file_path in sorted(package_dir.rglob("*.py")):
            imported = _imported_internal_packages(file_path)
            illegal = sorted(imported & forbidden)
            if illegal:
                rel = file_path.relative_to(SRC_ROOT)
                violations.append(f"{rel} illegally imports {illegal}")
    assert violations == []
