"""统一 SQLite 连接配置和错误转换，不隐式改变仓储的提交边界。"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path

import aiosqlite

from ocs_llm_answerer.answer.errors import RepositoryError


@asynccontextmanager
async def connect_database(database_path: Path) -> AsyncIterator[aiosqlite.Connection]:
    """打开启用外键的短连接，将数据库故障转换成仓储错误。

    Args:
        database_path: SQLite 数据库文件路径。

    Yields:
        使用命名列读取结果的连接；提交由调用方显式执行。

    Raises:
        RepositoryError: 连接、查询或提交失败，底层异常保留为 cause。
    """
    try:
        async with aiosqlite.connect(database_path) as db:
            await db.execute("PRAGMA foreign_keys = ON")
            db.row_factory = aiosqlite.Row
            yield db
    except aiosqlite.Error as exc:
        raise RepositoryError("SQLite operation failed") from exc


async def init_sqlite(database_path: Path, schema_path: Path | None = None) -> None:
    """创建当前结构，不迁移或改写旧数据库。

    Args:
        database_path: 数据库路径，父目录不存在时会创建。
        schema_path: 可选建表脚本，默认读取包内 schema.sql。

    Raises:
        OSError: 数据库目录或建表脚本不可访问。
        RepositoryError: 数据库连接或建表失败。
    """
    database_path.parent.mkdir(parents=True, exist_ok=True)
    schema_sql = (
        schema_path.read_text(encoding="utf-8")
        if schema_path is not None
        else files(__package__).joinpath("schema.sql").read_text(encoding="utf-8")
    )
    async with connect_database(database_path) as db:
        await db.executescript(schema_sql)
        await db.commit()
