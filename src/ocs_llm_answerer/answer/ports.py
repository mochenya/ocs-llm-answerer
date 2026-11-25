"""答题服务依赖的最小仓储协议，可由 SQLite 或内存替身实现。"""

from __future__ import annotations

from typing import Protocol

from ocs_llm_answerer.answer.models import CachedAnswer, LLMCallAudit, RequestAudit


class AnswerCachePort(Protocol):
    """已校验答案的读取与独立事务写入契约。"""

    async def get(
        self,
        question_hash: str,
        *,
        audit: RequestAudit | None = None,
        seen_at_ns: int | None = None,
    ) -> CachedAnswer | None:
        """读取答案并记录本次命中。

        Args:
            question_hash: 标准化题目的稳定身份。
            audit: 本次原始请求快照，提供时更新题目最近快照。
            seen_at_ns: 请求进入服务的 Unix 纳秒时间。

        Returns:
            答案及来源信息，未命中时为 None。

        Raises:
            RepositoryError: 读取或命中统计持久化失败。
        """
        ...

    async def set(
        self,
        record: CachedAnswer,
        *,
        seen_at_ns: int | None = None,
        audit: RequestAudit | None = None,
    ) -> None:
        """保存答案，不回滚已经提交的来源调用。

        Args:
            record: 引用已存在流水的合法答案。
            seen_at_ns: 本次请求时间。
            audit: 与答案数据分离的原始请求快照。

        Raises:
            RepositoryError: 写入或引用约束检查失败。
        """
        ...


class LLMRequestPort(Protocol):
    """成功和失败模型调用的独立审计事务契约。"""

    async def record(self, entry: LLMCallAudit) -> int:
        """提交单次模型调用流水并返回已提交主键。

        每次上游模型调用独立对应一条流水记录；后续缓存写入失败不会回滚已提交的记录。

        Args:
            entry: 完整的调用配置、状态、耗时、原始响应、错误及用量快照。

        Returns:
            已提交的调用流水主键。

        Raises:
            RepositoryError: 审计持久化失败。
        """
        ...
