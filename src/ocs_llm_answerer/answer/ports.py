"""答题服务依赖的最小仓储协议，可由 SQLite 或内存替身实现。"""

from typing import Protocol

from ocs_llm_answerer.answer.models import CachedAnswer, NormalizedQuestion, RequestAudit
from ocs_llm_answerer.llm.models import LLMCallResult, LLMProviderMetadata


class AnswerCachePort(Protocol):
    """已校验答案的读取与独立事务写入契约。"""

    async def get(
        self,
        question_hash: str,
        question_raw_json: str | None = None,
        seen_at_ns: int | None = None,
    ) -> CachedAnswer | None:
        """读取答案并记录本次命中。

        Args:
            question_hash: 标准化题目的稳定身份。
            question_raw_json: 本次原始载荷，提供时更新题目最近快照。
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
        seen_at_ns: int | None = None,
        *,
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
    """成功和失败调用的独立审计事务契约。"""

    async def record_success(
        self,
        question: NormalizedQuestion,
        metadata: LLMProviderMetadata,
        result: LLMCallResult,
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        latency_ms: int,
        *,
        audit: RequestAudit | None = None,
    ) -> int:
        """提交成功调用，后续缓存失败不撤销此记录。

        Args:
            question: 具有稳定身份的题目。
            metadata: Provider 配置快照。
            result: 已通过题型校验的调用结果。
            seen_at_ns: 请求进入服务的时间。
            request_started_at_ns: 调用开始时间。
            request_completed_at_ns: 调用完成时间。
            latency_ms: 单调时钟测得的耗时。
            audit: 原始请求快照。

        Returns:
            已提交的调用主键。

        Raises:
            RepositoryError: 审计持久化失败。
        """
        ...

    async def record_failure(
        self,
        question: NormalizedQuestion,
        metadata: LLMProviderMetadata,
        error: Exception,
        seen_at_ns: int,
        request_started_at_ns: int,
        request_completed_at_ns: int,
        latency_ms: int,
        response_body_raw: str | None = None,
        result: LLMCallResult | None = None,
        *,
        audit: RequestAudit | None = None,
    ) -> int:
        """记录失败现场，不生成成功缓存。

        Args:
            question: 具有稳定身份的题目。
            metadata: Provider 配置快照。
            error: 原始答题异常。
            seen_at_ns: 请求进入服务的时间。
            request_started_at_ns: 调用开始时间。
            request_completed_at_ns: 调用完成时间。
            latency_ms: 调用耗时。
            response_body_raw: 错误响应原文。
            result: 已收到但校验失败的模型结果。
            audit: 原始请求快照。

        Returns:
            已提交的失败调用主键。

        Raises:
            RepositoryError: 审计持久化失败。
        """
        ...
