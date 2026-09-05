from __future__ import annotations

from typing import Protocol

from ocs_llm_answerer.answer.models import Question
from ocs_llm_answerer.llm.models import LLMCallResult, LLMProviderMetadata


class LLMProvider(Protocol):
    """AnswerService 和测试使用的最小 provider 契约。"""

    name: str
    model: str
    request_metadata: LLMProviderMetadata

    async def aclose(self) -> None:
        """释放 Provider 持有的客户端资源。"""
        ...

    async def answer(self, request: Question) -> LLMCallResult:
        """返回结构化答案，不负责题型语义检查或 OCS 分隔符转换。

        Args:
            request: 答题层提供的标准化题目。

        Returns:
            答案项、原始响应及调用元数据。

        Raises:
            ProviderError: 可预期的上游连接、超时或响应错误。
        """
        ...
