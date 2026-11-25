"""不访问网络的类型完整 LLMClient / LLMSession 替身，用于服务、API 和生命周期契约测试。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

from pydantic import BaseModel

from ocs_llm_answerer.answer.models import AnswerPayload
from ocs_llm_answerer.llm.models import (
    LLMCallResult,
    LLMProviderMetadata,
    LLMUsage,
    UserInput,
)

TOutput = TypeVar("TOutput", bound=BaseModel)


class StubLLMSession:
    """记录单次请求内所有轮次输入的内存会话替身。"""

    def __init__(self, client: StubLLMClient, instructions: str) -> None:
        self._client = client
        self.instructions = instructions
        self.received_inputs: list[UserInput] = []
        self.response_models: list[type[BaseModel]] = []

    async def send_structured[T: BaseModel](
        self,
        user_input: UserInput,
        response_model: type[T],
    ) -> LLMCallResult[T]:
        """按客户端脚本顺序返回结果或抛出异常，并记录每次输入。"""
        self.received_inputs.append(user_input)
        self.response_models.append(response_model)
        return await self._client._dispatch(self, user_input, response_model)


class StubLLMClient:
    """保存会话历史、调用序列和关闭次数的通用 LLM 客户端替身。"""

    def __init__(
        self,
        *,
        error: Exception | None = None,
        answers: list[str] | None = None,
        script: Sequence[LLMCallResult[AnswerPayload] | Exception] | None = None,
    ) -> None:
        """配置固定结果、固定异常或按次执行的脚本序列。

        Args:
            error: 每次调用时抛出的同一个异常实例。
            answers: 固定返回的答案项，缺省时为单项 A。
            script: 每次 send_structured 依次消费的结果或异常；耗尽后重复最后一项。
        """
        self.name = "stub"
        self.model = "stub-model"
        self.error = error
        self._script = list(script) if script is not None else None
        self._script_index = 0
        self.sessions: list[StubLLMSession] = []
        self.received: list[UserInput] = []
        self.close_calls = 0
        self.request_metadata = LLMProviderMetadata(
            adapter="stub",
            base_url="https://test.invalid/v1",
            api_key_env="STUB_KEY",
            model=self.model,
            timeout_seconds=1,
            max_retries=0,
        )
        self.result: LLMCallResult[AnswerPayload] = LLMCallResult(
            parsed=AnswerPayload(
                answers=["A"] if answers is None else answers,
                explanation="",
                confidence=0.9,
            ),
            response_body_raw='{"upstream":"original"}',
            http_status=200,
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    def create_session(self, *, instructions: str) -> StubLLMSession:
        """为单次请求创建独立会话替身。"""
        session = StubLLMSession(self, instructions)
        self.sessions.append(session)
        return session

    async def _dispatch[T: BaseModel](
        self,
        session: StubLLMSession,
        user_input: UserInput,
        response_model: type[T],
    ) -> LLMCallResult[T]:
        self.received.append(user_input)
        if self._script is not None:
            index = min(self._script_index, len(self._script) - 1)
            self._script_index += 1
            outcome = self._script[index]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome  # type: ignore[return-value]
        if self.error is not None:
            raise self.error
        return self.result  # type: ignore[return-value]

    async def aclose(self) -> None:
        """记录关闭调用，不执行 I/O。"""
        self.close_calls += 1
