"""不访问网络的类型完整 Provider，用于服务和生命周期契约测试。"""

from ocs_llm_answerer.answer.models import Question
from ocs_llm_answerer.llm.models import LLMAnswer, LLMCallResult, LLMProviderMetadata, LLMUsage


class StubProvider:
    """保存调用和关闭次数的模型替身。"""

    def __init__(self, *, error: Exception | None = None, answers: list[str] | None = None) -> None:
        """配置固定结果或待抛出的异常。

        Args:
            error: 每次调用时抛出的同一个异常实例。
            answers: 结构化答案项，缺省时为单项 A。
        """
        self.name = "stub"
        self.model = "stub-model"
        self.error = error
        self.received: list[Question] = []
        self.close_calls = 0
        self.request_metadata = LLMProviderMetadata(
            adapter="stub",
            base_url="https://test.invalid/v1",
            api_key_env="STUB_KEY",
            model=self.model,
            timeout_seconds=1,
            max_retries=0,
        )
        self.result = LLMCallResult(
            answer=LLMAnswer(answers=["A"] if answers is None else answers, confidence=0.9),
            response_body_raw='{"upstream":"original"}',
            http_status=200,
            usage=LLMUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        )

    async def answer(self, request: Question) -> LLMCallResult:
        """保存题目并返回预设结果。

        Args:
            request: 服务传入的内部题目。

        Returns:
            预设的结构化调用结果。

        Raises:
            Exception: 初始化时指定的错误。
        """
        self.received.append(request)
        if self.error is not None:
            raise self.error
        return self.result

    async def aclose(self) -> None:
        """记录关闭调用，不执行 I/O。"""
        self.close_calls += 1
