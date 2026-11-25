"""业务无关的通用 LLM 客户端与单请求会话协议。"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel

from ocs_llm_answerer.llm.models import LLMCallResult, LLMProviderMetadata, UserInput


class LLMSession(Protocol):
    """单次请求作用域的模型会话，内部维护本厂商强类型原生消息数组（含思考块）。"""

    async def send_structured[TOutput: BaseModel](
        self,
        user_input: UserInput,
        response_model: type[TOutput],
    ) -> LLMCallResult[TOutput]:
        """追加用户输入并发起一次结构化生成调用。

        网络或上游 HTTP 故障不得污染会话历史；一旦收到合法上游响应并提取原生输出项，
        即保留该轮完整输出（含 reasoning 思考块等），以便业务层在解析或校验失败后
        继续追加纠正反馈。

        Args:
            user_input: 本轮发送的多模态用户输入。
            response_model: 期望解析的目标 Pydantic 模型。

        Returns:
            泛型结构化结果、原始响应、HTTP 状态与 token 用量。

        Raises:
            ProviderOutputError: 收到模型输出文本但结构解析失败（可纠正重试）。
            ProviderTimeoutError: 上游调用超时。
            ProviderUnavailableError: 连接失败、限流或上游暂时不可用。
            ProviderResponseError: 上游错误状态、损坏响应体或缺失输出文本。
        """
        ...


class LLMClient(Protocol):
    """长生命周期、并发安全的通用 LLM 客户端契约。"""

    name: str
    model: str
    request_metadata: LLMProviderMetadata

    def create_session(self, *, instructions: str) -> LLMSession:
        """创建单次请求作用域的新会话，各请求之间不共享对话历史。

        Args:
            instructions: 覆盖本次会话的系统指令。

        Returns:
            仅限当前业务请求使用的独立会话实例。
        """
        ...

    async def aclose(self) -> None:
        """释放客户端持有的底层连接资源。"""
        ...
