"""不依赖具体 SDK 的 Provider 错误契约。"""

from __future__ import annotations

from ocs_llm_answerer.llm.models import LLMUsage


class ProviderError(RuntimeError):
    """保留内部诊断信息的模型调用错误，不直接向 HTTP 客户端暴露原文。

    Attributes:
        raw_body: 上游响应原文；尚未收到响应时为 None。
        status_code: 上游实际状态码，不是本服务的 HTTP 响应码。
    """

    def __init__(
        self, message: str, *, raw_body: str | None = None, status_code: int | None = None
    ) -> None:
        """保存诊断信息。

        Args:
            message: 仅供内部审计使用的错误说明。
            raw_body: 未改写的上游响应正文。
            status_code: 上游 HTTP 状态码。
        """
        super().__init__(message)
        self.raw_body = raw_body
        self.status_code = status_code


class ProviderTimeoutError(ProviderError):
    """模型调用超过配置的等待时间。"""


class ProviderUnavailableError(ProviderError):
    """模型服务连接失败、限流或暂时不可用。"""


class ProviderResponseError(ProviderError):
    """模型服务返回错误状态或无法解析的响应。"""


class ProviderOutputError(ProviderResponseError):
    """上游返回了模型输出文本，但无法解析为目标结构化 Schema（可触发纠正重试）。

    Attributes:
        usage: 该次失败输出实际消耗的上游 token 用量。
        raw_output_text: 从响应中提取出的模型输出文本。
    """

    def __init__(
        self,
        message: str,
        *,
        raw_body: str | None = None,
        status_code: int | None = None,
        usage: LLMUsage | None = None,
        raw_output_text: str | None = None,
    ) -> None:
        """保存输出解析失败的完整诊断与用量现场。

        Args:
            message: 结构解析失败原因，可用于审计和下一轮纠正反馈。
            raw_body: 上游完整响应正文。
            status_code: 上游 HTTP 状态码。
            usage: 上游报告的 token 用量。
            raw_output_text: 未通过校验的模型输出文本。
        """
        super().__init__(message, raw_body=raw_body, status_code=status_code)
        self.usage = usage if usage is not None else LLMUsage()
        self.raw_output_text = raw_output_text
