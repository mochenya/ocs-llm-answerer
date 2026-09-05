"""不依赖具体 SDK 的 Provider 错误契约。"""


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
