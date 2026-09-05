"""答题用例与仓储共享的持久化错误契约。"""


class RepositoryError(RuntimeError):
    """仓储无法完成读写；原始数据库错误保留在异常链中。"""
