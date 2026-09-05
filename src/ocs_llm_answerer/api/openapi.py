"""补充手动解析请求体的 OpenAPI 组件，所有引用使用组件级路径。"""

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

from ocs_llm_answerer.api.schemas import AnswerRequest


def configure_openapi(app: FastAPI) -> None:
    """在路由组装完成后生成文档缓存，补入手动请求模型及其依赖。

    后续若动态修改路由，需要再次调用此函数重新生成缓存。

    Args:
        app: 已完成路由注册的应用。
    """
    schema = get_openapi(
        title=app.title,
        version=app.version,
        openapi_version=app.openapi_version,
        description=app.description,
        routes=app.routes,
    )
    request_schema = AnswerRequest.model_json_schema(ref_template="#/components/schemas/{model}")
    components = schema.setdefault("components", {}).setdefault("schemas", {})
    components.update(request_schema.pop("$defs", {}))
    components["AnswerRequest"] = request_schema
    app.openapi_schema = schema
