"""Deterministic, offline OpenAPI contract export."""

import json
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def application_version() -> str:
    try:
        return distribution_version("maitian-agent")
    except PackageNotFoundError:
        from maitian_agent import __version__

        return __version__


VERSION = application_version()


def openapi_document(app=None) -> dict:
    if app is None:
        from maitian_agent.api.routes import create_app

        app = create_app()
    from fastapi.openapi.utils import get_openapi

    document = get_openapi(title=app.title, version=application_version(), routes=app.routes)
    document["info"]["version"] = application_version()
    document["info"]["title"] = "麦田智囊 API"
    from maitian_agent.api.routes import FailedEnvelope

    failed_schema = FailedEnvelope.model_json_schema(ref_template="#/components/schemas/{model}")
    document.setdefault("components", {}).setdefault("schemas", {}).update(
        failed_schema.pop("$defs", {})
    )
    document["components"]["schemas"]["FailedEnvelope"] = failed_schema
    for path, path_item in document["paths"].items():
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            if path.startswith("/api/v1/"):
                for status in ("400", "404", "405", "413", "415", "422", "500", "503"):
                    operation.setdefault("responses", {})[status] = {
                        "description": "统一失败信封",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/FailedEnvelope"}
                            }
                        },
                    }
            if path == "/api/wisdom-transfer" and method == "post":
                for status in ("413", "415"):
                    operation["responses"][status] = {
                        "description": "上传失败",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"detail": {"type": "string"}},
                                }
                            }
                        },
                    }
            if path.startswith("/api/") and not path.startswith("/api/v1/") and method == "post":
                for status in ("422", "500", "503"):
                    operation["responses"][status] = {
                        "description": "Legacy error",
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"detail": {}}}
                            }
                        },
                    }
    return document


def export_openapi(path: str | Path = "openapi.json") -> None:
    target = Path(path)
    if not target.is_absolute():
        target = ROOT / target
    target.write_text(
        json.dumps(openapi_document(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    export_openapi(sys.argv[1] if len(sys.argv) > 1 else "openapi.json")
