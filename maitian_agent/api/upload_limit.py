"""Bound upload traffic before multipart parsing and spooling."""

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from maitian_agent.openapi import application_version

try:
    from starlette.routing import get_route_path
except ImportError:
    try:
        from starlette._utils import get_route_path
    except ImportError:

        def get_route_path(scope: Scope) -> str:
            path = scope["path"]
            root_path = scope.get("root_path", "")
            if root_path and (path == root_path or path.startswith(root_path + "/")):
                return path[len(root_path) :]
            return path


# Allow multipart headers/boundaries in addition to the exact file limit.
MULTIPART_ALLOWANCE_BYTES = 64 * 1024


class UploadLimitMiddleware:
    def __init__(self, app: ASGIApp, settings):
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        route_path = get_route_path(scope) if scope["type"] == "http" else ""
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or route_path not in {"/api/wisdom-transfer", "/api/v1/wisdom-transfer"}
        ):
            await self.app(scope, receive, send)
            return

        limit = self.settings.max_upload_mb * 1024 * 1024 + MULTIPART_ALLOWANCE_BYTES
        legacy = route_path == "/api/wisdom-transfer"

        def too_large():
            if legacy:
                return JSONResponse(
                    {"detail": "上传内容过大，请上传大小限制以内的图片"}, status_code=413
                )
            return JSONResponse(
                {
                    "status": "failed",
                    "data": None,
                    "error": {
                        "code": "UPLOAD_TOO_LARGE",
                        "message": "上传内容过大，请上传大小限制以内的图片",
                    },
                    "meta": {"model": None, "duration_ms": 0, "version": application_version()},
                },
                status_code=413,
            )

        response = too_large()
        for name, value in scope.get("headers", []):
            if name.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    continue
                if declared > limit:
                    await response(scope, receive, send)
                    return

        received = 0
        exceeded = False

        async def limited_receive():
            nonlocal received, exceeded
            if exceeded:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    exceeded = True
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise
        if exceeded:
            await response(scope, receive, send)
