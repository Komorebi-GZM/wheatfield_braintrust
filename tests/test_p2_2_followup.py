"""Regression tests for the P2-2 contract repair."""

import asyncio
import inspect
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import httpx

from maitian_agent.api import routes
from maitian_agent.config.settings import get_settings


def _request(app, method, path, **kwargs):
    root_path = kwargs.pop("_root_path", "")

    async def call():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, root_path=root_path), base_url="http://test"
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(call())


def _configure(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_BASE_URL", "https://model.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "configured-model")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://trusted.test")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()

    async def direct_call(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(routes, "run_in_threadpool", direct_call)


def test_framework_errors_use_v1_envelope_and_preserve_legacy_shapes(monkeypatch, tmp_path):
    _configure(monkeypatch, tmp_path)
    app = routes.create_app()
    checks = [
        ("get", "/api/v1/route", None, 405, "HTTP_ERROR", "post"),
        ("post", "/api/v1/missing", {}, 404, "HTTP_ERROR", None),
        ("post", "/api/v1/wisdom-transfer", None, 400, "VALIDATION_ERROR", None),
    ]
    for method, path, body, status, code, allow in checks:
        headers = {"Origin": "http://trusted.test"}
        if path.endswith("wisdom-transfer"):
            headers["Content-Type"] = "multipart/form-data"
        response = _request(app, method.upper(), path, json=body, headers=headers)
        assert response.status_code == status
        assert response.json()["status"] == "failed"
        assert response.json()["data"] is None
        assert response.json()["error"]["code"] == code
        assert response.json()["meta"]["model"] is None
        assert response.headers["access-control-allow-origin"] == "http://trusted.test"
        if allow:
            assert response.headers["allow"] == "POST"
        assert "Missing boundary" not in response.text
        assert "Method Not Allowed" not in response.text
        assert "Not Found" not in response.text

    cors_headers = {"Origin": "http://trusted.test"}
    legacy_get = _request(app, "GET", "/api/route", headers=cors_headers)
    legacy_404 = _request(app, "POST", "/api/missing", json={}, headers=cors_headers)
    legacy_boundary = _request(
        app,
        "POST",
        "/api/wisdom-transfer",
        content=b"",
        headers={**cors_headers, "Content-Type": "multipart/form-data"},
    )
    assert legacy_get.status_code == 405
    assert legacy_get.json() == {"detail": "Method Not Allowed"}
    assert legacy_get.headers["allow"] == "POST"
    assert legacy_404.status_code == 404
    assert legacy_404.json() == {"detail": "Not Found"}
    assert legacy_boundary.status_code == 400
    assert legacy_boundary.json() == {"detail": "Missing boundary in multipart."}
    for response in (legacy_get, legacy_404, legacy_boundary):
        assert response.headers["access-control-allow-origin"] == "http://trusted.test"

    from fastapi import FastAPI

    mounted = FastAPI()
    mounted.mount("/nested", routes.create_app())
    for method, path, expected_status in (
        ("GET", "/nested/api/v1/route", 405),
        ("POST", "/nested/api/v1/not-found", 404),
        ("POST", "/nested/api/v1/wisdom-transfer", 400),
    ):
        mounted_headers = {"Origin": "http://trusted.test"}
        if path.endswith("wisdom-transfer"):
            mounted_headers["Content-Type"] = "multipart/form-data"
        response = _request(mounted, method, path, headers=mounted_headers)
        assert response.status_code == expected_status
        assert response.json()["status"] == "failed"
        assert response.headers["access-control-allow-origin"] == "http://trusted.test"

    for method, path, expected_status, expected_body in (
        ("GET", "/tenant/api/route", 405, {"detail": "Method Not Allowed"}),
        ("POST", "/tenant/api/not-found", 404, {"detail": "Not Found"}),
        (
            "POST",
            "/tenant/api/wisdom-transfer",
            400,
            {"detail": "Missing boundary in multipart."},
        ),
    ):
        request_headers = {"Origin": "http://trusted.test"}
        if path.endswith("wisdom-transfer"):
            request_headers["Content-Type"] = "multipart/form-data"
        response = _request(
            routes.create_app(),
            method,
            path,
            headers=request_headers,
            _root_path="/tenant",
        )
        assert response.status_code == expected_status
        assert response.json() == expected_body
        if expected_status == 405:
            assert response.headers["allow"] == "POST"


def test_generated_openapi_tracks_runtime_routes_models_and_all_error_responses(
    monkeypatch, tmp_path
):
    _configure(monkeypatch, tmp_path)
    app = routes.create_app()
    schema = app.openapi()
    assert schema["paths"]["/"]["get"]["responses"]["200"]["content"]
    assert schema["paths"]["/health"]["get"]["responses"]["200"]["content"]
    for path in ("/api/v1/route", "/api/v1/wisdom-transfer"):
        responses = schema["paths"][path]["post"]["responses"]
        for code in ("400", "404", "405", "413", "415", "422", "500", "503"):
            assert "content" in responses[code]
            assert "schema" in responses[code]["content"]["application/json"]
    legacy_upload = schema["paths"]["/api/wisdom-transfer"]["post"]["responses"]
    for code in ("413", "415"):
        assert "schema" in legacy_upload[code]["content"]["application/json"]

    from maitian_agent.openapi import openapi_document

    baseline = json.dumps(openapi_document(app), sort_keys=True)
    route = next(
        route for route in routes.router.routes if getattr(route, "path", None) == "/api/v1/route"
    )
    original_operation_id = route.operation_id
    try:
        route.operation_id = "routeIntentMutationProbe"
        changed = json.dumps(openapi_document(routes.create_app()), sort_keys=True)
    finally:
        route.operation_id = original_operation_id
    assert baseline != changed


def test_all_json_api_business_entries_pass_named_input_models(monkeypatch, tmp_path):
    from maitian_agent.agents.results import (
        AgentResult,
        LessonPrepInput,
        MaterialInput,
        MeetingNotesInput,
        QuizInput,
        RouteData,
        RouteInput,
        TextData,
        WisdomData,
        WisdomInput,
    )

    _configure(monkeypatch, tmp_path)
    classes = {
        "quick_lesson_prep": LessonPrepInput,
        "classroom_companion": QuizInput,
        "material": MaterialInput,
        "meeting_notes": MeetingNotesInput,
        "router": RouteInput,
        "wisdom_transfer": WisdomInput,
    }

    class StrictAgent:
        def __init__(self, model):
            self.model = model

        def run(self, input_model):
            assert isinstance(input_model, self.model)
            if self.model is WisdomInput:
                return AgentResult(
                    status="needs_review",
                    data=WisdomData(text="待校订", review_reasons=["OCR_NO_TEXT"]),
                )
            data = (
                RouteData(
                    intent="general_chat",
                    description="通用对话",
                    available_intents=["general_chat"],
                )
                if self.model is RouteInput
                else TextData(text="ok")
            )
            return AgentResult(status="success", data=data)

    routes.agents = {key: StrictAgent(value) for key, value in classes.items()}
    app = routes.create_app()
    payloads = {
        "lesson-prep": {"subject": "语文", "grade": "三年级", "topic": "春天"},
        "quiz": {"subject": "语文", "grade": "三年级", "topic": "春天"},
        "material": {"subject": "语文", "grade": "三年级", "topic": "春天"},
        "meeting-notes": {"transcript": "讨论"},
        "route": {"user_input": "聊天"},
    }
    for endpoint, payload in payloads.items():
        assert _request(app, "POST", f"/api/v1/{endpoint}", json=payload).status_code == 200
    from io import BytesIO

    from PIL import Image

    image = BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    wisdom = _request(
        app,
        "POST",
        "/api/v1/wisdom-transfer",
        files={"file": ("lesson.png", image.getvalue(), "image/png")},
    )
    assert wisdom.status_code == 200
    assert wisdom.json()["status"] == "needs_review"


def test_each_agent_run_has_one_named_input_model_parameter():
    from maitian_agent.agents.classroom_companion import ClassroomCompanionAgent
    from maitian_agent.agents.material_agent import MaterialAgent
    from maitian_agent.agents.meeting_notes import MeetingNotesAgent
    from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
    from maitian_agent.agents.router import RouterAgent
    from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent

    for agent_type in (
        QuickLessonPrepAgent,
        ClassroomCompanionAgent,
        MaterialAgent,
        MeetingNotesAgent,
        RouterAgent,
        WisdomTransferAgent,
    ):
        parameters = list(inspect.signature(agent_type.run).parameters.values())
        assert len(parameters) == 2
        assert parameters[1].name == "input_model"


def test_actual_invocation_sets_model_and_uncalled_requests_do_not(monkeypatch, tmp_path):
    from io import BytesIO
    from unittest.mock import Mock

    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from PIL import Image

    from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
    from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent

    _configure(monkeypatch, tmp_path)
    routes.agents = {
        "quick_lesson_prep": QuickLessonPrepAgent(FakeListChatModel(responses=["done"]))
    }
    app = routes.create_app()
    called = _request(
        app,
        "POST",
        "/api/v1/lesson-prep",
        json={"subject": "语文", "grade": "三年级", "topic": "春天"},
    )
    assert called.json()["meta"]["model"], called.text
    assert "FakeListChatModel" in called.json()["meta"]["model"]

    uncalled = _request(app, "POST", "/api/v1/route", json={})
    assert uncalled.status_code == 422
    assert uncalled.json()["meta"]["model"] is None

    image = BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    reader = Mock()
    reader.readtext.return_value = []
    routes.agents["wisdom_transfer"] = WisdomTransferAgent(
        ocr_reader=reader, llm=FakeListChatModel(responses=["unused"])
    )
    empty = _request(
        app,
        "POST",
        "/api/v1/wisdom-transfer",
        files={"file": ("x.png", image.getvalue(), "image/png")},
    )
    assert empty.json()["status"] == "needs_review"
    assert empty.json()["meta"]["model"] is None

    reader.readtext.return_value = [((0, 0), "课堂导入", 0.9)]
    routes.agents["wisdom_transfer"] = WisdomTransferAgent(
        ocr_reader=reader, llm=FakeListChatModel(responses=["结构化结果"])
    )
    structured = _request(
        app,
        "POST",
        "/api/v1/wisdom-transfer",
        files={"file": ("x.png", image.getvalue(), "image/png")},
    )
    assert structured.json()["data"]["review_reasons"] == ["OCR_REQUIRES_REVIEW"]
    assert "FakeListChatModel" in structured.json()["meta"]["model"]

    too_early = _request(
        app,
        "POST",
        "/api/v1/wisdom-transfer",
        content=b"x",
        headers={"Content-Type": "multipart/form-data", "Content-Length": str(1024 * 1024 + 65537)},
    )
    assert too_early.status_code == 413
    assert too_early.json()["meta"]["model"] is None


def test_failed_after_fake_model_invocation_keeps_model_identifier(monkeypatch, tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent

    class FailAfterCall(FakeListChatModel):
        def _call(self, *args, **kwargs):
            super()._call(*args, **kwargs)
            raise RuntimeError("private failure")

    _configure(monkeypatch, tmp_path)
    routes.agents = {"quick_lesson_prep": QuickLessonPrepAgent(FailAfterCall(responses=["ok"]))}
    response = _request(
        routes.create_app(),
        "POST",
        "/api/v1/lesson-prep",
        json={"subject": "语文", "grade": "三年级", "topic": "春天"},
    )
    assert response.status_code == 500
    assert response.json()["meta"]["model"], response.text
    assert "private failure" not in response.text


def test_invoke_agent_explains_missing_model_configuration_without_secrets(monkeypatch):
    from maitian_agent.agent_call import invoke_agent
    from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
    from maitian_agent.agents.results import LessonPrepInput

    for name in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    messages = []
    result = invoke_agent(
        QuickLessonPrepAgent().run,
        LessonPrepInput(subject="语文", grade="三年级", topic="春天"),
        error_handler=messages.append,
    )
    assert result is None
    assert "请配置大模型" in messages[0]
    assert "LLM_BASE_URL" not in messages[0]
    assert "LLM_API_KEY" not in messages[0]


def test_cli_business_entries_pass_named_input_models(monkeypatch):
    from click.testing import CliRunner

    from maitian_agent.agents.results import (
        AgentResult,
        LessonPrepInput,
        RouteData,
        RouteInput,
        TextData,
    )
    from maitian_agent.cli import cli

    class LessonAgent:
        def run(self, input_model):
            assert isinstance(input_model, LessonPrepInput)
            return AgentResult(status="success", data=TextData(text="教案"))

    class RouteAgent:
        def run(self, input_model):
            assert isinstance(input_model, RouteInput)
            return AgentResult(
                status="success",
                data=RouteData(
                    intent="general_chat",
                    description="通用对话",
                    available_intents=["general_chat"],
                ),
            )

    monkeypatch.setattr("maitian_agent.cli.QuickLessonPrepAgent", LessonAgent)
    monkeypatch.setattr("maitian_agent.cli.RouterAgent", RouteAgent)
    runner = CliRunner()
    lesson = runner.invoke(
        cli, ["lesson-prep", "--subject", "语文", "--grade", "三年级", "--topic", "春天"]
    )
    route = runner.invoke(cli, ["route", "--input", "你好"])
    assert lesson.exit_code == 0, lesson.output
    assert route.exit_code == 0, route.output


def test_wheel_import_health_and_openapi_without_source_tree(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    wheel = wheel_dir / "maitian_agent-1.0.0-py3-none-any.whl"
    dist_info = "maitian_agent-1.0.0.dist-info"
    with zipfile.ZipFile(wheel, "w") as archive:
        for package_file in (root / "maitian_agent").rglob("*.py"):
            archive.write(package_file, package_file.relative_to(root))
        archive.writestr(
            f"{dist_info}/METADATA",
            "Metadata-Version: 2.1\nName: maitian-agent\nVersion: 1.0.0\n",
        )
        archive.writestr(
            f"{dist_info}/WHEEL",
            "Wheel-Version: 1.0\nGenerator: regression-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n",
        )
        archive.writestr(f"{dist_info}/RECORD", "")
    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extract)
    script = """
import json
import asyncio
import httpx
import os
import maitian_agent
from maitian_agent.api.routes import create_app
from maitian_agent.openapi import export_openapi, openapi_document
app = create_app()
async def check_health():
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
            assert (await client.get('/health')).status_code == 200
asyncio.run(check_health())
doc = openapi_document(app)
assert doc['info']['version'] == maitian_agent.__version__
export_openapi(os.environ['OPENAPI_TARGET'])
assert json.load(open(os.environ['OPENAPI_TARGET']))['info']['version'] == maitian_agent.__version__
print(json.dumps({'version': doc['info']['version']}))
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={
            **env,
            "PYTHONPATH": str(extract),
            "OPENAPI_TARGET": str(tmp_path / "wheel-openapi.json"),
        },
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr

    package_only = tmp_path / "package-only"
    package_only.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        for entry in archive.namelist():
            if entry.startswith("maitian_agent/"):
                archive.extract(entry, package_only)
    fallback_script = """
import asyncio
import httpx
import maitian_agent
import maitian_agent.openapi as openapi
from importlib.metadata import PackageNotFoundError
def missing_distribution(_name):
    raise PackageNotFoundError
openapi.distribution_version = missing_distribution
assert openapi.application_version() == maitian_agent.__version__
from maitian_agent.api.routes import create_app
app = create_app()
async def check_health():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url='http://test'
    ) as client:
        assert (await client.get('/health')).status_code == 200
asyncio.run(check_health())
assert openapi.openapi_document(app)['info']['version'] == maitian_agent.__version__
"""
    fallback = subprocess.run(
        [sys.executable, "-c", fallback_script],
        cwd=tmp_path,
        env={**env, "PYTHONPATH": str(package_only)},
        text=True,
        capture_output=True,
    )
    assert fallback.returncode == 0, fallback.stderr
