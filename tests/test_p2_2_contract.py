"""P2-2 contract acceptance tests using injected fake agents and models."""

from io import BytesIO
from unittest.mock import Mock

from click.testing import CliRunner
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from PIL import Image

from maitian_agent.agents.quick_lesson_prep import QuickLessonPrepAgent
from maitian_agent.agents.results import AgentResult
from maitian_agent.agents.router import RouterAgent
from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.api import routes
from maitian_agent.config.settings import get_settings


def test_v1_has_explicit_paths_operations_and_concrete_success_models(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    schema = routes.create_app().openapi()
    expected = {
        "/api/v1/lesson-prep": "lessonPrep",
        "/api/v1/wisdom-transfer": "wisdomTransfer",
        "/api/v1/quiz": "generateQuiz",
        "/api/v1/material": "recommendMaterial",
        "/api/v1/meeting-notes": "processMeetingNotes",
        "/api/v1/route": "routeIntent",
    }
    for path, operation_id in expected.items():
        operation = schema["paths"][path]["post"]
        assert operation["operationId"] == operation_id
        success = operation["responses"]["200"]["content"]["application/json"]["schema"]
        assert "$ref" in success
    assert (
        len(
            {
                operation["post"]["operationId"]
                for path, item in schema["paths"].items()
                for operation in [item]
                if "post" in operation
            }
        )
        == 12
    )


def test_blank_route_is_a_safe_422_v1_envelope(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://trusted.test")
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    response = client.post(
        "/api/v1/route",
        json={"user_input": "   "},
        headers={"Origin": "http://trusted.test"},
    )
    body = response.json()
    assert response.status_code == 422
    assert body["status"] == "failed"
    assert body["data"] is None
    assert body["error"]["code"] == "VALIDATION_ERROR"
    assert body["error"]["message"]
    assert body["meta"]["version"]
    assert body["meta"]["duration_ms"] >= 0
    assert body["meta"]["model"] is None
    assert response.headers["access-control-allow-origin"] == "http://trusted.test"


def test_v1_validation_error_does_not_echo_input(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    response = client.post("/api/v1/route", json={"user_input": ["SECRET_INPUT"]})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert "SECRET_INPUT" not in response.text
    assert "ctx" not in response.text


def test_legacy_route_keeps_detail_shape_and_rejects_blank_input(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    response = client.post("/api/route", json={"user_input": "  "})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_schema_export_is_a_deterministic_root_command():
    from maitian_agent.openapi import export_openapi, openapi_document

    assert callable(export_openapi)
    document = openapi_document()
    assert {"/", "/health"}.issubset(document["paths"])
    assert sum("post" in item for item in document["paths"].values()) == 12
    assert all(
        item["post"]["deprecated"]
        for path, item in document["paths"].items()
        if path.startswith("/api/") and not path.startswith("/api/v1/")
    )


def test_quick_lesson_agent_returns_typed_result():
    agent = QuickLessonPrepAgent(llm=FakeListChatModel(responses=["教案正文"]))
    result = agent.run(subject="语文", grade="三年级", topic="春天")
    assert isinstance(result, AgentResult)
    assert result.status == "success"
    assert result.data.text == "教案正文"


def test_router_model_failure_propagates():
    class FailingModel(FakeListChatModel):
        def _call(self, *args, **kwargs):
            raise RuntimeError("SECRET_MODEL_FAILURE")

    agent = RouterAgent(llm=FailingModel(responses=[]))
    try:
        agent.run({"user_input": "帮我备课"})
    except RuntimeError as exc:
        assert "SECRET_MODEL_FAILURE" in str(exc)
    else:
        raise AssertionError("model failure must propagate")


def test_ocr_empty_result_is_needs_review_without_model_call():
    reader = Mock()
    reader.readtext.return_value = []
    agent = WisdomTransferAgent(ocr_reader=reader)
    result = agent.run("image.png")
    assert result.status == "needs_review"
    assert result.data.text == ""
    assert result.data.review_reasons == ["OCR_NO_TEXT"]


def _png_bytes() -> bytes:
    image = BytesIO()
    Image.new("RGB", (1, 1)).save(image, format="PNG")
    return image.getvalue()


def test_v1_and_legacy_413_keep_compatible_shapes(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://trusted.test")
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    huge = b"x" * (2 * 1024 * 1024)
    headers = {"Origin": "http://trusted.test", "Content-Length": str(len(huge) + 200)}
    legacy = client.post(
        "/api/wisdom-transfer",
        files={"file": ("big.png", huge, "image/png")},
        headers=headers,
    )
    assert legacy.status_code == 413
    assert legacy.json() == {"detail": "上传内容过大，请上传大小限制以内的图片"}
    assert legacy.headers["access-control-allow-origin"] == "http://trusted.test"

    v1 = client.post(
        "/api/v1/wisdom-transfer",
        files={"file": ("big.png", huge, "image/png")},
        headers=headers,
    )
    body = v1.json()
    assert v1.status_code == 413
    assert body["status"] == "failed"
    assert body["data"] is None
    assert body["error"]["code"] == "UPLOAD_TOO_LARGE"
    assert "SECRET" not in v1.text
    assert v1.headers["access-control-allow-origin"] == "http://trusted.test"


def test_v1_llm_not_configured_uses_envelope(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    response = client.post(
        "/api/v1/lesson-prep",
        json={"subject": "语文", "grade": "三年级", "topic": "春天"},
    )
    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "failed"
    assert body["data"] is None
    assert body["error"]["code"] == "LLM_NOT_CONFIGURED"
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["llm_configured"] is False


def test_ocr_with_text_is_needs_review():
    reader = Mock()
    reader.readtext.return_value = [((0, 0), "课堂导入", 0.9)]
    agent = WisdomTransferAgent(ocr_reader=reader, llm=FakeListChatModel(responses=["结构化教案"]))
    result = agent.run("image.png")
    assert result.status == "needs_review"
    assert "课堂导入" in result.data.text
    assert result.data.review_reasons == ["OCR_REQUIRES_REVIEW"]


def test_ocr_execution_error_is_not_needs_review():
    reader = Mock()
    reader.readtext.side_effect = RuntimeError("OCR engine crashed")
    agent = WisdomTransferAgent(ocr_reader=reader)
    try:
        agent.run("image.png")
    except RuntimeError as exc:
        assert "OCR" in str(exc)
    else:
        raise AssertionError("OCR execution errors must not be converted to needs_review")


def test_cli_failure_exits_without_traceback(monkeypatch, tmp_path):
    from maitian_agent.cli import cli

    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["lesson-prep", "-s", "语文", "-g", "三年级", "-t", "春天"])
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert result.output


def test_cli_needs_review_exits_zero(monkeypatch, tmp_path):
    from maitian_agent.agents.results import AgentResult, WisdomData
    from maitian_agent.cli import cli

    class FakeAgent:
        def run(self, image_path):
            return AgentResult(
                status="needs_review",
                data=WisdomData(text="识别草稿", review_reasons=["OCR_REQUIRES_REVIEW"]),
            )

    image = tmp_path / "lesson.png"
    image.write_bytes(_png_bytes())
    monkeypatch.setattr("maitian_agent.cli.WisdomTransferAgent", lambda: FakeAgent())
    runner = CliRunner()
    result = runner.invoke(cli, ["wisdom-transfer", "--image", str(image)])
    assert result.exit_code == 0
    assert "待校订" in result.output
