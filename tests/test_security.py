"""Offline regression tests for upload, path, CORS and error boundaries."""
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from PIL import Image

from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.api import routes
from maitian_agent.config.settings import get_settings
from maitian_agent.tools.file import FileProcessor


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://trusted.test")
    monkeypatch.setenv("LLM_BASE_URL", "http://fake.invalid/v1")
    monkeypatch.setenv("LLM_MODEL", "fake")
    monkeypatch.setenv("LLM_API_KEY", "placeholder")
    get_settings.cache_clear()
    reader = Mock()
    reader.readtext.return_value = [(None, "教案", 0.99)]
    agent = WisdomTransferAgent(llm=FakeListChatModel(responses=["结构化教案"]), ocr_reader=reader)
    monkeypatch.setattr(routes, "agents", {"wisdom_transfer": agent})
    # No lifespan here: use explicitly injected real Agent instead of network clients.
    client = TestClient(routes.create_app())
    yield client, reader, tmp_path / "uploads"
    client.close()
    get_settings.cache_clear()


def png():
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_valid_upload(api):
    client, reader, directory = api
    for _ in range(2):
        response = client.post("/api/wisdom-transfer", files={"file": ("original.png", png(), "image/png")})
        assert response.status_code == 200
        assert "结构化教案" in response.json()["result"]
    paths = [Path(call.args[0]) for call in reader.readtext.call_args_list]
    assert len(set(paths)) == 2
    for path in paths:
        assert path.parent == directory.resolve()
        assert path.name != "original.png"
        assert len(path.stem) == 32
        assert path.read_bytes() == png()


@pytest.mark.parametrize(("name", "content", "mime", "status"), [
    ("large.png", b"x" * (1024 * 1024 + 1), "image/png", 413),
    ("bad.txt", b"text", "image/png", 415),
    ("fake.png", b"not an image", "image/png", 415),
    ("magic.png", b"\x89PNG\r\n\x1a\nforged", "image/png", 415),
])
def test_invalid_upload(api, name, content, mime, status):
    client, reader, directory = api
    response = client.post("/api/wisdom-transfer", files={"file": (name, content, mime)})
    assert response.status_code == status
    reader.readtext.assert_not_called()
    assert not directory.exists()


def test_server_path_removed(api):
    client, reader, _ = api
    assert client.post("/api/wisdom-transfer", json={"image_path": "/etc/passwd"}).status_code == 422
    reader.readtext.assert_not_called()


@pytest.mark.parametrize("operation", ["save_bytes", "save_upload", "delete_file"])
@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink", "extension"])
def test_write_boundaries(tmp_path, operation, kind):
    base = tmp_path / "uploads"
    processor = FileProcessor(str(base), ["png"])
    outside = tmp_path / "outside.png"
    outside.write_bytes(b"original")
    (base / "link.png").symlink_to(outside)
    filename = {"parent": "../outside.png", "absolute": str(outside),
                "symlink": "link.png", "extension": "bad.exe"}[kind]
    with pytest.raises(ValueError):
        if operation == "save_bytes":
            processor.save_bytes(b"changed", filename)
        elif operation == "save_upload":
            processor.save_upload(BytesIO(b"changed"), filename)
        else:
            processor.delete_file(filename)
    assert outside.read_bytes() == b"original"
    assert not (base / "bad.exe").exists()


def test_cors(api):
    client, _, _ = api
    for origin, allowed in [("http://trusted.test", True), ("http://evil.test", False)]:
        response = client.get("/health", headers={"Origin": origin})
        assert (response.headers.get("access-control-allow-origin") == origin) is allowed
        assert "access-control-allow-credentials" not in response.headers


def test_cors_reloads_settings(api, monkeypatch):
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://new.test")
    get_settings.cache_clear()
    client = TestClient(routes.create_app())
    assert client.get("/health", headers={"Origin": "http://new.test"}).headers["access-control-allow-origin"] == "http://new.test"
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "http://trusted.test"}).headers
    client.close()


def test_error_hidden(api, monkeypatch):
    client, _, _ = api
    failing = Mock()
    failing.run.side_effect = RuntimeError("SECRET_TRACE_xyz")
    monkeypatch.setitem(routes.agents, "router", failing)
    response = client.post("/api/route", json={"user_input": "测试"})
    assert response.status_code == 500
    assert response.json()["detail"] == "服务内部错误，请稍后重试"
    assert "SECRET_TRACE_xyz" not in response.text


def test_legacy_agent_error_hidden(api):
    client, reader, _ = api
    reader.readtext.side_effect = RuntimeError("SECRET_TRACE_xyz")
    response = client.post("/api/wisdom-transfer", files={"file": ("lesson.png", png(), "image/png")})
    assert response.status_code == 500
    assert "SECRET_TRACE_xyz" not in response.text


def test_default_local_origins(monkeypatch, tmp_path):
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        client = TestClient(routes.create_app())
        for origin in ["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:5173"]:
            assert client.get("/health", headers={"Origin": origin}).headers["access-control-allow-origin"] == origin
        assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://untrusted.test"}).headers
        client.close()
    finally:
        get_settings.cache_clear()


def test_valid_file_writes(tmp_path):
    processor = FileProcessor(str(tmp_path), ["png"])
    path = Path(processor.save_bytes(b"first", "bytes.png"))
    assert path.read_bytes() == b"first"
    upload = BytesIO(b"second")
    upload.name = "original.png"
    saved = Path(processor.save_upload(upload))
    assert saved.parent == tmp_path.resolve()
    assert saved.name != upload.name
    assert saved.read_bytes() == b"second"
    assert processor.delete_file(str(saved))
