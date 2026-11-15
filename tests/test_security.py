"""Offline regression tests for upload, path, CORS and error boundaries."""
import asyncio
import os
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from PIL import Image

from maitian_agent.agents.wisdom_transfer import WisdomTransferAgent
from maitian_agent.api import routes
from maitian_agent.config.settings import get_settings
from maitian_agent.tools.file import FileProcessor
from maitian_agent.tools.upload import delete_saved_upload


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


def image_bytes(extension="png"):
    buffer = BytesIO()
    Image.new("RGB", (1, 1)).save(buffer, format=extension.upper())
    return buffer.getvalue()


def png():
    return image_bytes()


@pytest.mark.parametrize("extension", ["png", "jpeg", "webp"])
def test_valid_upload(api, extension):
    client, reader, directory = api
    def read(path, **kwargs):
        assert Path(path).read_bytes() == image_bytes(extension)
        return [(None, "教案", 0.99)]

    reader.readtext.side_effect = read
    for _ in range(2):
        response = client.post("/api/wisdom-transfer", files={"file": (f"original.{extension}", image_bytes(extension), f"image/{extension}")})
        assert response.status_code == 200
        assert "结构化教案" in response.json()["result"]
    paths = [Path(call.args[0]) for call in reader.readtext.call_args_list]
    assert len(set(paths)) == 2
    for path in paths:
        assert path.parent == directory.resolve()
        assert path.name != f"original.{extension}"
        assert len(path.stem) == 32
        assert not path.exists()
    assert not list(directory.iterdir())


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


@pytest.mark.parametrize("root_path", ["", "/prefix"])
@pytest.mark.parametrize("declared", [None, "99999999", "1"])
def test_upload_body_limit_before_save(api, monkeypatch, declared, root_path):
    client, reader, _ = api
    save = AsyncMock()
    monkeypatch.setattr(routes, "save_image_upload", save)
    headers = [(b"content-type", b"multipart/form-data; boundary=test")]
    if declared is not None:
        headers.append((b"content-length", declared.encode()))
    chunks = iter([
        b'--test\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n\r\n',
        *([b"x" * 65536] * 24), b"\r\n--test--\r\n",
    ])
    sent, received = [], []

    async def receive():
        chunk = next(chunks)
        received.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": chunk != b"\r\n--test--\r\n"}

    async def send(message):
        sent.append(message)

    asyncio.run(client.app({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": root_path + "/api/wisdom-transfer", "root_path": root_path,
        "raw_path": (root_path + "/api/wisdom-transfer").encode(), "query_string": b"", "headers": headers,
        "client": ("test", 123), "server": ("test", 80),
    }, receive, send))
    assert sent[0]["status"] == 413
    assert sum(message["type"] == "http.response.start" for message in sent) == 1
    if declared == "99999999":
        assert received == []
    else:
        assert sum(received) < 24 * 65536
    save.assert_not_called()
    reader.readtext.assert_not_called()


def test_large_pixel_image_rejected(api):
    client, reader, directory = api
    buffer = BytesIO()
    Image.new("1", (10000, 10000)).save(buffer, format="PNG")
    response = client.post("/api/wisdom-transfer", files={"file": ("bomb.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 413
    reader.readtext.assert_not_called()
    assert not directory.exists() or not list(directory.iterdir())


def test_failed_upload_deleted(api, monkeypatch):
    client, reader, directory = api
    reader.readtext.side_effect = RuntimeError("OCR failed")
    response = client.post("/api/wisdom-transfer", files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 500
    assert not Path(reader.readtext.call_args.args[0]).exists()
    assert not list(directory.iterdir())


def test_startup_cleans_stale_uploads(api, monkeypatch):
    _, _, directory = api
    directory.mkdir()
    stale, fresh = directory / "stale.png", directory / "fresh.png"
    stale.write_bytes(png())
    fresh.write_bytes(png())
    os.utime(stale, (time.time() - 48 * 3600,) * 2)
    monkeypatch.setattr(routes, "startup_event", AsyncMock())
    with TestClient(routes.create_app()) as client:
        assert client.get("/health").status_code == 200
        assert not stale.exists()
        assert fresh.exists()


def test_exact_file_limit_allows_multipart_overhead(api):
    client, reader, _ = api
    content = png().ljust(1024 * 1024, b"\0")
    response = client.post("/api/wisdom-transfer", files={"file": ("x.png", content, "image/png")})
    assert response.status_code == 200
    assert not Path(reader.readtext.call_args.args[0]).exists()


def test_pixel_limit_checked_before_decode(api, monkeypatch):
    client, reader, _ = api
    buffer = BytesIO()
    Image.new("1", (3000, 3000)).save(buffer, format="PNG")
    load = Mock(side_effect=AssertionError("must reject before decoding"))
    monkeypatch.setattr(Image.Image, "load", load)
    response = client.post("/api/wisdom-transfer", files={"file": ("x.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 413
    load.assert_not_called()
    reader.readtext.assert_not_called()


def test_delete_failure_preserves_result(api, monkeypatch, caplog):
    client, _, _ = api
    monkeypatch.setattr(routes, "delete_saved_upload", Mock(side_effect=OSError("disk error")))
    response = client.post("/api/wisdom-transfer", files={"file": ("x.png", png(), "image/png")})
    assert response.status_code == 200
    assert "删除本次上传文件失败" in caplog.text


@pytest.mark.parametrize("extension", ["png", "jpeg", "webp"])
def test_agent_exception_upload_deleted(api, monkeypatch, extension):
    client, _, directory = api
    failing = Mock()
    failing.run.side_effect = RuntimeError("agent failed")
    monkeypatch.setitem(routes.agents, "wisdom_transfer", failing)
    response = client.post("/api/wisdom-transfer", files={"file": (f"x.{extension}", image_bytes(extension), f"image/{extension}")})
    assert response.status_code == 500
    assert not list(directory.iterdir())


@pytest.mark.parametrize("kind", ["parent", "absolute", "symlink"])
def test_delete_saved_upload_boundaries(tmp_path, kind):
    directory = tmp_path / "uploads"
    directory.mkdir()
    outside = tmp_path / "uploads-other" / "outside.webp"
    outside.parent.mkdir()
    outside.write_bytes(b"original")
    link = directory / "link.webp"
    link.symlink_to(outside)
    paths = {
        "parent": str(directory / ".." / "uploads-other" / "outside.webp"),
        "absolute": str(outside),
        "symlink": str(link),
    }
    with pytest.raises(ValueError):
        delete_saved_upload(paths[kind], str(directory))
    assert outside.read_bytes() == b"original"
