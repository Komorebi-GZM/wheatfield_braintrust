"""Offline smoke checks for the installed application entry points."""

import importlib

from click.testing import CliRunner


def test_api_routes_import():
    importlib.import_module("maitian_agent.api.routes")


def test_cli_help():
    from maitian_agent.cli import cli

    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Usage:" in result.output
    assert "--help" in result.output


def test_api_startup_without_ocr_dependencies(monkeypatch):
    import sys

    from fastapi.testclient import TestClient

    from maitian_agent.api.routes import app

    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "wisdom_transfer" in response.json()["agents"]

    assert "easyocr" not in sys.modules
    assert "torch" not in sys.modules
