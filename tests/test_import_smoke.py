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
