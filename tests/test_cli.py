from __future__ import annotations

from typer.testing import CliRunner

from postsynth.cli import app


def test_cli_requires_one_source() -> None:
    result = CliRunner().invoke(app, ["generate", "sft", "--count", "1"])

    assert result.exit_code != 0
    assert "Provide exactly one of --seed or --examples" in result.output

