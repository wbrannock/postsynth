from __future__ import annotations

from typer.testing import CliRunner

import postsynth.cli as cli
from postsynth.cli import app
from postsynth.models import ModelSpec


def test_cli_requires_one_source() -> None:
    result = CliRunner().invoke(app, ["generate", "sft", "--count", "1"])

    assert result.exit_code != 0
    assert "Provide exactly one of --seed or --examples" in result.output


def test_cli_rejects_model_and_model_preset() -> None:
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "sft",
            "--seed",
            "math",
            "--model",
            "google/gemini-3.5-flash",
            "--model-preset",
            "default",
        ],
    )

    assert result.exit_code != 0
    assert "Use either --model or --model-preset" in result.output


def test_cli_rejects_non_positive_concurrency() -> None:
    result = CliRunner().invoke(
        app,
        ["generate", "sft", "--seed", "math", "--concurrency", "0"],
    )

    assert result.exit_code != 0


def test_models_presets_command() -> None:
    result = CliRunner().invoke(app, ["models", "presets"])

    assert result.exit_code == 0
    assert "default" in result.output
    assert "google/gemini-3.5-flash-20260519" in result.output


def test_models_list_command(monkeypatch) -> None:
    monkeypatch.setattr(
        cli,
        "fetch_openrouter_models",
        lambda force_refresh=False: [
            ModelSpec.model_validate(
                {
                    "id": "google/gemini-3.5-flash",
                    "name": "Google: Gemini 3.5 Flash",
                    "context_length": 1048576,
                    "architecture": {"output_modalities": ["text"]},
                    "pricing": {"prompt": "0.0000015", "completion": "0.000009"},
                }
            )
        ],
    )

    result = CliRunner().invoke(
        app,
        ["models", "list", "--provider", "google", "--text-only"],
    )

    assert result.exit_code == 0
    assert "google/gemini-3.5-flash" in result.output
