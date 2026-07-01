from __future__ import annotations

import pytest

from postsynth.config import OpenRouterConfig
from postsynth.models import (
    DEFAULT_MODEL_ID,
    ModelSpec,
    filter_models,
    resolve_model_selection,
)


def test_default_config_uses_pinned_gemini_flash() -> None:
    assert OpenRouterConfig().model == DEFAULT_MODEL_ID
    assert DEFAULT_MODEL_ID == "google/gemini-3.5-flash-20260519"


def test_model_preset_resolution_records_reproducibility_metadata() -> None:
    selection = resolve_model_selection(model_preset="default")

    assert selection.model == DEFAULT_MODEL_ID
    assert selection.floating is False
    assert selection.metadata()["floating_model"] is False
    assert selection.metadata()["preset"] == "default"


def test_latest_preset_is_marked_floating() -> None:
    selection = resolve_model_selection(model_preset="latest-gemini-flash")

    assert selection.model == "~google/gemini-flash-latest"
    assert selection.floating is True


def test_model_and_preset_conflict() -> None:
    with pytest.raises(ValueError):
        resolve_model_selection(model="google/gemini-3.5-flash", model_preset="default")


def test_filter_models_by_provider_and_text_output() -> None:
    catalog = [
        ModelSpec.model_validate(
            {
                "id": "google/gemini-3.5-flash",
                "name": "Gemini",
                "architecture": {"output_modalities": ["text"]},
            }
        ),
        ModelSpec.model_validate(
            {
                "id": "google/image-model",
                "name": "Image",
                "architecture": {"output_modalities": ["image"]},
            }
        ),
        ModelSpec.model_validate(
            {
                "id": "openai/gpt",
                "name": "GPT",
                "architecture": {"output_modalities": ["text"]},
            }
        ),
    ]

    filtered = filter_models(catalog, provider="google", text_only=True)

    assert [model.id for model in filtered] == ["google/gemini-3.5-flash"]
