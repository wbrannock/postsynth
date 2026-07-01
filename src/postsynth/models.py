from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_MODEL_ID = "google/gemini-3.5-flash-20260519"
MODEL_CACHE_TTL_SECONDS = 60 * 60


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    canonical_slug: str | None = None
    name: str
    context_length: int | None = None
    architecture: dict[str, Any] | None = None
    pricing: dict[str, Any] | None = None
    supported_parameters: list[str] = Field(default_factory=list)
    fetched_at: str | None = None

    @property
    def text_output(self) -> bool:
        output_modalities = (self.architecture or {}).get("output_modalities") or []
        return "text" in output_modalities


@dataclass(frozen=True)
class ModelPreset:
    name: str
    model: str
    description: str
    floating: bool = False


@dataclass(frozen=True)
class ModelSelection:
    requested: str
    model: str
    source: str
    floating: bool
    preset: str | None = None
    spec: ModelSpec | None = None

    def metadata(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "requested": self.requested,
            "resolved": self.model,
            "source": self.source,
            "preset": self.preset,
            "floating_model": self.floating,
        }
        if self.spec:
            data.update(
                {
                    "id": self.spec.id,
                    "canonical_slug": self.spec.canonical_slug,
                    "name": self.spec.name,
                    "context_length": self.spec.context_length,
                    "pricing": self.spec.pricing,
                    "supported_parameters": self.spec.supported_parameters,
                    "catalog_fetched_at": self.spec.fetched_at,
                }
            )
        return data


MODEL_PRESETS: dict[str, ModelPreset] = {
    "default": ModelPreset(
        name="default",
        model=DEFAULT_MODEL_ID,
        description="Pinned Gemini 3.5 Flash for reproducible synthetic data generation.",
    ),
    "cheap": ModelPreset(
        name="cheap",
        model="google/gemini-3.1-flash-lite-20260507",
        description="Pinned low-cost Gemini Flash Lite model.",
    ),
    "quality": ModelPreset(
        name="quality",
        model="google/gemini-2.5-pro",
        description="Pinned Gemini Pro model for higher-quality generations.",
    ),
    "latest-gemini-flash": ModelPreset(
        name="latest-gemini-flash",
        model="~google/gemini-flash-latest",
        description="Floating OpenRouter alias for the latest Gemini Flash model.",
        floating=True,
    ),
}


def resolve_model_selection(
    *,
    model: str | None = None,
    model_preset: str | None = "default",
    catalog: list[ModelSpec] | None = None,
) -> ModelSelection:
    if model and model_preset:
        raise ValueError("Use either model or model_preset, not both")
    if model:
        spec = find_model(catalog or [], model)
        return ModelSelection(
            requested=model,
            model=model,
            source="model",
            floating=_is_floating_model(model),
            spec=spec,
        )

    preset_name = model_preset or "default"
    try:
        preset = MODEL_PRESETS[preset_name]
    except KeyError as error:
        known = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(f"Unknown model preset '{preset_name}'. Known presets: {known}") from error

    spec = find_model(catalog or [], preset.model)
    return ModelSelection(
        requested=preset_name,
        model=preset.model,
        source="preset",
        floating=preset.floating or _is_floating_model(preset.model),
        preset=preset.name,
        spec=spec,
    )


def find_model(catalog: list[ModelSpec], model: str) -> ModelSpec | None:
    for spec in catalog:
        if model in {spec.id, spec.canonical_slug}:
            return spec
    return None


def fetch_openrouter_models(
    *,
    base_url: str = "https://openrouter.ai/api/v1",
    force_refresh: bool = False,
    ttl_seconds: int = MODEL_CACHE_TTL_SECONDS,
) -> list[ModelSpec]:
    cache_path = _model_cache_path()
    cached = _read_cached_models(cache_path, ttl_seconds=ttl_seconds)
    if cached is not None and not force_refresh:
        return cached

    url = base_url.rstrip("/") + "/models"
    request = Request(url, headers={"User-Agent": "postsynth"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as error:
        if cached is not None:
            return cached
        raise RuntimeError(f"Could not fetch OpenRouter model catalog: {error}") from error

    fetched_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    raw_models = payload.get("data", [])
    specs = [
        ModelSpec.model_validate({**model, "fetched_at": fetched_at})
        for model in raw_models
        if isinstance(model, dict)
    ]
    _write_cached_models(cache_path, specs)
    return specs


def filter_models(
    catalog: list[ModelSpec],
    *,
    provider: str | None = None,
    text_only: bool = False,
) -> list[ModelSpec]:
    models = catalog
    if provider:
        prefix = provider.rstrip("/") + "/"
        models = [model for model in models if model.id.startswith(prefix)]
    if text_only:
        models = [model for model in models if model.text_output]
    return models


def _is_floating_model(model: str) -> bool:
    return model.startswith("~") or model.endswith("-latest")


def _model_cache_path() -> Path:
    cache_home = os.getenv("XDG_CACHE_HOME")
    base = Path(cache_home) if cache_home else Path.home() / ".cache"
    return base / "postsynth" / "openrouter-models.json"


def _read_cached_models(path: Path, *, ttl_seconds: int) -> list[ModelSpec] | None:
    try:
        if time.time() - path.stat().st_mtime > ttl_seconds:
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    models = payload.get("data", [])
    if not isinstance(models, list):
        return None
    return [ModelSpec.model_validate(model) for model in models if isinstance(model, dict)]


def _write_cached_models(path: Path, specs: list[ModelSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data": [spec.model_dump() for spec in specs]}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
