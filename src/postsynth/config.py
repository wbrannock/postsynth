from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from postsynth.models import DEFAULT_MODEL_ID


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str = DEFAULT_MODEL_ID
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    app_title: str = "postsynth"
    timeout_seconds: float = 60.0
    max_retries: int = 2
    requested_model: str | None = None
    model_source: str = "preset"
    model_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class GenerationConfig:
    count: int
    batch_size: int = 25
    max_batches: int | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    repair_attempts: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.max_batches is not None and self.max_batches < 1:
            raise ValueError("max_batches must be at least 1")
        if self.repair_attempts < 0:
            raise ValueError("repair_attempts cannot be negative")


@dataclass(frozen=True)
class OutputConfig:
    path: Path
    write_dataset_card: bool = True
