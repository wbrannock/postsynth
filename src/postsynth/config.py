from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenRouterConfig:
    model: str = "openai/gpt-4o-mini"
    api_key: str | None = None
    base_url: str = "https://openrouter.ai/api/v1"
    http_referer: str | None = None
    app_title: str = "postsynth"
    timeout_seconds: float = 60.0
    max_retries: int = 2


@dataclass(frozen=True)
class GenerationConfig:
    count: int
    temperature: float = 0.7
    max_tokens: int = 4096
    repair_attempts: int = 1

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1")
        if self.repair_attempts < 0:
            raise ValueError("repair_attempts cannot be negative")


@dataclass(frozen=True)
class OutputConfig:
    path: Path
    write_dataset_card: bool = True

