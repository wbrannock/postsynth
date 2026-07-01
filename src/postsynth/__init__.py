"""Synthetic post-training dataset generation for TRL."""

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.generator import (
    GenerationResult,
    Synthesizer,
    generate_dpo,
    generate_grpo,
    generate_kto,
    generate_sft,
)

__all__ = [
    "GenerationConfig",
    "GenerationResult",
    "OpenRouterConfig",
    "OutputConfig",
    "Synthesizer",
    "generate_dpo",
    "generate_grpo",
    "generate_kto",
    "generate_sft",
]
