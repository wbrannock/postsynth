from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.json_utils import parse_json_object
from postsynth.llm import LLMClient, OpenRouterClient
from postsynth.models import resolve_model_selection
from postsynth.prompts import build_generation_messages, build_repair_messages
from postsynth.schemas import DatasetKind, GeneratedItems, ROW_ADAPTERS
from postsynth.writers import write_dataset_card, write_jsonl


@dataclass(frozen=True)
class GenerationResult:
    kind: DatasetKind
    output_path: Path
    rows: int
    requested: int
    invalid_rows: int
    repair_attempted: int
    repair_succeeded: int
    dropped_rows: int
    batches_attempted: int
    incomplete: bool
    batch_diagnostics: list[dict[str, Any]]


class Synthesizer:
    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        provider_config: OpenRouterConfig | None = None,
    ) -> None:
        config = provider_config or OpenRouterConfig()
        if config.model_metadata is None:
            selection = resolve_model_selection(
                model=config.model if provider_config else None,
                model_preset=None if provider_config else "default",
            )
            config = replace(
                config,
                model=selection.model,
                requested_model=selection.requested,
                model_source=selection.source,
                model_metadata=selection.metadata(),
            )
        self.provider_config = config
        self.llm_client = llm_client or OpenRouterClient(self.provider_config)

    def generate(
        self,
        kind: DatasetKind,
        *,
        seed: str | None = None,
        examples: list[dict[str, Any]] | None = None,
        generation_config: GenerationConfig,
        output_config: OutputConfig,
        progress_callback: Callable[[int], None] | None = None,
    ) -> GenerationResult:
        _validate_source(seed=seed, examples=examples)
        source_mode = "seed" if seed is not None else "examples"
        rows: list[BaseModel] = []
        stats = {
            "invalid_rows": 0,
            "repair_attempted": 0,
            "repair_succeeded": 0,
            "dropped_rows": 0,
            "batches_attempted": 0,
            "incomplete": 0,
        }
        batch_diagnostics: list[dict[str, Any]] = []
        max_batches = generation_config.max_batches or _default_max_batches(
            generation_config.count,
            generation_config.batch_size,
        )

        while len(rows) < generation_config.count and stats["batches_attempted"] < max_batches:
            remaining = generation_config.count - len(rows)
            batch_count = min(remaining, generation_config.batch_size)
            batch_config = replace(generation_config, count=batch_count)
            batch_index = stats["batches_attempted"] + 1
            messages = build_generation_messages(
                kind,
                count=batch_count,
                seed=seed,
                examples=examples,
            )
            response = self.llm_client.complete(
                messages,
                model=self.provider_config.model,
                temperature=generation_config.temperature,
                max_tokens=generation_config.max_tokens,
            )
            batch_rows, batch_stats = self._parse_validate_repair(
                kind,
                response=response,
                generation_config=batch_config,
            )
            rows.extend(batch_rows)
            for key in ("invalid_rows", "repair_attempted", "repair_succeeded", "dropped_rows"):
                stats[key] += batch_stats[key]
            stats["batches_attempted"] += 1
            batch_diagnostics.append(
                {
                    "batch": batch_index,
                    "requested": batch_count,
                    "accepted": len(batch_rows),
                    "dropped": batch_stats["dropped_rows"],
                    "invalid_rows": batch_stats["invalid_rows"],
                    "repair_attempted": batch_stats["repair_attempted"],
                    "repair_succeeded": batch_stats["repair_succeeded"],
                    "error_stage": batch_stats["error_stage"],
                    "error_summary": batch_stats["error_summary"],
                }
            )
            if progress_callback:
                progress_callback(len(batch_rows))

        rows = rows[: generation_config.count]
        if len(rows) < generation_config.count:
            stats["incomplete"] = 1
        write_jsonl(output_config.path, rows)
        if output_config.write_dataset_card:
            write_dataset_card(
                output_config.path,
                kind=kind,
                rows_written=len(rows),
                source_mode=source_mode,
                provider_config=self.provider_config,
                generation_config=generation_config,
                validation_stats=stats,
                batch_diagnostics=batch_diagnostics,
            )
        return GenerationResult(
            kind=kind,
            output_path=output_config.path,
            rows=len(rows),
            requested=generation_config.count,
            invalid_rows=stats["invalid_rows"],
            repair_attempted=stats["repair_attempted"],
            repair_succeeded=stats["repair_succeeded"],
            dropped_rows=stats["dropped_rows"],
            batches_attempted=stats["batches_attempted"],
            incomplete=bool(stats["incomplete"]),
            batch_diagnostics=batch_diagnostics,
        )

    def _parse_validate_repair(
        self,
        kind: DatasetKind,
        *,
        response: str,
        generation_config: GenerationConfig,
    ) -> tuple[list[BaseModel], dict[str, Any]]:
        rows, invalid = _parse_and_validate_rows(kind, response)
        repair_attempted = 0
        repair_succeeded = 0

        if invalid and generation_config.repair_attempts > 0:
            repair_attempted = 1
            repair_messages = build_repair_messages(
                kind,
                invalid_output=response,
                validation_error=_format_invalid_rows(invalid),
                count=generation_config.count,
            )
            repaired = self.llm_client.complete(
                repair_messages,
                model=self.provider_config.model,
                temperature=0.0,
                max_tokens=generation_config.max_tokens,
            )
            rows, invalid = _parse_and_validate_rows(kind, repaired)
            repair_succeeded = 1 if not invalid else 0

        rows = rows[: generation_config.count]
        dropped_rows = max(0, generation_config.count - len(rows))
        stats = {
            "invalid_rows": len(invalid),
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "dropped_rows": dropped_rows,
            "error_stage": _error_stage(invalid),
            "error_summary": _error_summary(invalid),
        }
        return rows, stats


def generate_sft(
    *,
    seed: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    output_path: str | Path,
    count: int,
    batch_size: int = 25,
    max_batches: int | None = None,
    provider_config: OpenRouterConfig | None = None,
    llm_client: LLMClient | None = None,
) -> GenerationResult:
    return _generate_kind(
        "sft",
        seed=seed,
        examples=examples,
        output_path=output_path,
        count=count,
        batch_size=batch_size,
        max_batches=max_batches,
        provider_config=provider_config,
        llm_client=llm_client,
    )


def generate_dpo(
    *,
    seed: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    output_path: str | Path,
    count: int,
    batch_size: int = 25,
    max_batches: int | None = None,
    provider_config: OpenRouterConfig | None = None,
    llm_client: LLMClient | None = None,
) -> GenerationResult:
    return _generate_kind(
        "dpo",
        seed=seed,
        examples=examples,
        output_path=output_path,
        count=count,
        batch_size=batch_size,
        max_batches=max_batches,
        provider_config=provider_config,
        llm_client=llm_client,
    )


def generate_grpo(
    *,
    seed: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    output_path: str | Path,
    count: int,
    batch_size: int = 25,
    max_batches: int | None = None,
    provider_config: OpenRouterConfig | None = None,
    llm_client: LLMClient | None = None,
) -> GenerationResult:
    return _generate_kind(
        "grpo",
        seed=seed,
        examples=examples,
        output_path=output_path,
        count=count,
        batch_size=batch_size,
        max_batches=max_batches,
        provider_config=provider_config,
        llm_client=llm_client,
    )


def generate_kto(
    *,
    seed: str | None = None,
    examples: list[dict[str, Any]] | None = None,
    output_path: str | Path,
    count: int,
    batch_size: int = 25,
    max_batches: int | None = None,
    provider_config: OpenRouterConfig | None = None,
    llm_client: LLMClient | None = None,
) -> GenerationResult:
    return _generate_kind(
        "kto",
        seed=seed,
        examples=examples,
        output_path=output_path,
        count=count,
        batch_size=batch_size,
        max_batches=max_batches,
        provider_config=provider_config,
        llm_client=llm_client,
    )


def _generate_kind(
    kind: DatasetKind,
    *,
    seed: str | None,
    examples: list[dict[str, Any]] | None,
    output_path: str | Path,
    count: int,
    batch_size: int,
    max_batches: int | None,
    provider_config: OpenRouterConfig | None,
    llm_client: LLMClient | None,
) -> GenerationResult:
    synthesizer = Synthesizer(llm_client=llm_client, provider_config=provider_config)
    return synthesizer.generate(
        kind,
        seed=seed,
        examples=examples,
        generation_config=GenerationConfig(
            count=count,
            batch_size=batch_size,
            max_batches=max_batches,
        ),
        output_config=OutputConfig(path=Path(output_path)),
    )


def _validate_source(
    *,
    seed: str | None,
    examples: list[dict[str, Any]] | None,
) -> None:
    if bool(seed) == bool(examples):
        raise ValueError("Provide exactly one of seed or examples")


def _parse_generated_items(response: str) -> GeneratedItems:
    return GeneratedItems.model_validate(parse_json_object(response))


def _parse_and_validate_rows(
    kind: DatasetKind,
    response: str,
) -> tuple[list[BaseModel], list[tuple[int, str]]]:
    try:
        parsed = _parse_generated_items(response)
    except (ValueError, ValidationError) as error:
        return [], [(-1, str(error))]
    return _validate_rows(kind, parsed.items)


def _validate_rows(
    kind: DatasetKind,
    items: list[Any],
) -> tuple[list[BaseModel], list[tuple[int, str]]]:
    adapter = ROW_ADAPTERS[kind]
    rows: list[BaseModel] = []
    invalid: list[tuple[int, str]] = []
    for index, item in enumerate(items):
        try:
            rows.append(adapter.validate_python(item))
        except ValidationError as error:
            invalid.append((index, str(error)))
    return rows, invalid


def _format_invalid_rows(invalid: list[tuple[int, str]]) -> str:
    return json.dumps(
        [{"index": index, "error": error} for index, error in invalid],
        ensure_ascii=False,
        indent=2,
    )


def _default_max_batches(count: int, batch_size: int) -> int:
    expected_batches = (count + batch_size - 1) // batch_size
    return max(expected_batches + 3, expected_batches * 3)


def _error_stage(invalid: list[tuple[int, str]]) -> str | None:
    if not invalid:
        return None
    if invalid[0][0] == -1:
        return "response_schema"
    return "row_schema"


def _error_summary(invalid: list[tuple[int, str]]) -> str | None:
    if not invalid:
        return None
    summary = invalid[0][1].replace("\n", " ")
    if len(summary) > 300:
        return summary[:297] + "..."
    return summary
