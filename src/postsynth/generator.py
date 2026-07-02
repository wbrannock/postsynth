from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ValidationError

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.json_utils import parse_json_object, salvage_json_object
from postsynth.llm import LLMClient, LLMResponse, OpenRouterClient
from postsynth.models import resolve_model_selection
from postsynth.prompts import build_generation_messages, build_repair_messages
from postsynth.schemas import DatasetKind, GeneratedItems, ROW_ADAPTERS
from postsynth.writers import write_dataset_card, write_jsonl

ROW_TOKEN_BUDGET = 400
COMPLETION_TOKEN_HEADROOM = 2048


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
    failed_batches: int
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
            "failed_batches": 0,
            "incomplete": 0,
        }
        batch_diagnostics: list[dict[str, Any]] = []
        max_batches = generation_config.max_batches or _default_max_batches(
            generation_config.count,
            generation_config.batch_size,
        )

        messages_cache: dict[int, list[dict[str, str]]] = {}

        def _messages_for(batch_count: int) -> list[dict[str, str]]:
            if batch_count not in messages_cache:
                messages_cache[batch_count] = build_generation_messages(
                    kind,
                    count=batch_count,
                    seed=seed,
                    examples=examples,
                )
            return messages_cache[batch_count]

        # All shared state below is mutated only by this (collector) thread;
        # workers run _run_batch, which touches nothing shared.
        in_flight: dict[Future, tuple[int, int]] = {}
        batches_dispatched = 0
        last_exception: BaseException | None = None
        any_batch_succeeded = False

        def _can_dispatch() -> bool:
            expected = len(rows) + sum(count for _, count in in_flight.values())
            return (
                expected < generation_config.count
                and batches_dispatched < max_batches
                and len(in_flight) < generation_config.concurrency
                and not (last_exception is not None and not any_batch_succeeded)
            )

        with ThreadPoolExecutor(max_workers=generation_config.concurrency) as executor:

            def _dispatch_one() -> None:
                nonlocal batches_dispatched
                expected = len(rows) + sum(count for _, count in in_flight.values())
                batch_count = min(
                    generation_config.count - expected,
                    generation_config.batch_size,
                )
                batches_dispatched += 1
                batch_config = replace(generation_config, count=batch_count)
                future = executor.submit(
                    self._run_batch,
                    kind,
                    messages=_messages_for(batch_count),
                    batch_config=batch_config,
                )
                in_flight[future] = (batches_dispatched, batch_count)

            try:
                while _can_dispatch():
                    _dispatch_one()
                while in_flight:
                    done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in done:
                        batch_index, batch_count = in_flight.pop(future)
                        error = future.exception()
                        stats["batches_attempted"] += 1
                        if error is not None:
                            last_exception = error
                            stats["failed_batches"] += 1
                            stats["dropped_rows"] += batch_count
                            batch_diagnostics.append(
                                {
                                    "batch": batch_index,
                                    "requested": batch_count,
                                    "accepted": 0,
                                    "dropped": batch_count,
                                    "invalid_rows": 0,
                                    "repair_attempted": 0,
                                    "repair_succeeded": 0,
                                    "truncated": False,
                                    "error_stage": "request_failure",
                                    "error_summary": _truncate_summary(str(error)),
                                }
                            )
                            if progress_callback:
                                progress_callback(0)
                            continue
                        any_batch_succeeded = True
                        batch_rows, batch_stats = future.result()
                        rows.extend(batch_rows)
                        for key in (
                            "invalid_rows",
                            "repair_attempted",
                            "repair_succeeded",
                            "dropped_rows",
                        ):
                            stats[key] += batch_stats[key]
                        batch_diagnostics.append(
                            {
                                "batch": batch_index,
                                "requested": batch_count,
                                "accepted": len(batch_rows),
                                "dropped": batch_stats["dropped_rows"],
                                "invalid_rows": batch_stats["invalid_rows"],
                                "repair_attempted": batch_stats["repair_attempted"],
                                "repair_succeeded": batch_stats["repair_succeeded"],
                                "truncated": batch_stats["truncated"],
                                "error_stage": batch_stats["error_stage"],
                                "error_summary": batch_stats["error_summary"],
                            }
                        )
                        if progress_callback:
                            progress_callback(len(batch_rows))
                    while _can_dispatch():
                        _dispatch_one()
            except BaseException:
                executor.shutdown(wait=False, cancel_futures=True)
                raise

        if last_exception is not None and not any_batch_succeeded:
            raise RuntimeError("all generation batches failed") from last_exception

        batch_diagnostics.sort(key=lambda diagnostic: diagnostic["batch"])
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
            failed_batches=stats["failed_batches"],
            incomplete=bool(stats["incomplete"]),
            batch_diagnostics=batch_diagnostics,
        )

    def _run_batch(
        self,
        kind: DatasetKind,
        *,
        messages: list[dict[str, str]],
        batch_config: GenerationConfig,
    ) -> tuple[list[BaseModel], dict[str, Any]]:
        response, truncated = _response_parts(
            self.llm_client.complete(
                messages,
                model=self.provider_config.model,
                temperature=batch_config.temperature,
                max_tokens=_resolve_max_tokens(batch_config),
            )
        )
        return self._parse_validate_repair(
            kind,
            response=response,
            truncated=truncated,
            generation_config=batch_config,
        )

    def _parse_validate_repair(
        self,
        kind: DatasetKind,
        *,
        response: str,
        generation_config: GenerationConfig,
        truncated: bool = False,
    ) -> tuple[list[BaseModel], dict[str, Any]]:
        rows, invalid = _parse_and_validate_rows(kind, response, allow_salvage=truncated)
        repair_attempted = 0
        repair_succeeded = 0

        # Repairing a truncated response is doomed: the repair prompt embeds the
        # cut-off output and the full corrected JSON cannot fit either.
        if invalid and not truncated and generation_config.repair_attempts > 0:
            repair_attempted = 1
            repair_messages = build_repair_messages(
                kind,
                invalid_output=response,
                validation_error=_format_invalid_rows(invalid),
                count=generation_config.count,
            )
            repaired, repair_truncated = _response_parts(
                self.llm_client.complete(
                    repair_messages,
                    model=self.provider_config.model,
                    temperature=0.0,
                    max_tokens=_resolve_max_tokens(generation_config),
                )
            )
            rows, invalid = _parse_and_validate_rows(
                kind, repaired, allow_salvage=repair_truncated
            )
            repair_succeeded = 1 if not invalid else 0

        rows = rows[: generation_config.count]
        dropped_rows = max(0, generation_config.count - len(rows))
        stats = {
            "invalid_rows": len(invalid),
            "repair_attempted": repair_attempted,
            "repair_succeeded": repair_succeeded,
            "dropped_rows": dropped_rows,
            "truncated": truncated,
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
    concurrency: int = 1,
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
        concurrency=concurrency,
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
    concurrency: int = 1,
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
        concurrency=concurrency,
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
    concurrency: int = 1,
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
        concurrency=concurrency,
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
    concurrency: int = 1,
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
        concurrency=concurrency,
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
    concurrency: int,
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
            concurrency=concurrency,
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
    *,
    allow_salvage: bool = False,
) -> tuple[list[BaseModel], list[tuple[int, str]]]:
    try:
        parsed = _parse_generated_items(response)
    except (ValueError, ValidationError) as error:
        if not allow_salvage:
            return [], [(-1, str(error))]
        try:
            parsed = GeneratedItems.model_validate(salvage_json_object(response))
        except (ValueError, ValidationError):
            return [], [(-1, str(error))]
    return _validate_rows(kind, parsed.items)


def _response_parts(response: str | LLMResponse) -> tuple[str, bool]:
    if isinstance(response, LLMResponse):
        return response.content, response.truncated
    return response, False


def _resolve_max_tokens(config: GenerationConfig) -> int:
    if config.max_tokens is not None:
        return config.max_tokens
    return ROW_TOKEN_BUDGET * config.count + COMPLETION_TOKEN_HEADROOM


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
    return _truncate_summary(invalid[0][1])


def _truncate_summary(text: str) -> str:
    summary = text.replace("\n", " ")
    if len(summary) > 300:
        return summary[:297] + "..."
    return summary
