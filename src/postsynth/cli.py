from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.generator import Synthesizer
from postsynth.models import (
    MODEL_PRESETS,
    fetch_openrouter_models,
    filter_models,
    find_model,
    resolve_model_selection,
)
from postsynth.schemas import DatasetKind

app = typer.Typer(help="Generate synthetic post-training datasets for TRL.")
generate_app = typer.Typer(help="Generate TRL-native datasets.")
models_app = typer.Typer(help="Inspect OpenRouter models and postsynth presets.")
app.add_typer(generate_app, name="generate")
app.add_typer(models_app, name="models")
console = Console(width=140)


def _generate_command(
    kind: DatasetKind,
    *,
    seed: str | None,
    examples: Path | None,
    count: int,
    out: Path,
    model: str | None,
    model_preset: str | None,
    allow_floating_model: bool,
    refresh_models: bool,
    temperature: float,
    max_tokens: int,
    no_dataset_card: bool,
) -> None:
    if bool(seed) == bool(examples):
        raise typer.BadParameter("Provide exactly one of --seed or --examples.")
    if model and model_preset:
        raise typer.BadParameter("Use either --model or --model-preset, not both.")
    if not model and model_preset and model_preset not in MODEL_PRESETS:
        known = ", ".join(sorted(MODEL_PRESETS))
        raise typer.BadParameter(
            f"Unknown model preset '{model_preset}'. Known presets: {known}"
        )
    try:
        catalog = fetch_openrouter_models(force_refresh=refresh_models)
    except RuntimeError:
        catalog = []
    try:
        model_selection = resolve_model_selection(
            model=model,
            model_preset=model_preset or ("default" if model is None else None),
            catalog=catalog,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error
    if model_selection.floating and not allow_floating_model:
        raise typer.BadParameter(
            "Floating model aliases are less reproducible. Pass --allow-floating-model to use one."
        )

    example_rows = _load_examples(examples) if examples else None
    provider_config = OpenRouterConfig(
        model=model_selection.model,
        requested_model=model_selection.requested,
        model_source=model_selection.source,
        model_metadata=model_selection.metadata(),
    )
    try:
        result = Synthesizer(provider_config=provider_config).generate(
            kind,
            seed=seed,
            examples=example_rows,
            generation_config=GenerationConfig(
                count=count,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            output_config=OutputConfig(path=out, write_dataset_card=not no_dataset_card),
        )
    except RuntimeError as error:
        console.print(f"Error: {error}")
        raise typer.Exit(1) from error
    console.print(
        f"Wrote {result.rows}/{result.requested} {kind.upper()} rows to {result.output_path}"
    )
    if result.dropped_rows:
        console.print(f"Dropped {result.dropped_rows} rows after validation.")


CommonSeed = Annotated[
    str | None,
    typer.Option("--seed", help="Seed instruction describing the dataset to generate."),
]
CommonExamples = Annotated[
    Path | None,
    typer.Option(
        "--examples",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        help="JSONL examples to imitate.",
    ),
]
CommonCount = Annotated[int, typer.Option("--count", min=1, help="Rows to request.")]
CommonOut = Annotated[Path, typer.Option("--out", help="Output JSONL path.")]
CommonModel = Annotated[
    str | None,
    typer.Option("--model", help="Explicit OpenRouter model slug."),
]
CommonModelPreset = Annotated[
    str | None,
    typer.Option("--model-preset", help="postsynth model preset."),
]
CommonAllowFloating = Annotated[
    bool,
    typer.Option(
        "--allow-floating-model",
        help="Allow latest aliases such as ~google/gemini-flash-latest.",
    ),
]
CommonRefreshModels = Annotated[
    bool,
    typer.Option("--refresh-models", help="Refresh the OpenRouter model catalog cache."),
]
CommonTemperature = Annotated[
    float,
    typer.Option("--temperature", min=0.0, max=2.0, help="Sampling temperature."),
]
CommonMaxTokens = Annotated[
    int,
    typer.Option("--max-tokens", min=1, help="Maximum completion tokens."),
]
CommonNoCard = Annotated[
    bool,
    typer.Option("--no-dataset-card", help="Skip writing the sibling dataset card."),
]


@generate_app.command("sft")
def generate_sft(
    seed: CommonSeed = None,
    examples: CommonExamples = None,
    count: CommonCount = 10,
    out: CommonOut = Path("data/generated/sft.jsonl"),
    model: CommonModel = None,
    model_preset: CommonModelPreset = None,
    allow_floating_model: CommonAllowFloating = False,
    refresh_models: CommonRefreshModels = False,
    temperature: CommonTemperature = 0.7,
    max_tokens: CommonMaxTokens = 4096,
    no_dataset_card: CommonNoCard = False,
) -> None:
    _generate_command(
        "sft",
        seed=seed,
        examples=examples,
        count=count,
        out=out,
        model=model,
        model_preset=model_preset,
        allow_floating_model=allow_floating_model,
        refresh_models=refresh_models,
        temperature=temperature,
        max_tokens=max_tokens,
        no_dataset_card=no_dataset_card,
    )


@generate_app.command("dpo")
def generate_dpo(
    seed: CommonSeed = None,
    examples: CommonExamples = None,
    count: CommonCount = 10,
    out: CommonOut = Path("data/generated/dpo.jsonl"),
    model: CommonModel = None,
    model_preset: CommonModelPreset = None,
    allow_floating_model: CommonAllowFloating = False,
    refresh_models: CommonRefreshModels = False,
    temperature: CommonTemperature = 0.7,
    max_tokens: CommonMaxTokens = 4096,
    no_dataset_card: CommonNoCard = False,
) -> None:
    _generate_command(
        "dpo",
        seed=seed,
        examples=examples,
        count=count,
        out=out,
        model=model,
        model_preset=model_preset,
        allow_floating_model=allow_floating_model,
        refresh_models=refresh_models,
        temperature=temperature,
        max_tokens=max_tokens,
        no_dataset_card=no_dataset_card,
    )


@generate_app.command("grpo")
def generate_grpo(
    seed: CommonSeed = None,
    examples: CommonExamples = None,
    count: CommonCount = 10,
    out: CommonOut = Path("data/generated/grpo.jsonl"),
    model: CommonModel = None,
    model_preset: CommonModelPreset = None,
    allow_floating_model: CommonAllowFloating = False,
    refresh_models: CommonRefreshModels = False,
    temperature: CommonTemperature = 0.7,
    max_tokens: CommonMaxTokens = 4096,
    no_dataset_card: CommonNoCard = False,
) -> None:
    _generate_command(
        "grpo",
        seed=seed,
        examples=examples,
        count=count,
        out=out,
        model=model,
        model_preset=model_preset,
        allow_floating_model=allow_floating_model,
        refresh_models=refresh_models,
        temperature=temperature,
        max_tokens=max_tokens,
        no_dataset_card=no_dataset_card,
    )


@generate_app.command("kto")
def generate_kto(
    seed: CommonSeed = None,
    examples: CommonExamples = None,
    count: CommonCount = 10,
    out: CommonOut = Path("data/generated/kto.jsonl"),
    model: CommonModel = None,
    model_preset: CommonModelPreset = None,
    allow_floating_model: CommonAllowFloating = False,
    refresh_models: CommonRefreshModels = False,
    temperature: CommonTemperature = 0.7,
    max_tokens: CommonMaxTokens = 4096,
    no_dataset_card: CommonNoCard = False,
) -> None:
    _generate_command(
        "kto",
        seed=seed,
        examples=examples,
        count=count,
        out=out,
        model=model,
        model_preset=model_preset,
        allow_floating_model=allow_floating_model,
        refresh_models=refresh_models,
        temperature=temperature,
        max_tokens=max_tokens,
        no_dataset_card=no_dataset_card,
    )


def _load_examples(path: Path | None) -> list[dict[str, object]]:
    if path is None:
        return []
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise typer.BadParameter(
                    f"{path}:{line_number} is not valid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise typer.BadParameter(f"{path}:{line_number} must be a JSON object")
            rows.append(row)
    if not rows:
        raise typer.BadParameter(f"{path} did not contain any JSONL rows")
    return rows


@models_app.command("presets")
def model_presets() -> None:
    table = Table(title="postsynth model presets")
    table.add_column("Preset")
    table.add_column("Model", no_wrap=True)
    table.add_column("Floating")
    table.add_column("Description")
    for preset in MODEL_PRESETS.values():
        table.add_row(
            preset.name,
            preset.model,
            "yes" if preset.floating else "no",
            preset.description,
        )
    console.print(table)


@models_app.command("list")
def models_list(
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Filter by OpenRouter provider prefix, e.g. google."),
    ] = None,
    text_only: Annotated[
        bool,
        typer.Option("--text-only", help="Only show models with text output support."),
    ] = False,
    refresh_models: CommonRefreshModels = False,
) -> None:
    catalog = filter_models(
        fetch_openrouter_models(force_refresh=refresh_models),
        provider=provider,
        text_only=text_only,
    )
    table = Table(title="OpenRouter models")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Context", justify="right")
    table.add_column("Prompt $/token")
    table.add_column("Completion $/token")
    for model in catalog:
        pricing = model.pricing or {}
        table.add_row(
            model.id,
            model.name,
            str(model.context_length or ""),
            str(pricing.get("prompt", "")),
            str(pricing.get("completion", "")),
        )
    console.print(table)


@models_app.command("show")
def models_show(
    model: Annotated[str, typer.Argument(help="OpenRouter model slug.")],
    refresh_models: CommonRefreshModels = False,
) -> None:
    catalog = fetch_openrouter_models(force_refresh=refresh_models)
    spec = find_model(catalog, model)
    if spec is None:
        raise typer.BadParameter(f"Model '{model}' was not found in the OpenRouter catalog.")
    console.print_json(data=spec.model_dump())


if __name__ == "__main__":
    app()
