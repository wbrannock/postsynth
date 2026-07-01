from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.generator import Synthesizer
from postsynth.schemas import DatasetKind

app = typer.Typer(help="Generate synthetic post-training datasets for TRL.")
generate_app = typer.Typer(help="Generate TRL-native datasets.")
app.add_typer(generate_app, name="generate")
console = Console()


def _generate_command(
    kind: DatasetKind,
    *,
    seed: str | None,
    examples: Path | None,
    count: int,
    out: Path,
    model: str,
    temperature: float,
    max_tokens: int,
    no_dataset_card: bool,
) -> None:
    if bool(seed) == bool(examples):
        raise typer.BadParameter("Provide exactly one of --seed or --examples.")

    example_rows = _load_examples(examples) if examples else None
    provider_config = OpenRouterConfig(model=model)
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
    str,
    typer.Option("--model", help="OpenRouter model slug."),
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
    model: CommonModel = "openai/gpt-4o-mini",
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
    model: CommonModel = "openai/gpt-4o-mini",
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
    model: CommonModel = "openai/gpt-4o-mini",
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
    model: CommonModel = "openai/gpt-4o-mini",
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


if __name__ == "__main__":
    app()
