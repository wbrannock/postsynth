# Python API

## Generate functions

One function per dataset kind: `generate_sft`, `generate_dpo`, `generate_grpo`,
`generate_kto`. All share the same signature:

```python
from postsynth import generate_sft

result = generate_sft(
    seed="Math tutoring conversations for middle-school students.",
    count=25,
    batch_size=5,
    concurrency=4,          # library default is 1
    output_path="data/generated/math_sft.jsonl",
)

print(result.rows)
```

Pass `examples=[{...}, ...]` instead of `seed` to imitate existing rows.

!!! note "Concurrency default"
    The CLI defaults to 4 concurrent batches; the Python API defaults to
    `concurrency=1` so library behaviour is sequential and deterministic
    unless you opt in.

## Synthesizer

For full control, construct a `Synthesizer` with explicit configs:

```python
from pathlib import Path
from postsynth import GenerationConfig, OpenRouterConfig, OutputConfig, Synthesizer

result = Synthesizer(
    provider_config=OpenRouterConfig(model="google/gemini-2.5-pro"),
).generate(
    "sft",
    seed="Realistic customer-support escalations.",
    generation_config=GenerationConfig(count=200, batch_size=25, concurrency=4),
    output_config=OutputConfig(path=Path("data/generated/sft.jsonl")),
    progress_callback=lambda accepted: print(f"+{accepted} rows"),
)
```

`Synthesizer` also accepts a custom `llm_client` implementing the `LLMClient`
protocol (`complete(messages, *, model, temperature, max_tokens)`), which is
how the test suite substitutes fakes.

## Configuration

### `GenerationConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `count` | required | Valid rows to produce |
| `batch_size` | 25 | Rows requested per model call |
| `max_batches` | auto | Batch budget; defaults to `max(expected + 3, expected * 3)` |
| `concurrency` | 1 | Batches in flight at once |
| `temperature` | 0.7 | Sampling temperature |
| `max_tokens` | `None` (auto) | Completion token budget per request; `None` scales with batch size |
| `repair_attempts` | 1 | LLM repair calls per invalid batch |

### `OpenRouterConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `model` | pinned Gemini Flash | OpenRouter model slug |
| `api_key` | env `OPENROUTER_API_KEY` | API key |
| `timeout_seconds` | 60 | Per-request timeout |
| `max_retries` | 2 | Retries for non-rate-limit errors |
| `reasoning_effort` | `"low"` | Reasoning effort requested from thinking models; `None` to omit |

## `GenerationResult`

Returned by every generate call:

| Field | Description |
|-------|-------------|
| `rows` / `requested` | Rows written vs requested |
| `output_path` | Where the JSONL landed |
| `invalid_rows` | Rows that failed schema validation |
| `repair_attempted` / `repair_succeeded` | LLM repair call outcomes |
| `dropped_rows` | Rows lost after validation and repair |
| `batches_attempted` | Model-call batches used |
| `failed_batches` | Batches whose request errored after retries |
| `incomplete` | True if the run stopped short of `count` |
| `batch_diagnostics` | Per-batch detail (also written to the dataset card) |
