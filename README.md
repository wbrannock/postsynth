# postsynth

Generate synthetic post-training datasets in TRL-native formats.

Full documentation lives in [`docs/`](docs/index.md) — build it locally with
`uv run mkdocs serve`.

`postsynth` v1 focuses on conversational JSONL rows for common TRL workflows:

- SFT: `messages`
- DPO: `prompt`, `chosen`, `rejected`
- GRPO: `prompt`
- KTO: `prompt`, `completion`, `label`

## Install

```bash
uv sync --dev
```

## Configure OpenRouter

Set your key in the environment:

```bash
export OPENROUTER_API_KEY="..."
```

For local development, you can also create a `.env` file:

```bash
OPENROUTER_API_KEY=...
```

`.env` files are ignored by git.

## CLI

Generate from a seed instruction:

```bash
postsynth generate sft \
  --seed "Customer-support chats about subscription billing." \
  --count 100 \
  --batch-size 25 \
  --max-batches 20 \
  --concurrency 4 \
  --out data/generated/sft.jsonl
```

Generate from examples:

```bash
postsynth generate dpo \
  --examples examples/dpo.jsonl \
  --count 100 \
  --out data/generated/dpo.jsonl
```

By default, postsynth uses the pinned OpenRouter model
`google/gemini-3.5-flash-20260519` for reproducibility. Use a preset:

```bash
postsynth generate sft \
  --seed "Math tutoring conversations." \
  --model-preset cheap \
  --count 100 \
  --out data/generated/sft.jsonl
```

Or pass an explicit OpenRouter model slug:

```bash
postsynth generate sft \
  --seed "Math tutoring conversations." \
  --model google/gemini-3.5-flash \
  --count 100 \
  --out data/generated/sft.jsonl
```

Floating OpenRouter aliases such as `~google/gemini-flash-latest` are blocked by
default because they are less reproducible. To use one intentionally:

```bash
postsynth generate sft \
  --seed "Math tutoring conversations." \
  --model-preset latest-gemini-flash \
  --allow-floating-model \
  --count 100 \
  --out data/generated/sft.jsonl
```

Inspect available presets and OpenRouter models:

```bash
postsynth models presets
postsynth models list --provider google --text-only
postsynth models show google/gemini-3.5-flash
```

The CLI writes a JSONL dataset and a sibling dataset card, for example
`sft.dataset.md`, with non-secret generation metadata including requested model,
resolved model, model source, floating-model status, and catalog metadata when
available.

Large generations run in batches, with up to `--concurrency` batches in flight
at once (CLI default 4; the Python API defaults to `concurrency=1`). OpenRouter
rate limits are handled automatically — rate-limited requests back off together
and honour `Retry-After` — but under sustained limiting a run can only go as
fast as your account allows, so lower `--concurrency` if you see repeated 429s.

The tqdm progress bar advances when valid rows are accepted, not merely when a
request finishes. If a batch returns malformed JSON, too few valid rows, or a
failed request, postsynth keeps requesting fill batches until it reaches
`--count` or exhausts the attempt budget. The completion token budget scales
with `--batch-size` automatically; pass `--max-tokens` to override it.

Tune request size with `--batch-size`, cap total model calls with
`--max-batches`, or disable progress output in scripts with `--no-progress`.
Dataset cards include aggregate validation stats and per-batch diagnostics so
you can see which batches were accepted, repaired, truncated, dropped, or
failed.

## Python API

```python
from postsynth import generate_sft

result = generate_sft(
    seed="Math tutoring conversations for middle-school students.",
    count=25,
    batch_size=5,
    concurrency=4,
    output_path="data/generated/math_sft.jsonl",
)

print(result.rows)
```

## TRL Loading

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/generated/sft.jsonl")
```

The generated rows follow the conversational formats documented by TRL:
https://huggingface.co/docs/trl/en/dataset_formats
