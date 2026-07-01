# postsynth

Generate synthetic post-training datasets in TRL-native formats.

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

Large generations run in batches and show a tqdm progress bar by default. The
progress bar advances when valid rows are accepted, not merely when a request
finishes. If a batch returns malformed JSON or too few valid rows, postsynth keeps
requesting fill batches until it reaches `--count` or exhausts the attempt
budget.

Tune request size with `--batch-size`, cap total model calls with
`--max-batches`, or disable progress output in scripts with `--no-progress`.
Dataset cards include aggregate validation stats and per-batch diagnostics so
you can see which batches were accepted, repaired, dropped, or left incomplete.

## Python API

```python
from postsynth import generate_sft

result = generate_sft(
    seed="Math tutoring conversations for middle-school students.",
    count=25,
    batch_size=5,
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
