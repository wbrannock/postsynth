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
  --out data/generated/sft.jsonl
```

Generate from examples:

```bash
postsynth generate dpo \
  --examples examples/dpo.jsonl \
  --count 100 \
  --out data/generated/dpo.jsonl
```

The CLI writes a JSONL dataset and a sibling dataset card, for example
`sft.dataset.md`, with non-secret generation metadata.

## Python API

```python
from postsynth import generate_sft

result = generate_sft(
    seed="Math tutoring conversations for middle-school students.",
    count=25,
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
