# postsynth

Generate synthetic post-training datasets in TRL-native formats.

postsynth v1 focuses on conversational JSONL rows for common TRL workflows:

| Kind | Row fields |
|------|-----------|
| SFT  | `messages` |
| DPO  | `prompt`, `chosen`, `rejected` |
| GRPO | `prompt` |
| KTO  | `prompt`, `completion`, `label` |

Rows follow the [conversational formats documented by TRL](https://huggingface.co/docs/trl/en/dataset_formats) and load directly with `datasets`:

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="data/generated/sft.jsonl")
```

## Install

```bash
pip install postsynth
```

Or for development:

```bash
uv sync --dev
```

## Configure OpenRouter

postsynth generates data through [OpenRouter](https://openrouter.ai). Set your key in the environment:

```bash
export OPENROUTER_API_KEY="..."
```

For local development you can also put it in a `.env` file (ignored by git).

## Quickstart

```bash
postsynth generate sft \
  --seed "Customer-support chats about subscription billing." \
  --count 100 \
  --out data/generated/sft.jsonl
```

This writes `sft.jsonl` plus a sibling dataset card (`sft.dataset.md`) recording
the model, generation settings, and per-batch diagnostics.

Next steps:

- [CLI](cli.md) — all commands and options
- [Python API](python-api.md) — `generate_sft` and friends
- [Models](models.md) — presets, pinning, and the OpenRouter catalog
- [Generation pipeline](generation-pipeline.md) — how batching, concurrency, validation, and repair work under the hood
