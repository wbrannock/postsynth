# CLI

## Generate commands

One command per dataset kind, all with the same options:

```bash
postsynth generate sft   --seed "..." --count 100 --out data/generated/sft.jsonl
postsynth generate dpo   --examples examples/dpo.jsonl --count 100 --out data/generated/dpo.jsonl
postsynth generate grpo  --seed "..." --count 100 --out data/generated/grpo.jsonl
postsynth generate kto   --seed "..." --count 100 --out data/generated/kto.jsonl
```

Provide exactly one source:

- `--seed` — a natural-language instruction describing the dataset to generate
- `--examples` — a JSONL file of rows to imitate

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--count` | 10 | Valid rows to produce |
| `--out` | `data/generated/<kind>.jsonl` | Output JSONL path |
| `--batch-size` | 25 | Rows requested per model call |
| `--max-batches` | auto | Cap on total model-call batches before stopping short |
| `--concurrency` | 4 | Batches in flight at once |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | auto | Completion tokens per request; defaults to scaling with `--batch-size` |
| `--model` | preset `default` | Explicit OpenRouter model slug |
| `--model-preset` | `default` | Named preset (see [Models](models.md)) |
| `--allow-floating-model` | off | Permit floating aliases like `~google/gemini-flash-latest` |
| `--refresh-models` | off | Refresh the cached OpenRouter model catalog |
| `--no-progress` | off | Disable the tqdm progress bar |
| `--no-dataset-card` | off | Skip writing the sibling dataset card |

## Progress and results

The progress bar advances when valid rows are **accepted**, not merely when a
request finishes. If a batch under-delivers, postsynth keeps requesting fill
batches until it reaches `--count` or exhausts the batch budget — see the
[generation pipeline](generation-pipeline.md) for details.

After a run the CLI reports rows written, rows dropped by validation, and any
failed batch requests. Full per-batch diagnostics land in the dataset card.

## Model inspection

```bash
postsynth models presets
postsynth models list --provider google --text-only
postsynth models show google/gemini-3.5-flash
```
