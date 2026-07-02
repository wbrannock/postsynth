# Models

## Presets

postsynth pins model versions by default so datasets are reproducible.

| Preset | Model | Notes |
|--------|-------|-------|
| `default` | `google/gemini-3.5-flash-20260519` | Pinned Gemini Flash |
| `cheap` | `google/gemini-3.1-flash-lite-20260507` | Pinned low-cost Flash Lite |
| `quality` | `google/gemini-2.5-pro` | Pinned Gemini Pro for higher quality |
| `latest-gemini-flash` | `~google/gemini-flash-latest` | Floating alias — see below |

```bash
postsynth generate sft --seed "..." --model-preset quality --count 100 --out out.jsonl
```

Any OpenRouter slug works via `--model`:

```bash
postsynth generate sft --seed "..." --model anthropic/claude-sonnet-4.6 --count 100 --out out.jsonl
```

## Floating models

Aliases prefixed with `~` (like `~google/gemini-flash-latest`) track whatever
the provider currently serves, so two runs may use different underlying models.
They are blocked by default; pass `--allow-floating-model` to use one
intentionally. The dataset card records both the requested alias and the
resolved model.

## The catalog cache

`postsynth models list` and model resolution consult the OpenRouter catalog,
cached on disk for one hour. `--refresh-models` forces a refetch. If the
catalog is unreachable, generation proceeds without catalog metadata rather
than failing.

## Reasoning models

Thinking models (including the default Gemini Flash) spend part of their
completion budget on reasoning tokens. postsynth requests **low** reasoning
effort by default — synthetic row generation is formulaic, and reclaimed
reasoning tokens go to actual rows instead. Override via
`OpenRouterConfig(reasoning_effort="high")` or disable the request field
entirely with `reasoning_effort=None`.
