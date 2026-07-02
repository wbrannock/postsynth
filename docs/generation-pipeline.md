# Generation pipeline

How postsynth turns one request into a validated dataset, and what happens
when batches misbehave.

## Batching and the fill loop

A run of `count` rows is split into batches of `batch_size` rows per model
call. Batches are validated row-by-row, and only valid rows count toward
`count`. If a batch under-delivers — malformed JSON, invalid rows, or a failed
request — the fill loop dispatches additional batches until `count` is reached
or the batch budget (`max_batches`, default `max(expected + 3, expected * 3)`)
is exhausted. If the budget runs out first, the run stops short and the result
is marked `incomplete`.

The identical generation prompt is built once per batch size and reused across
batches.

## Concurrency

Up to `concurrency` batches run in flight at once on a bounded thread pool.
Batch sizing accounts for rows already accepted *and* rows expected from
in-flight batches, so a run never requests more than `count` rows total and
the progress bar never overshoots.

A single failed batch request (after retries) does not kill the run: it is
recorded in the diagnostics as `error_stage: "request_failure"`, counted in
`failed_batches`, and the fill loop makes up the rows. The exception is a
failure before *any* batch has succeeded — a bad API key or unknown model
fails fast with a `RuntimeError` instead of burning the whole batch budget.

## Rate limiting

OpenRouter returns HTTP 429 when you exceed your account or model rate limits.
postsynth handles this with a single coordinated policy:

- Rate-limited requests retry up to 5 times with exponential backoff (capped
  at 30s), honouring the server's `Retry-After` header when present.
- The backoff is **shared across all worker threads**: when one request is
  rate-limited, every thread pauses until the cooldown expires, instead of
  each thread independently hammering the API.
- Other transient errors keep a shorter budget: `max_retries + 1` attempts
  (3 by default) with 1–8s exponential backoff.

Under sustained limiting a run can only go as fast as your account allows —
lower `--concurrency` if you see repeated 429 backoffs.

## Token budget

Each request's completion budget defaults to:

```
max_tokens = 400 × batch_rows + 2048
```

The per-row estimate covers typical conversational rows; the headroom absorbs
reasoning tokens, which thinking models spend from the same budget (postsynth
also requests low reasoning effort — see [Models](models.md#reasoning-models)).
Pass `max_tokens` explicitly to override the formula. A budget that is too
small for the batch size is the most common cause of truncated, unparseable
responses.

## Validation, salvage, and repair

Each response passes through up to three recovery stages:

1. **Parse and validate.** The response is parsed as `{"items": [...]}`
   (tolerating markdown code fences) and each row is validated against the
   dataset kind's schema. Invalid rows are counted and dropped; valid rows are
   kept even when siblings fail.
2. **Salvage (truncated responses only).** If the model stopped at the token
   limit (`finish_reason: "length"`), the JSON tail is cut off mid-row.
   postsynth trims the text back to the last complete JSON value, closes the
   open brackets, and validates the recovered rows — no extra API call. LLM
   repair is skipped for truncated responses: the repair prompt would embed
   the oversized output and hit the same limit.
3. **LLM repair (malformed, non-truncated responses).** If parsing or row
   validation fails on a complete response, one repair call (configurable via
   `repair_attempts`) asks the model to fix its own output at temperature 0.

Whatever cannot be recovered is dropped, and the fill loop makes up the
difference.

## Dataset cards and diagnostics

Unless `--no-dataset-card` is passed, every run writes a sibling markdown card
containing the resolved model and generation settings, aggregate validation
stats, and one diagnostic entry per batch: rows requested and accepted,
truncation, repair attempts, and the error stage and summary for anything that
went wrong (`response_schema`, `row_schema`, or `request_failure`). When a run
behaves unexpectedly, the card is the first place to look.
