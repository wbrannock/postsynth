from __future__ import annotations

import json
from typing import Any

from postsynth.schemas import DatasetKind, ROW_DESCRIPTIONS


SYSTEM_PROMPT = (
    "You generate high-quality synthetic post-training dataset rows for TRL. "
    "Return only valid JSON. Do not include markdown, comments, or prose."
)


def build_generation_messages(
    kind: DatasetKind,
    *,
    count: int,
    seed: str | None,
    examples: list[dict[str, Any]] | None,
) -> list[dict[str, str]]:
    source = (
        f"Seed instruction:\n{seed}"
        if seed is not None
        else "Seed examples:\n" + json.dumps(examples or [], ensure_ascii=False, indent=2)
    )
    user = f"""
Create exactly {count} diverse synthetic rows for the "{kind}" dataset type.

Return a JSON object with this shape:
{{"items": [ ... ]}}

Required row format:
{ROW_DESCRIPTIONS[kind]}

Quality requirements:
- Return minified JSON on a single line with no indentation or extra whitespace.
- Use conversational TRL message arrays.
- Keep prompts realistic and useful for post-training.
- Avoid duplicate rows.
- Do not include private, copyrighted, or credential-like data.
- For rejected or negative examples, make them plausibly wrong but not toxic.

{source}
""".strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def build_repair_messages(
    kind: DatasetKind,
    *,
    invalid_output: str,
    validation_error: str,
    count: int,
) -> list[dict[str, str]]:
    user = f"""
Repair the JSON for a "{kind}" TRL dataset generation response.

Return only a JSON object with exactly this shape:
{{"items": [ ... ]}}

Return minified JSON on a single line with no indentation or extra whitespace.
It must contain exactly {count} valid items.
Required row format:
{ROW_DESCRIPTIONS[kind]}

Validation error:
{validation_error}

Invalid output:
{invalid_output}
""".strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

