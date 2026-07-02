from __future__ import annotations

import json
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(stripped[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed


def salvage_json_object(text: str) -> dict[str, Any]:
    """Recover the largest parseable object prefix from truncated JSON output.

    Truncated generations usually cut mid-row inside the "items" array; the
    complete rows before the cut are still recoverable by trimming to the last
    fully closed value and closing the remaining open containers.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _strip_code_fence(stripped)
    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found to salvage")
    stripped = stripped[start:]

    candidates: list[tuple[int, str]] = []
    stack: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                break
            stack.pop()
            candidates.append((index, "".join(reversed(stack))))

    for index, closers in reversed(candidates[-100:]):
        try:
            parsed = json.loads(stripped[: index + 1] + closers)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not salvage a JSON object from truncated output")


def _strip_code_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

