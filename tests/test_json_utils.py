from __future__ import annotations

import json

import pytest

from postsynth.json_utils import parse_json_object, salvage_json_object


def _items_json(rows: int) -> str:
    return json.dumps(
        {
            "items": [
                {
                    "messages": [
                        {"role": "user", "content": f"Question {index}"},
                        {"role": "assistant", "content": f"Answer {index}"},
                    ]
                }
                for index in range(rows)
            ]
        }
    )


def test_parse_json_object_handles_code_fence() -> None:
    parsed = parse_json_object("```json\n" + _items_json(1) + "\n```")
    assert len(parsed["items"]) == 1


def test_salvage_recovers_complete_rows_from_truncated_output() -> None:
    full = _items_json(3)
    truncated = full[: full.rfind('{"messages"')]  # cut mid third row

    parsed = salvage_json_object(truncated)
    assert len(parsed["items"]) == 2
    assert parsed["items"][1]["messages"][1]["content"] == "Answer 1"


def test_salvage_handles_cut_inside_string() -> None:
    full = _items_json(2)
    truncated = full[: full.rfind("Answer 1") + 3]  # cut inside a string value

    # Salvage trims to the last complete JSON value: row 0 survives intact and
    # row 1 loses its cut-off assistant message (row validation drops it later).
    parsed = salvage_json_object(truncated)
    assert parsed["items"][0]["messages"][1]["content"] == "Answer 0"
    assert all(m["content"] for item in parsed["items"] for m in item["messages"])


def test_salvage_raises_when_nothing_recoverable() -> None:
    with pytest.raises(ValueError):
        salvage_json_object('{"items": [{"mess')

    with pytest.raises(ValueError):
        salvage_json_object("no json here")
