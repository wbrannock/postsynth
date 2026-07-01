from __future__ import annotations

import pytest
from pydantic import ValidationError

from postsynth.schemas import DPORow, GRPORow, KTORow, SFTRow


def test_sft_row_requires_assistant_message() -> None:
    with pytest.raises(ValidationError):
        SFTRow.model_validate(
            {"messages": [{"role": "user", "content": "Write a haiku."}]}
        )


def test_trl_conversational_rows_validate() -> None:
    SFTRow.model_validate(
        {
            "messages": [
                {"role": "user", "content": "Write a haiku."},
                {"role": "assistant", "content": "Clear winter moonlight..."},
            ]
        }
    )
    DPORow.model_validate(
        {
            "prompt": [{"role": "user", "content": "Explain gravity."}],
            "chosen": [{"role": "assistant", "content": "Gravity attracts mass."}],
            "rejected": [{"role": "assistant", "content": "Gravity is a color."}],
        }
    )
    GRPORow.model_validate(
        {"prompt": [{"role": "user", "content": "Solve 2 + 2."}]}
    )
    KTORow.model_validate(
        {
            "prompt": [{"role": "user", "content": "Solve 2 + 2."}],
            "completion": [{"role": "assistant", "content": "4"}],
            "label": True,
        }
    )

