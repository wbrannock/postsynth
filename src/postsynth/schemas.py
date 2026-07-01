from __future__ import annotations

from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator


DatasetKind: TypeAlias = Literal["sft", "dpo", "grpo", "kto"]


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field(min_length=1)


class SFTRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def requires_assistant_message(self) -> "SFTRow":
        if not any(message.role == "assistant" for message in self.messages):
            raise ValueError("SFT rows must include at least one assistant message")
        return self


class DPORow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: list[Message] = Field(min_length=1)
    chosen: list[Message] = Field(min_length=1)
    rejected: list[Message] = Field(min_length=1)


class GRPORow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: list[Message] = Field(min_length=1)


class KTORow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: list[Message] = Field(min_length=1)
    completion: list[Message] = Field(min_length=1)
    label: bool


class GeneratedItems(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[Any]


ROW_ADAPTERS: dict[DatasetKind, TypeAdapter[Any]] = {
    "sft": TypeAdapter(SFTRow),
    "dpo": TypeAdapter(DPORow),
    "grpo": TypeAdapter(GRPORow),
    "kto": TypeAdapter(KTORow),
}


ROW_DESCRIPTIONS: dict[DatasetKind, str] = {
    "sft": (
        'Each item must be {"messages": [{"role": "user", "content": ...}, '
        '{"role": "assistant", "content": ...}]}.'
    ),
    "dpo": (
        'Each item must be {"prompt": [{"role": "user", "content": ...}], '
        '"chosen": [{"role": "assistant", "content": ...}], '
        '"rejected": [{"role": "assistant", "content": ...}]}.'
    ),
    "grpo": (
        'Each item must be {"prompt": [{"role": "user", "content": ...}]}.'
    ),
    "kto": (
        'Each item must be {"prompt": [{"role": "user", "content": ...}], '
        '"completion": [{"role": "assistant", "content": ...}], "label": true or false}.'
    ),
}

