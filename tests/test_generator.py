from __future__ import annotations

import json
from pathlib import Path

from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.generator import Synthesizer


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append(messages)
        return self.responses.pop(0)


def test_generate_sft_writes_jsonl_and_dataset_card(tmp_path: Path) -> None:
    output = tmp_path / "sft.jsonl"
    llm = FakeLLM(
        [
            json.dumps(
                {
                    "items": [
                        {
                            "messages": [
                                {"role": "user", "content": "Name a primary color."},
                                {"role": "assistant", "content": "Red."},
                            ]
                        }
                    ]
                }
            )
        ]
    )
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "sft",
        seed="color questions",
        generation_config=GenerationConfig(count=1),
        output_config=OutputConfig(path=output),
    )

    assert result.rows == 1
    rows = output.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["messages"][1]["content"] == "Red."
    card = (tmp_path / "sft.dataset.md").read_text(encoding="utf-8")
    assert "test/model" in card
    assert "secret" not in card
    assert "<redacted>" in card


def test_invalid_rows_are_repaired(tmp_path: Path) -> None:
    output = tmp_path / "dpo.jsonl"
    invalid = json.dumps({"items": [{"prompt": [{"role": "user", "content": "Q"}]}]})
    repaired = json.dumps(
        {
            "items": [
                {
                    "prompt": [{"role": "user", "content": "Q"}],
                    "chosen": [{"role": "assistant", "content": "Good answer."}],
                    "rejected": [{"role": "assistant", "content": "Bad answer."}],
                }
            ]
        }
    )
    llm = FakeLLM([invalid, repaired])
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "dpo",
        seed="preference data",
        generation_config=GenerationConfig(count=1),
        output_config=OutputConfig(path=output, write_dataset_card=False),
    )

    assert result.rows == 1
    assert result.repair_attempted == 1
    assert result.repair_succeeded == 1
    assert len(llm.calls) == 2


def test_malformed_json_is_repaired(tmp_path: Path) -> None:
    output = tmp_path / "grpo.jsonl"
    repaired = json.dumps(
        {"items": [{"prompt": [{"role": "user", "content": "Solve 3 + 5."}]}]}
    )
    llm = FakeLLM(["not json", repaired])
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "grpo",
        seed="math prompts",
        generation_config=GenerationConfig(count=1),
        output_config=OutputConfig(path=output, write_dataset_card=False),
    )

    assert result.rows == 1
    assert result.repair_attempted == 1
    assert result.repair_succeeded == 1
