from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

import postsynth.generator
from postsynth.config import GenerationConfig, OpenRouterConfig, OutputConfig
from postsynth.generator import Synthesizer


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls: list[list[dict[str, str]]] = []
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with self._lock:
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
    assert "model_metadata" in card


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


def test_generation_runs_in_batches_and_reports_progress(tmp_path: Path) -> None:
    output = tmp_path / "sft.jsonl"
    responses = []
    for index in range(3):
        responses.append(
            json.dumps(
                {
                    "items": [
                        {
                            "messages": [
                                {"role": "user", "content": f"Question {index}"},
                                {"role": "assistant", "content": f"Answer {index}"},
                            ]
                        }
                    ]
                }
            )
        )
    llm = FakeLLM(responses)
    progress_updates: list[int] = []
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "sft",
        seed="batched data",
        generation_config=GenerationConfig(count=3, batch_size=1),
        output_config=OutputConfig(path=output, write_dataset_card=False),
        progress_callback=progress_updates.append,
    )

    assert result.rows == 3
    assert len(llm.calls) == 3
    assert progress_updates == [1, 1, 1]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 3


def test_generation_fills_until_requested_valid_rows(tmp_path: Path) -> None:
    output = tmp_path / "sft.jsonl"
    valid_one = json.dumps(
        {
            "items": [
                {
                    "messages": [
                        {"role": "user", "content": "Question 1"},
                        {"role": "assistant", "content": "Answer 1"},
                    ]
                }
            ]
        }
    )
    valid_two = json.dumps(
        {
            "items": [
                {
                    "messages": [
                        {"role": "user", "content": "Question 2"},
                        {"role": "assistant", "content": "Answer 2"},
                    ]
                }
            ]
        }
    )
    llm = FakeLLM(["not json", "still not json", valid_one, valid_two])
    progress_updates: list[int] = []
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "sft",
        seed="fill data",
        generation_config=GenerationConfig(count=2, batch_size=1, max_batches=3),
        output_config=OutputConfig(path=output),
        progress_callback=progress_updates.append,
    )

    assert result.rows == 2
    assert result.incomplete is False
    assert result.batches_attempted == 3
    assert progress_updates == [0, 1, 1]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    card = (tmp_path / "sft.dataset.md").read_text(encoding="utf-8")
    assert '"Batches"' not in card
    assert '"accepted": 0' in card
    assert '"error_stage": "response_schema"' in card


def test_generation_reports_incomplete_when_attempt_budget_exhausted(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sft.jsonl"
    llm = FakeLLM(["not json", "still not json"])
    synthesizer = Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )

    result = synthesizer.generate(
        "sft",
        seed="bad data",
        generation_config=GenerationConfig(count=1, batch_size=1, max_batches=1),
        output_config=OutputConfig(path=output, write_dataset_card=False),
    )

    assert result.rows == 0
    assert result.incomplete is True
    assert result.batches_attempted == 1


def _sft_response(rows: int = 1) -> str:
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


class ConcurrencyProbeLLM:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._in_flight = 0
        self.peak_in_flight = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with self._lock:
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        time.sleep(0.05)
        with self._lock:
            self._in_flight -= 1
        return _sft_response()


class FlakyLLM:
    """Succeeds, then raises once, then succeeds for all later calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._call_count = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        with self._lock:
            self._call_count += 1
            if self._call_count == 2:
                raise RuntimeError("transient request failure")
        return _sft_response()


class AlwaysFailingLLM:
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        raise RuntimeError("no auth")


def _synthesizer(llm: object) -> Synthesizer:
    return Synthesizer(
        llm_client=llm,
        provider_config=OpenRouterConfig(api_key="secret", model="test/model"),
    )


def test_batches_run_concurrently(tmp_path: Path) -> None:
    llm = ConcurrencyProbeLLM()
    result = _synthesizer(llm).generate(
        "sft",
        seed="concurrent data",
        generation_config=GenerationConfig(count=4, batch_size=1, concurrency=4),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert result.rows == 4
    assert llm.peak_in_flight >= 2


def test_concurrent_fill_does_not_overshoot_count(tmp_path: Path) -> None:
    # Every batch requests 2 rows but only delivers 1, forcing top-up dispatches.
    llm = FakeLLM([_sft_response(1) for _ in range(8)])
    progress_updates: list[int] = []
    result = _synthesizer(llm).generate(
        "sft",
        seed="under-delivering data",
        generation_config=GenerationConfig(
            count=4, batch_size=2, concurrency=2, repair_attempts=0
        ),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
        progress_callback=progress_updates.append,
    )

    assert result.rows == 4
    assert result.incomplete is False
    assert sum(progress_updates) == 4
    assert [d["batch"] for d in result.batch_diagnostics] == list(
        range(1, result.batches_attempted + 1)
    )


def test_sequential_fill_behaviour_is_preserved_with_explicit_concurrency(
    tmp_path: Path,
) -> None:
    output = tmp_path / "sft.jsonl"
    llm = FakeLLM(["not json", "still not json", _sft_response(), _sft_response()])
    progress_updates: list[int] = []
    result = _synthesizer(llm).generate(
        "sft",
        seed="fill data",
        generation_config=GenerationConfig(
            count=2, batch_size=1, max_batches=3, concurrency=1
        ),
        output_config=OutputConfig(path=output, write_dataset_card=False),
        progress_callback=progress_updates.append,
    )

    assert result.rows == 2
    assert result.batches_attempted == 3
    assert progress_updates == [0, 1, 1]


def test_all_batches_failing_raises(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="all generation batches failed"):
        _synthesizer(AlwaysFailingLLM()).generate(
            "sft",
            seed="doomed data",
            generation_config=GenerationConfig(count=4, batch_size=1, concurrency=2),
            output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
        )


def test_single_failed_batch_is_tolerated(tmp_path: Path) -> None:
    result = _synthesizer(FlakyLLM()).generate(
        "sft",
        seed="flaky data",
        generation_config=GenerationConfig(count=2, batch_size=1, max_batches=4),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert result.rows == 2
    assert result.incomplete is False
    assert result.failed_batches == 1
    failures = [d for d in result.batch_diagnostics if d["error_stage"] == "request_failure"]
    assert len(failures) == 1
    assert "transient request failure" in failures[0]["error_summary"]


def test_generation_messages_are_built_once_per_batch_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_calls: list[int] = []
    original = postsynth.generator.build_generation_messages

    def counting_build(kind, *, count, seed, examples):
        build_calls.append(count)
        return original(kind, count=count, seed=seed, examples=examples)

    monkeypatch.setattr(postsynth.generator, "build_generation_messages", counting_build)
    llm = FakeLLM([_sft_response() for _ in range(3)])
    result = _synthesizer(llm).generate(
        "sft",
        seed="memoized data",
        generation_config=GenerationConfig(count=3, batch_size=1),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert result.rows == 3
    assert build_calls == [1]
    assert len(llm.calls) == 3
    assert len({calls[1]["content"] for calls in llm.calls}) == 1


def test_generation_config_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        GenerationConfig(count=1, concurrency=0)


class TruncatingLLM:
    """First call returns a truncated payload; later calls return valid rows."""

    def __init__(self, truncated_payload: str) -> None:
        self._lock = threading.Lock()
        self._calls = 0
        self._truncated_payload = truncated_payload
        self.call_count = 0

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ):
        from postsynth.llm import LLMResponse

        with self._lock:
            self._calls += 1
            self.call_count = self._calls
            first = self._calls == 1
        if first:
            return LLMResponse(content=self._truncated_payload, truncated=True)
        return LLMResponse(content=_sft_response())


def test_truncated_batch_is_salvaged_without_repair_call(tmp_path: Path) -> None:
    full = _sft_response(3)
    truncated = full[: full.rfind('{"messages"')]  # third row cut off
    llm = TruncatingLLM(truncated)

    result = _synthesizer(llm).generate(
        "sft",
        seed="truncated data",
        generation_config=GenerationConfig(count=3, batch_size=3, concurrency=1),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert result.rows == 3
    # One truncated generation call (salvaged, no repair) + one fill call.
    assert llm.call_count == 2
    assert result.repair_attempted == 0
    assert result.batch_diagnostics[0]["truncated"] is True
    assert result.batch_diagnostics[0]["accepted"] == 2


class MaxTokensRecordingLLM:
    def __init__(self) -> None:
        self.max_tokens_seen: list[int] = []

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.max_tokens_seen.append(max_tokens)
        return _sft_response(5)


def test_max_tokens_scales_with_batch_size_by_default(tmp_path: Path) -> None:
    llm = MaxTokensRecordingLLM()
    _synthesizer(llm).generate(
        "sft",
        seed="auto budget",
        generation_config=GenerationConfig(count=5, batch_size=5),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert llm.max_tokens_seen == [400 * 5 + 2048]


def test_explicit_max_tokens_is_respected(tmp_path: Path) -> None:
    llm = MaxTokensRecordingLLM()
    _synthesizer(llm).generate(
        "sft",
        seed="explicit budget",
        generation_config=GenerationConfig(count=5, batch_size=5, max_tokens=1234),
        output_config=OutputConfig(path=tmp_path / "sft.jsonl", write_dataset_card=False),
    )

    assert llm.max_tokens_seen == [1234]
