from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from openai import RateLimitError

from postsynth.config import OpenRouterConfig
from postsynth.llm import OpenRouterClient, _Cooldown


def _rate_limit_error(retry_after: str | None = None) -> RateLimitError:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    response = httpx.Response(
        429,
        headers=headers,
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    return RateLimitError("rate limited", response=response, body=None)


def _completion(content: str, finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]
    )


class StubCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.kwargs: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> SimpleNamespace:
        self.calls += 1
        self.kwargs.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _client_with_stub(outcomes: list[object]) -> tuple[OpenRouterClient, StubCompletions]:
    client = OpenRouterClient(OpenRouterConfig(api_key="secret", model="test/model"))
    completions = StubCompletions(outcomes)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def _complete(client: OpenRouterClient) -> str:
    return client.complete(
        [{"role": "user", "content": "hi"}],
        model="test/model",
        temperature=0.0,
        max_tokens=16,
    )


def test_rate_limit_errors_are_retried_beyond_default_attempts() -> None:
    client, completions = _client_with_stub(
        [
            _rate_limit_error("0"),
            _rate_limit_error("0"),
            _rate_limit_error("0"),
            _completion("ok"),
        ]
    )

    assert _complete(client).content == "ok"
    assert completions.calls == 4


def test_rate_limit_honours_retry_after_via_shared_cooldown() -> None:
    client, _ = _client_with_stub([_rate_limit_error("0.2"), _completion("ok")])

    start = time.monotonic()
    assert _complete(client).content == "ok"
    assert time.monotonic() - start >= 0.2


def test_non_rate_limit_errors_keep_default_attempt_budget() -> None:
    client, completions = _client_with_stub([RuntimeError("boom")] * 5)

    with pytest.raises(RuntimeError, match="boom"):
        _complete(client)
    # max_retries=2 -> 3 attempts, no rate-limit extension.
    assert completions.calls == 3


def test_truncated_completion_is_flagged() -> None:
    client, _ = _client_with_stub([_completion("cut off", finish_reason="length")])

    response = _complete(client)
    assert response.content == "cut off"
    assert response.truncated is True


def test_reasoning_effort_is_sent_by_default() -> None:
    client, completions = _client_with_stub([_completion("ok")])

    _complete(client)
    assert completions.kwargs[0]["extra_body"] == {"reasoning": {"effort": "low"}}


def test_reasoning_effort_can_be_disabled() -> None:
    client = OpenRouterClient(
        OpenRouterConfig(api_key="secret", model="test/model", reasoning_effort=None)
    )
    completions = StubCompletions([_completion("ok")])
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    _complete(client)
    assert "extra_body" not in completions.kwargs[0]


def test_cooldown_blocks_other_threads_until_deadline() -> None:
    cooldown = _Cooldown()
    cooldown.pause_for(0.15)
    elapsed: list[float] = []

    def waiter() -> None:
        start = time.monotonic()
        cooldown.wait()
        elapsed.append(time.monotonic() - start)

    thread = threading.Thread(target=waiter)
    thread.start()
    thread.join()
    assert elapsed[0] >= 0.1
