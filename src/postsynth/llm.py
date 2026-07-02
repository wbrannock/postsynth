from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from tenacity import (
    RetryCallState,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from postsynth.config import OpenRouterConfig

RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_MAX_WAIT_SECONDS = 30.0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    truncated: bool = False


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str | LLMResponse:
        ...


class _Cooldown:
    """Shared pause gate so concurrent workers back off together after a 429."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pause_until = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._pause_until - time.monotonic()
            if delay <= 0:
                return
            time.sleep(delay)

    def pause_for(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, seconds)
        with self._lock:
            self._pause_until = max(self._pause_until, deadline)


def _retry_after_seconds(error: RateLimitError) -> float | None:
    response = getattr(error, "response", None)
    if response is None:
        return None
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


class OpenRouterClient:
    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        load_dotenv()
        self.config = config or OpenRouterConfig()
        api_key = self.config.api_key or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required. Set it in the environment or a local .env file."
            )

        headers: dict[str, str] = {"X-OpenRouter-Title": self.config.app_title}
        if self.config.http_referer:
            headers["HTTP-Referer"] = self.config.http_referer

        self._client = OpenAI(
            api_key=api_key,
            base_url=self.config.base_url,
            default_headers=headers,
            timeout=self.config.timeout_seconds,
            max_retries=0,  # tenacity below is the single retry layer
        )
        self._cooldown = _Cooldown()

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        attempts = max(1, self.config.max_retries + 1)
        default_wait = wait_exponential(multiplier=1, min=1, max=8)
        rate_limit_wait = wait_exponential(multiplier=1, min=1, max=RATE_LIMIT_MAX_WAIT_SECONDS)

        def _stop(state: RetryCallState) -> bool:
            limit = (
                max(attempts, RATE_LIMIT_MAX_ATTEMPTS)
                if isinstance(state.outcome.exception(), RateLimitError)
                else attempts
            )
            return state.attempt_number >= limit

        def _wait(state: RetryCallState) -> float:
            error = state.outcome.exception()
            if isinstance(error, RateLimitError):
                seconds = _retry_after_seconds(error)
                if seconds is None:
                    seconds = rate_limit_wait(state)
                self._cooldown.pause_for(seconds)
                # The cooldown gate does the actual waiting, shared across threads.
                return 0.0
            return default_wait(state)

        create_kwargs: dict[str, object] = {}
        if self.config.reasoning_effort:
            create_kwargs["extra_body"] = {
                "reasoning": {"effort": self.config.reasoning_effort}
            }

        @retry(stop=_stop, wait=_wait, reraise=True)
        def _call() -> LLMResponse:
            self._cooldown.wait()
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                **create_kwargs,  # type: ignore[arg-type]
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content:
                raise RuntimeError("OpenRouter returned an empty response")
            return LLMResponse(content=content, truncated=choice.finish_reason == "length")

        return _call()
