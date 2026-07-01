from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from postsynth.config import OpenRouterConfig


class LLMClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        ...


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
        )

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        attempts = max(1, self.config.max_retries + 1)

        @retry(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            reraise=True,
        )
        def _call() -> str:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError("OpenRouter returned an empty response")
            return content

        return _call()

