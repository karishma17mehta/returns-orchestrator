"""LLM client abstraction.

Agents depend on the `LLMClient` protocol, not a vendor SDK, so tests can
inject fakes and the provider can be swapped. `complete()` returns an
instance of the given pydantic model: the OpenAI implementation uses
structured outputs (`chat.completions.parse`), so the schema is enforced
by the API itself — malformed JSON is not a failure mode the callers need
to handle. Transient failures are retried with exponential backoff.

Set LANGSMITH_TRACING=true (plus LANGSMITH_API_KEY) to trace every call
in LangSmith; the client is wrapped only when enabled and the `langsmith`
package is installed.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Protocol, TypeVar

from pydantic import BaseModel

log = logging.getLogger("returns.agents.llm")

TModel = TypeVar("TModel", bound=BaseModel)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, output_model: type[TModel]) -> TModel:
        """Run one completion, returning a validated output_model instance."""
        ...


class LLMUnavailableError(Exception):
    """Raised when no LLM backend is configured (e.g. missing API key)."""


class UsageTracker:
    """Thread-safe token/call accounting shared across agents."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.calls += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "calls": self.calls,
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            }


class OpenAIClient:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.usage = UsageTracker()
        self._client = None

    def _get_client(self):
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise LLMUnavailableError(
                    "OPENAI_API_KEY is not set; agent review requires an LLM backend"
                )
            from openai import OpenAI

            # SDK-level retries off; we handle retry/backoff ourselves.
            client = OpenAI(max_retries=0)
            if os.environ.get("LANGSMITH_TRACING", "").lower() == "true":
                try:
                    from langsmith.wrappers import wrap_openai

                    client = wrap_openai(client)
                    log.info("LangSmith tracing enabled for OpenAI calls")
                except ImportError:
                    log.warning(
                        "LANGSMITH_TRACING is set but the langsmith package is "
                        "not installed; tracing disabled"
                    )
            self._client = client
        return self._client

    def complete(self, system: str, user: str, output_model: type[TModel]) -> TModel:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.parse(
                    model=self.model,
                    temperature=self.temperature,
                    response_format=output_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                if completion.usage:
                    self.usage.record(
                        completion.usage.prompt_tokens,
                        completion.usage.completion_tokens,
                    )
                message = completion.choices[0].message
                if message.parsed is None:
                    raise RuntimeError(
                        f"model refused structured output: {message.refusal!r}"
                    )
                return message.parsed
            except Exception as e:
                last_error = e
                wait = self.backoff_base_s * (2**attempt)
                log.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries, e, wait,
                )
                if attempt + 1 < self.max_retries:
                    time.sleep(wait)
        raise last_error
