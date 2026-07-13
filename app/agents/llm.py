"""LLM client abstraction.

Agents depend on the `LLMClient` protocol, not a vendor SDK, so tests can
inject fakes and the provider can be swapped. The default implementation
uses OpenAI chat completions with JSON-mode output (gpt-4o-mini, matching
the rest of this project), with bounded retries and token accounting.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Protocol

log = logging.getLogger("returns.agents.llm")


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict:
        """Run one completion and return the parsed JSON object."""
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
            self._client = OpenAI(max_retries=0)
        return self._client

    def complete_json(self, system: str, user: str) -> dict:
        client = self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                completion = client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
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
                return json.loads(completion.choices[0].message.content)
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
