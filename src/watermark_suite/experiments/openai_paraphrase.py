from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .models import JsonObject


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _integer(value: Any, name: str) -> int:
    result = _field(value, name, 0)
    return int(result) if result is not None else 0


def _usage(response: Any) -> JsonObject:
    usage = _field(response, "usage")
    if usage is None:
        return {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
        }
    input_details = _field(usage, "input_tokens_details", {})
    output_details = _field(usage, "output_tokens_details", {})
    return {
        "input_tokens": _integer(usage, "input_tokens"),
        "cached_input_tokens": _integer(input_details, "cached_tokens"),
        "cache_write_input_tokens": _integer(
            input_details,
            "cache_write_tokens",
        ),
        "output_tokens": _integer(usage, "output_tokens"),
        "reasoning_output_tokens": _integer(
            output_details,
            "reasoning_tokens",
        ),
        "total_tokens": _integer(usage, "total_tokens"),
    }


@dataclass(frozen=True)
class OpenAIParaphraseResult:
    text: str
    provenance: JsonObject


class OpenAIParaphraser:
    """Responses API boundary for reproducible paraphrase transformations."""

    def __init__(
        self,
        client: Any = None,
        *,
        max_attempts: int = 4,
        initial_retry_delay_seconds: float = 2.0,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        if initial_retry_delay_seconds < 0:
            raise ValueError(
                "initial_retry_delay_seconds must not be negative"
            )
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client
        self._max_attempts = max_attempts
        self._initial_retry_delay_seconds = initial_retry_delay_seconds

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code in {408, 409, 429} or status_code >= 500
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
        }

    def paraphrase(
        self,
        text: str,
        *,
        model: str,
        instruction: str,
        max_output_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
        request_identity: str | None = None,
    ) -> OpenAIParaphraseResult:
        request: JsonObject = {
            "model": model,
            "instructions": instruction,
            "input": text,
            "max_output_tokens": max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            request["temperature"] = temperature
        if reasoning_effort is not None:
            request["reasoning"] = {"effort": reasoning_effort}

        started = time.perf_counter()
        response = None
        attempts = 0
        while attempts < self._max_attempts:
            attempts += 1
            try:
                response = self._client.responses.create(**request)
                break
            except Exception as error:
                if (
                    attempts >= self._max_attempts
                    or not self._is_retryable(error)
                ):
                    raise RuntimeError(
                        "OpenAI paraphrase request failed "
                        f"after {attempts} application attempt(s) "
                        f"(model={model!r}, "
                        f"sample_id={request_identity!r}): {error}"
                    ) from error
                delay = min(
                    self._initial_retry_delay_seconds
                    * (2 ** (attempts - 1)),
                    30.0,
                )
                time.sleep(delay)
        assert response is not None
        latency_seconds = time.perf_counter() - started
        status = _field(response, "status")
        if status != "completed":
            raise RuntimeError(
                "OpenAI paraphrase response did not complete "
                f"(id={_field(response, 'id')!r}, status={status!r})"
            )
        output_text = _field(response, "output_text", "")
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError(
                "OpenAI paraphrase response contained no output text "
                f"(id={_field(response, 'id')!r})"
            )
        return OpenAIParaphraseResult(
            text=output_text.strip(),
            provenance={
                "provider": "openai",
                "endpoint": "responses",
                "requested_model": model,
                "response_model": _field(response, "model"),
                "response_id": _field(response, "id"),
                "response_status": status,
                "response_created_at": _field(response, "created_at"),
                "service_tier": _field(response, "service_tier"),
                "reasoning_effort": reasoning_effort,
                "application_attempts": attempts,
                "latency_seconds": latency_seconds,
                "usage": _usage(response),
            },
        )

    def paraphrase_many(
        self,
        texts: list[str],
        *,
        concurrency: int,
        model: str,
        instruction: str,
        max_output_tokens: int,
        temperature: float | None,
        reasoning_effort: str | None,
        request_identities: list[str] | None = None,
    ) -> list[OpenAIParaphraseResult]:
        if request_identities is None:
            request_identities = [None] * len(texts)
        elif len(request_identities) != len(texts):
            raise ValueError(
                "request_identities must have the same length as texts"
            )

        def call(item: tuple[str, str | None]) -> OpenAIParaphraseResult:
            text, request_identity = item
            return self.paraphrase(
                text,
                model=model,
                instruction=instruction,
                max_output_tokens=max_output_tokens,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                request_identity=request_identity,
            )

        requests = list(zip(texts, request_identities))
        if len(texts) <= 1:
            return [call(request) for request in requests]
        with ThreadPoolExecutor(
            max_workers=min(concurrency, len(texts))
        ) as executor:
            return list(executor.map(call, requests))
