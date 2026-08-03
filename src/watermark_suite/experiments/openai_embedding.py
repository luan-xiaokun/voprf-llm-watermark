from __future__ import annotations

from typing import Any


class OpenAIEmbedder:
    """Small Embeddings API boundary used by text evaluation stages."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI()
        self._client = client

    def embed_many(
        self,
        texts: list[str],
        *,
        model: str,
        dimensions: int | None,
    ) -> list[list[float]]:
        if not texts:
            return []
        request: dict[str, Any] = {
            "model": model,
            "input": texts,
            "encoding_format": "float",
        }
        if dimensions is not None:
            request["dimensions"] = dimensions
        response = self._client.embeddings.create(**request)
        data = getattr(response, "data", None)
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError(
                "OpenAI embedding response count does not match its input"
            )
        ordered = sorted(data, key=lambda item: int(getattr(item, "index")))
        embeddings = [getattr(item, "embedding", None) for item in ordered]
        if any(not isinstance(value, list) or not value for value in embeddings):
            raise RuntimeError(
                "OpenAI embedding response contains an invalid vector"
            )
        return embeddings
