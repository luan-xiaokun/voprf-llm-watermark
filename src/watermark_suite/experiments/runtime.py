from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .errors import ResolutionError
from .models import JsonObject


def _torch_dtype(name: str, device: str) -> Any:
    import torch

    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ResolutionError(f"unknown dtype {name!r}")
    return torch.bfloat16 if device == "cuda" else "auto"


class LocalModelRuntime:
    """Single-machine implementation that keeps one causal model resident."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None

    def get(
        self,
        model: JsonObject,
        *,
        device: str,
        dtype: str,
        padding_side: str,
    ) -> tuple[Any, Any]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if selected_device == "auto":
            selected_device = "cpu"
        key = repr(
            (
                model["checkpoint"],
                model["revision"],
                model["tokenizer_checkpoint"],
                model["tokenizer_revision"],
                selected_device,
                dtype,
                padding_side,
            )
        )
        if self._key == key:
            return self._model, self._tokenizer
        self.release()
        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_location"],
            padding_side=padding_side,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        loaded_model = AutoModelForCausalLM.from_pretrained(
            model["location"],
            torch_dtype=_torch_dtype(dtype, selected_device),
            local_files_only=True,
        )
        loaded_model.to(selected_device)
        loaded_model.eval()
        self._key = key
        self._model = loaded_model
        self._tokenizer = tokenizer
        return loaded_model, tokenizer

    def tokenizer(
        self, model: JsonObject, *, padding_side: str = "left"
    ) -> Any:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_location"],
            padding_side=padding_side,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def release(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        self._key = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
