from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

from .errors import ResolutionError
from .identity import canonical_json, directory_snapshot
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


def _verify_material(
    location: str,
    verification: JsonObject,
    *,
    label: str,
) -> None:
    path = Path(location)
    if not path.is_dir():
        raise ResolutionError(f"{label} is no longer available: {path}")
    kind = verification.get("kind")
    if kind == "directory-snapshot":
        if directory_snapshot(path) != verification.get("digest"):
            raise ResolutionError(
                f"{label} changed after Plan resolution: {path}"
            )
        return
    if kind == "huggingface-cache":
        if path.name != verification.get("commit"):
            raise ResolutionError(
                f"{label} cache revision changed after Plan resolution: {path}"
            )
        return
    raise ResolutionError(f"{label} has unsupported verification metadata")


def verify_model_materials(model: JsonObject) -> None:
    _verify_material(
        model["location"],
        model["verification"],
        label="model checkpoint",
    )
    if (
        model["tokenizer_location"] != model["location"]
        or model["tokenizer_verification"] != model["verification"]
    ):
        _verify_material(
            model["tokenizer_location"],
            model["tokenizer_verification"],
            label="model tokenizer",
        )


class LocalModelRuntime:
    """Single-machine implementation that keeps one local model resident."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._verified: set[str] = set()

    def _verify(self, model: JsonObject) -> None:
        key = canonical_json(
            {
                "location": model["location"],
                "verification": model["verification"],
                "tokenizer_location": model["tokenizer_location"],
                "tokenizer_verification": model["tokenizer_verification"],
            }
        )
        if key not in self._verified:
            verify_model_materials(model)
            self._verified.add(key)

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
        self._verify(model)
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

        key = repr(
            (
                "tokenizer",
                model["tokenizer_checkpoint"],
                model["tokenizer_revision"],
                padding_side,
            )
        )
        if self._key == key:
            return self._tokenizer
        self.release()
        self._verify(model)
        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_location"],
            padding_side=padding_side,
            local_files_only=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._key = key
        self._tokenizer = tokenizer
        return tokenizer

    def fill_mask(
        self,
        model: JsonObject,
        *,
        device: str,
        dtype: str,
    ) -> tuple[Any, Any]:
        import torch
        from transformers import (
            AutoModelForMaskedLM,
            AutoTokenizer,
            pipeline,
        )

        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if selected_device == "auto":
            selected_device = "cpu"
        key = repr(
            (
                "fill-mask",
                model["checkpoint"],
                model["revision"],
                model["tokenizer_checkpoint"],
                model["tokenizer_revision"],
                selected_device,
                dtype,
            )
        )
        if self._key == key:
            return self._model, self._tokenizer
        self.release()
        self._verify(model)
        tokenizer = AutoTokenizer.from_pretrained(
            model["tokenizer_location"],
            local_files_only=True,
        )
        loaded_model = AutoModelForMaskedLM.from_pretrained(
            model["location"],
            torch_dtype=_torch_dtype(dtype, selected_device),
            local_files_only=True,
        )
        loaded_model.to(selected_device)
        loaded_model.eval()
        unmasker = pipeline(
            "fill-mask",
            model=loaded_model,
            tokenizer=tokenizer,
            device=torch.device(selected_device),
        )
        self._key = key
        self._model = unmasker
        self._tokenizer = tokenizer
        return unmasker, tokenizer

    def encoder(self, model: JsonObject, *, device: str) -> Any:
        import torch
        from sentence_transformers import SentenceTransformer

        selected_device = (
            "cuda" if device == "auto" and torch.cuda.is_available() else device
        )
        if selected_device == "auto":
            selected_device = "cpu"
        key = repr(
            (
                "encoder",
                model["checkpoint"],
                model["revision"],
                selected_device,
            )
        )
        if self._key == key:
            return self._model
        self.release()
        self._verify(model)
        encoder = SentenceTransformer(
            model["location"],
            device=selected_device,
            local_files_only=True,
        )
        self._key = key
        self._model = encoder
        return encoder

    def release(self) -> None:
        import torch

        self._model = None
        self._tokenizer = None
        self._key = None
        self._verified.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
