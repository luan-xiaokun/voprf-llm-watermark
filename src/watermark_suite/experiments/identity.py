from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .errors import ResolutionError


def canonicalize(value: Any) -> Any:
    """Return a JSON-compatible value with deterministic mapping order."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ResolutionError("identity values cannot contain NaN or infinity")
        return {"$float": value.hex()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ResolutionError("identity mapping keys must be strings")
        return {
            key: canonicalize(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    raise ResolutionError(
        f"identity value {value!r} has unsupported type {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def identity_for(value: Any, *, prefix: str) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest}"


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def derived_seed(seed: int, identity: str) -> int:
    payload = f"{seed}:{identity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")
