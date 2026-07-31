from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from ..errors import PlanValidationError, ResolutionError
from ..identity import directory_snapshot
from ..models import ArtifactRef, JsonObject


def require(settings: JsonObject, fields: set[str], *, kind: str) -> None:
    missing = sorted(
        field for field in fields if field not in settings
    )
    if missing:
        raise PlanValidationError(
            f"{kind} stage is missing settings: {', '.join(missing)}"
        )


def validate_execution_settings(settings: JsonObject) -> None:
    if settings["device"] not in {"auto", "cpu", "cuda"}:
        raise PlanValidationError("device must be auto, cpu, or cuda")
    if settings["dtype"] not in {
        "auto",
        "float32",
        "float16",
        "bfloat16",
    }:
        raise PlanValidationError(
            "dtype must be auto, float32, float16, or bfloat16"
        )


def resolve_path(value: str, repository: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repository / path
    return path.resolve()


def _resolve_checkpoint(
    checkpoint: str,
    revision: str | None,
    repository: Path,
) -> tuple[str, str, str, JsonObject]:
    local = resolve_path(checkpoint, repository)
    if local.exists():
        if not local.is_dir():
            raise ResolutionError(
                f"model checkpoint must be a directory: {local}"
            )
        snapshot = directory_snapshot(local)
        if revision is not None and revision != snapshot:
            raise ResolutionError(
                f"local model {local} declared revision {revision!r}, but its "
                f"verified snapshot is {snapshot!r}"
            )
        return (
            checkpoint,
            snapshot,
            str(local),
            {"kind": "directory-snapshot", "digest": snapshot},
        )
    try:
        location = Path(
            snapshot_download(
                repo_id=checkpoint,
                revision=revision,
                local_files_only=True,
            )
        ).resolve()
    except Exception as error:
        detail = f" at revision {revision!r}" if revision else ""
        raise ResolutionError(
            f"model {checkpoint!r}{detail} is not available in the local "
            "Hugging Face cache; Plan checking never downloads models"
        ) from error
    commit = location.name
    return (
        checkpoint,
        commit,
        str(location),
        {"kind": "huggingface-cache", "commit": commit},
    )


def resolve_model(
    value: Any,
    *,
    models: JsonObject,
    repository: Path,
) -> JsonObject:
    requested = value
    if isinstance(value, str) and value in models:
        value = models[value]
    if isinstance(value, str):
        value = {"checkpoint": value}
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"model {requested!r} must resolve to a string or mapping"
        )
    unknown = sorted(
        set(value)
        - {
            "checkpoint",
            "revision",
            "tokenizer_checkpoint",
            "tokenizer_revision",
        }
    )
    if unknown:
        raise PlanValidationError(
            f"model {requested!r} has unknown fields: {', '.join(unknown)}"
        )
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise PlanValidationError(
            f"model {requested!r} needs a checkpoint"
        )
    (
        model_name,
        model_revision,
        location,
        model_verification,
    ) = _resolve_checkpoint(
        checkpoint, value.get("revision"), repository
    )
    tokenizer_checkpoint = value.get("tokenizer_checkpoint", checkpoint)
    tokenizer_requested_revision = value.get(
        "tokenizer_revision", value.get("revision")
    )
    if (
        tokenizer_checkpoint == checkpoint
        and tokenizer_requested_revision == value.get("revision")
    ):
        (
            tokenizer_name,
            tokenizer_revision,
            tokenizer_location,
            tokenizer_verification,
        ) = (
            model_name,
            model_revision,
            location,
            model_verification,
        )
    else:
        (
            tokenizer_name,
            tokenizer_revision,
            tokenizer_location,
            tokenizer_verification,
        ) = _resolve_checkpoint(
            tokenizer_checkpoint,
            tokenizer_requested_revision,
            repository,
        )
    return {
        "requested": requested,
        "checkpoint": model_name,
        "revision": model_revision,
        "location": location,
        "verification": model_verification,
        "tokenizer_checkpoint": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_location": tokenizer_location,
        "tokenizer_verification": tokenizer_verification,
    }


def _jsonl_records(
    path: Path,
    *,
    limit: int | None = None,
) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if limit is not None and len(records) >= limit:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ResolutionError(
                    f"{path}:{line_number}: invalid JSON"
                ) from error
            if not isinstance(record, dict):
                raise ResolutionError(
                    f"{path}:{line_number}: record must be an object"
                )
            records.append(record)
    return records


def artifact_records(artifact: ArtifactRef) -> list[JsonObject]:
    return _jsonl_records(artifact.path / "records.jsonl")


def batches(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise PlanValidationError("batch_size must be positive")
    return [
        values[index : index + size]
        for index in range(0, len(values), size)
    ]
