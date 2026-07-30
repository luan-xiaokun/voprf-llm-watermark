from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import snapshot_download

from ..errors import PlanValidationError, ResolutionError
from ..identity import identity_for, sha256_file
from ..models import JsonObject


ELI5_SYSTEM_MESSAGE = (
    "Explain the following question in about 500 words like I'm 5 years old. "
    "Use very simple language, short sentences, and analogies a child can "
    "understand."
)


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


def _directory_snapshot(path: Path) -> str:
    files = sorted(item for item in path.rglob("*") if item.is_file())
    document = [
        {
            "path": item.relative_to(path).as_posix(),
            "size": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    return identity_for(document, prefix="snapshot")


def _resolve_checkpoint(
    checkpoint: str,
    revision: str | None,
    repository: Path,
) -> tuple[str, str, str]:
    local = resolve_path(checkpoint, repository)
    if local.exists():
        if not local.is_dir():
            raise ResolutionError(
                f"model checkpoint must be a directory: {local}"
            )
        selected_revision = revision or _directory_snapshot(local)
        return checkpoint, selected_revision, str(local)
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
    return checkpoint, commit, str(location)


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
    model_name, model_revision, location = _resolve_checkpoint(
        checkpoint, value.get("revision"), repository
    )
    tokenizer_checkpoint = value.get("tokenizer_checkpoint", checkpoint)
    tokenizer_name, tokenizer_revision, tokenizer_location = (
        _resolve_checkpoint(
            tokenizer_checkpoint,
            value.get("tokenizer_revision", value.get("revision")),
            repository,
        )
    )
    return {
        "requested": requested,
        "checkpoint": model_name,
        "revision": model_revision,
        "location": location,
        "tokenizer_checkpoint": tokenizer_name,
        "tokenizer_revision": tokenizer_revision,
        "tokenizer_location": tokenizer_location,
    }


def _dataset_source(
    value: Any, datasets: JsonObject
) -> tuple[Any, str | None]:
    alias = value if isinstance(value, str) and value in datasets else None
    return (datasets[value], value) if alias else (value, None)


def _jsonl_records(path: Path) -> list[JsonObject]:
    records: list[JsonObject] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
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


def _load_dataset_records(spec: JsonObject) -> list[JsonObject]:
    if spec["format"] == "jsonl":
        return _jsonl_records(Path(spec["path"]))
    dataset = load_dataset(spec["path"], split=spec["split"])
    return [dict(item) for item in dataset]


def _sample_identity(
    record: JsonObject,
    *,
    id_field: str | None,
    dataset_kind: str,
) -> str:
    if id_field and id_field in record:
        return f"{dataset_kind}:{record[id_field]}"
    content = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return f"{dataset_kind}:{digest}"


def resolve_dataset(
    value: Any,
    *,
    datasets: JsonObject,
    repository: Path,
    num_samples: int,
) -> JsonObject:
    raw, alias = _dataset_source(value, datasets)
    if isinstance(raw, str):
        raw = {
            "kind": raw,
            "format": "jsonl" if raw == "c4" else "huggingface",
            "path": (
                "data/c4_realnewslike_subset_1000.jsonl"
                if raw == "c4"
                else "data/eli5"
            ),
        }
    if not isinstance(raw, dict):
        raise PlanValidationError(
            f"dataset {value!r} must resolve to a string or mapping"
        )
    unknown = sorted(
        set(raw)
        - {
            "kind",
            "format",
            "path",
            "split",
            "prompt_field",
            "sample_id_field",
        }
    )
    if unknown:
        raise PlanValidationError(
            f"dataset {value!r} has unknown fields: {', '.join(unknown)}"
        )
    kind = raw.get("kind")
    data_format = raw.get("format")
    path_value = raw.get("path")
    if kind not in {"c4", "eli5"}:
        raise PlanValidationError(
            f"dataset {value!r} kind must be c4 or eli5"
        )
    if data_format not in {"jsonl", "huggingface"}:
        raise PlanValidationError(
            f"dataset {value!r} format must be jsonl or huggingface"
        )
    if not isinstance(path_value, str):
        raise PlanValidationError(f"dataset {value!r} needs a path")
    path = resolve_path(path_value, repository)
    if not path.exists():
        raise ResolutionError(f"dataset path does not exist: {path}")
    snapshot = (
        sha256_file(path) if path.is_file() else _directory_snapshot(path)
    )
    prompt_field = raw.get(
        "prompt_field", "prompt_text" if kind == "c4" else "question"
    )
    id_field = raw.get(
        "sample_id_field", "original_index" if kind == "c4" else None
    )
    resolved = {
        "alias": alias,
        "kind": kind,
        "format": data_format,
        "path": str(path),
        "split": raw.get("split", "train"),
        "prompt_field": prompt_field,
        "sample_id_field": id_field,
        "snapshot": snapshot,
    }
    records = _load_dataset_records(resolved)
    if num_samples <= 0:
        raise PlanValidationError("num_samples must be positive")
    selected = records[: min(num_samples, len(records))]
    if len(selected) != num_samples:
        raise PlanValidationError(
            f"dataset contains {len(records)} records, fewer than "
            f"num_samples={num_samples}"
        )
    sample_ids = [
        _sample_identity(
            record, id_field=id_field, dataset_kind=kind
        )
        for record in selected
    ]
    if len(sample_ids) != len(set(sample_ids)):
        raise PlanValidationError(
            f"dataset {value!r} yields duplicate Sample identities"
        )
    resolved["selection"] = {
        "mode": "first",
        "count": num_samples,
        "sample_ids": sample_ids,
    }
    return resolved


def load_selected_samples(spec: JsonObject) -> list[JsonObject]:
    path = Path(spec["path"])
    current_snapshot = (
        sha256_file(path) if path.is_file() else _directory_snapshot(path)
    )
    if current_snapshot != spec["snapshot"]:
        raise ResolutionError(
            f"dataset snapshot changed after Plan resolution: {path}"
        )
    records = _load_dataset_records(spec)
    count = spec["selection"]["count"]
    selected = records[:count]
    actual_ids = [
        _sample_identity(
            record,
            id_field=spec["sample_id_field"],
            dataset_kind=spec["kind"],
        )
        for record in selected
    ]
    if actual_ids != spec["selection"]["sample_ids"]:
        raise ResolutionError("selected Sample manifest no longer matches")
    return [
        {"sample_id": sample_id, **record}
        for sample_id, record in zip(actual_ids, selected)
    ]


def format_prompt(
    sample: JsonObject, dataset: JsonObject, tokenizer: Any
) -> tuple[str, str]:
    source_prompt = str(sample[dataset["prompt_field"]])
    if dataset["kind"] != "eli5":
        return source_prompt, source_prompt
    if getattr(tokenizer, "chat_template", None) is not None:
        model_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ELI5_SYSTEM_MESSAGE},
                {"role": "user", "content": source_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        model_prompt = (
            f"{ELI5_SYSTEM_MESSAGE}\n\nQuestion: {source_prompt}\nAnswer:"
        )
    return source_prompt, model_prompt


def batches(values: list[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise PlanValidationError("batch_size must be positive")
    return [
        values[index : index + size]
        for index in range(0, len(values), size)
    ]
