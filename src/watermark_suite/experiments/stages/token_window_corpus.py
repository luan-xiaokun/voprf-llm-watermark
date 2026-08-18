from __future__ import annotations

import hashlib
import shutil
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkFile, WorkItem, WorkResult
from .common import require, resolve_model


def _resolved_dataset(value: Any, datasets: JsonObject) -> JsonObject:
    requested = value
    if isinstance(value, str) and value in datasets:
        value = datasets[value]
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"streaming dataset {requested!r} must resolve to a mapping"
        )
    allowed = {
        "kind",
        "path",
        "config",
        "split",
        "revision",
        "text_field",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise PlanValidationError(
            "streaming dataset has unknown fields: " + ", ".join(unknown)
        )
    if value.get("kind") != "huggingface-stream":
        raise PlanValidationError(
            "token-window-corpus requires dataset.kind=huggingface-stream"
        )
    for field in ("path", "config", "split", "revision"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise PlanValidationError(
                f"streaming dataset requires a non-empty {field}"
            )
    text_field = value.get("text_field", "text")
    if not isinstance(text_field, str) or not text_field:
        raise PlanValidationError("streaming dataset text_field must be non-empty")
    return {**value, "text_field": text_field}


def load_streaming_dataset(source: JsonObject) -> Any:
    """Production adapter at the remote Hugging Face streaming seam."""

    from datasets import IterableDataset, load_dataset

    dataset = load_dataset(
        source["path"],
        source["config"],
        split=source["split"],
        revision=source["revision"],
        streaming=True,
    )
    if not isinstance(dataset, IterableDataset):
        raise PlanValidationError(
            "streaming C4 loader returned a materialized Dataset"
        )
    return dataset


class TokenWindowCorpusStageAdapter:
    kind = "token-window-corpus"
    revision = "token-window-corpus-v1"
    accepted_settings = {
        "dataset",
        "tokenizer",
        "sample_num",
        "window_token_num",
        "max_windows_per_document",
        "selection_seed",
        "shuffle_buffer_size",
        "stream_batch_size",
        "checkpoint_sample_num",
    }
    _required = accepted_settings

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        for field in (
            "sample_num",
            "window_token_num",
            "max_windows_per_document",
            "shuffle_buffer_size",
            "stream_batch_size",
            "checkpoint_sample_num",
        ):
            value = settings[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if not isinstance(settings["selection_seed"], int) or isinstance(
            settings["selection_seed"], bool
        ):
            raise PlanValidationError("selection_seed must be an integer")
        dataset = _resolved_dataset(settings["dataset"], context.datasets)
        tokenizer = resolve_model(
            settings["tokenizer"],
            models=context.models,
            repository=context.repository,
        )
        selection = {
            "mode": "streaming-buffer-shuffle-first-eligible",
            "seed": settings["selection_seed"],
            "shuffle_buffer_size": settings["shuffle_buffer_size"],
        }
        window = {
            "token_num": settings["window_token_num"],
            "max_per_document": settings["max_windows_per_document"],
            "stride": settings["window_token_num"],
            "drop_incomplete": True,
        }
        population = {
            "kind": "c4-token-windows",
            "dataset": dataset,
            "tokenizer": {
                "checkpoint": tokenizer["tokenizer_checkpoint"],
                "revision": tokenizer["tokenizer_revision"],
            },
            "selection": selection,
            "window": window,
            "sample_num": settings["sample_num"],
        }
        population["identity"] = identity_for(population, prefix="population")
        semantic = {
            "dataset": dataset,
            "tokenizer": tokenizer,
            "sample_num": settings["sample_num"],
            "selection": selection,
            "window": window,
            "population": population,
        }
        return ResolvedStageDefinition(
            settings={**settings, "dataset": dataset, "tokenizer": tokenizer},
            semantic_settings=semantic,
            execution_settings={
                "stream_batch_size": settings["stream_batch_size"],
                "checkpoint_sample_num": settings["checkpoint_sample_num"],
                "worker_count": 1,
                "max_in_flight": 1,
                "resume_compatibility": {
                    "checkpoint_sample_num": settings["checkpoint_sample_num"]
                },
            },
            artifact_schema_revision="token-window-corpus-v1",
            resource_key=(
                "tokenizer:"
                f"{tokenizer['tokenizer_checkpoint']}@"
                f"{tokenizer['tokenizer_revision']}"
            ),
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if inputs:
            raise PlanValidationError("token-window-corpus accepts no inputs")
        return definition

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> "_TokenWindowCorpusExecution":
        return _TokenWindowCorpusExecution(context)


class _TokenWindowCorpusExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        tokenizer_model = {
            **context.semantic_settings["tokenizer"],
            "checkpoint": context.semantic_settings["tokenizer"][
                "tokenizer_checkpoint"
            ],
            "revision": context.semantic_settings["tokenizer"][
                "tokenizer_revision"
            ],
            "location": context.semantic_settings["tokenizer"][
                "tokenizer_location"
            ],
            "verification": context.semantic_settings["tokenizer"][
                "tokenizer_verification"
            ],
        }
        self.tokenizer = context.runtime.tokenizer(
            tokenizer_model, padding_side="right"
        )
        self.pending = (
            context.workspace
            / "attempts"
            / context.attempt_identity
            / "pending-token-windows"
        )
        self.pending.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _token_digest(token_ids: list[int]) -> str:
        encoded = np.asarray(token_ids, dtype="<u4").tobytes()
        return hashlib.sha256(encoded).hexdigest()

    def _windows(self) -> Iterator[JsonObject]:
        semantic = self.context.semantic_settings
        source = semantic["dataset"]
        selection = semantic["selection"]
        window = semantic["window"]
        dataset = load_streaming_dataset(source).shuffle(
            seed=selection["seed"],
            buffer_size=selection["shuffle_buffer_size"],
        )
        stream_batch_size = self.context.execution_settings["stream_batch_size"]
        text_field = source["text_field"]
        buffered: list[tuple[int, JsonObject, str]] = []

        def encode_buffer() -> Iterator[JsonObject]:
            if not buffered:
                return
            tokenized = self.tokenizer(
                [text for _, _, text in buffered],
                add_special_tokens=False,
                padding=False,
                truncation=False,
            )["input_ids"]
            token_num = window["token_num"]
            for (source_ordinal, item, _), token_ids in zip(buffered, tokenized):
                for index, start in enumerate(
                    range(0, len(token_ids) - token_num + 1, window["stride"])
                ):
                    if index >= window["max_per_document"]:
                        break
                    selected = [int(value) for value in token_ids[start : start + token_num]]
                    if any(value < 0 or value >= 2**32 for value in selected):
                        raise PlanValidationError(
                            "token-window-corpus only supports uint32 token ids"
                        )
                    digest = self._token_digest(selected)
                    locator = "\n".join(
                        str(item.get(field, ""))
                        for field in ("url", "timestamp")
                    )
                    locator_hash = hashlib.sha256(
                        locator.encode("utf-8")
                    ).hexdigest()
                    sample_id = identity_for(
                        {
                            "population": semantic["population"]["identity"],
                            "source_ordinal": source_ordinal,
                            "token_start": start,
                            "token_sha256": digest,
                        },
                        prefix="c4-window",
                    )
                    yield {
                        "sample_id": sample_id,
                        "source_ordinal": source_ordinal,
                        "source_locator_sha256": locator_hash,
                        "token_start": start,
                        "token_sha256": digest,
                        "token_ids": selected,
                    }

        for source_ordinal, item in enumerate(dataset):
            text = item.get(text_field)
            if not isinstance(text, str) or not text.strip():
                continue
            buffered.append((source_ordinal, dict(item), text))
            if len(buffered) >= stream_batch_size:
                yield from encode_buffer()
                buffered.clear()
        if buffered:
            yield from encode_buffer()

    def work_items(self) -> Iterable[WorkItem]:
        target = self.context.semantic_settings["sample_num"]
        block_size = self.context.execution_settings["checkpoint_sample_num"]
        block: list[JsonObject] = []
        produced = 0
        ordinal = 0
        for window in self._windows():
            block.append(window)
            produced += 1
            if len(block) >= block_size or produced >= target:
                sample_ids = tuple(item["sample_id"] for item in block)
                yield WorkItem(
                    identity=identity_for(
                        {"ordinal": ordinal, "sample_ids": sample_ids},
                        prefix="token-window-block",
                    ),
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=tuple(block),
                )
                ordinal += 1
                block = []
            if produced >= target:
                return
        raise PlanValidationError(
            f"streaming dataset ended after {produced} eligible token windows; "
            f"requested {target}"
        )

    def execute(self, item: WorkItem) -> WorkResult:
        import pyarrow as pa
        import pyarrow.parquet as pq

        token_num = self.context.semantic_settings["window"]["token_num"]
        rows = list(item.payload)
        table = pa.table(
            {
                "sample_id": pa.array(
                    [row["sample_id"] for row in rows], type=pa.string()
                ),
                "source_ordinal": pa.array(
                    [row["source_ordinal"] for row in rows], type=pa.int64()
                ),
                "source_locator_sha256": pa.array(
                    [row["source_locator_sha256"] for row in rows],
                    type=pa.string(),
                ),
                "token_start": pa.array(
                    [row["token_start"] for row in rows], type=pa.int32()
                ),
                "token_sha256": pa.array(
                    [row["token_sha256"] for row in rows], type=pa.string()
                ),
                "token_ids": pa.array(
                    [row["token_ids"] for row in rows],
                    type=pa.list_(pa.uint32(), token_num),
                ),
            }
        )
        name = f"part-{item.ordinal:08d}-{item.identity}.parquet"
        source_path = self.pending / name
        pq.write_table(
            table,
            source_path,
            compression="zstd",
            use_dictionary=False,
            row_group_size=len(rows),
        )
        artifact_path = f"parts/{name}"
        return WorkResult(
            records=(
                {
                    "sample_id": item.identity,
                    "part": artifact_path,
                    "sample_num": len(rows),
                    "first_sample_id": rows[0]["sample_id"],
                    "last_sample_id": rows[-1]["sample_id"],
                },
            ),
            files=(
                WorkFile(
                    source_path=source_path,
                    artifact_path=artifact_path,
                ),
            ),
        )

    def summarize(self, records: Iterable[JsonObject]) -> JsonObject:
        records = list(records)
        sample_num = sum(int(record["sample_num"]) for record in records)
        expected = self.context.semantic_settings["sample_num"]
        if sample_num != expected:
            raise PlanValidationError(
                f"token-window-corpus produced {sample_num} samples, expected {expected}"
            )
        return {
            "sample_num": sample_num,
            "part_num": len(records),
            "population": self.context.semantic_settings["population"],
            "dataset": self.context.semantic_settings["dataset"],
            "window": self.context.semantic_settings["window"],
            "retention": {
                "format": "parquet",
                "compression": "zstd",
                "raw_text_retained": False,
                "token_ids_retained": True,
            },
        }

    def close(self) -> None:
        if self.pending.is_dir():
            shutil.rmtree(self.pending)
