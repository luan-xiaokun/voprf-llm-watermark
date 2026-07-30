from __future__ import annotations

import inspect
import math
import statistics
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import torch

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from .common import batches, require
from .watermarks import detector_for


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _artifact_records(artifact: ArtifactRef) -> list[JsonObject]:
    from ..engine import _read_jsonlines

    return list(_read_jsonlines(artifact.path / "records.jsonl"))


class DetectionStageAdapter:
    kind = "detection"
    revision = "detection-v1"
    accepted_settings = {
        "batch_size",
        "target_field",
        "token_num",
        "significance_levels",
        "use_local",
        "step_size",
        "device",
    }
    _required = accepted_settings

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        del context
        require(settings, self._required, kind=self.kind)
        if settings["batch_size"] <= 0:
            raise PlanValidationError("batch_size must be positive")
        levels = settings["significance_levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or any(not 0 < value < 1 for value in levels)
        ):
            raise PlanValidationError(
                "significance_levels must be a non-empty list between 0 and 1"
            )
        semantic = {
            key: settings[key]
            for key in (
                "batch_size",
                "target_field",
                "token_num",
                "significance_levels",
                "use_local",
                "step_size",
            )
        }
        return ResolvedStageDefinition(
            settings=dict(settings),
            semantic_settings=semantic,
            execution_settings={"device": settings["device"]},
            artifact_schema_revision="watermark-detection-v1",
            resource_key="input-bound:detection",
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if len(inputs) != 1:
            raise PlanValidationError(
                "detection requires exactly one source Artifact"
            )
        source = inputs[0]
        source_semantic = source.manifest.get("semantic_settings", {})
        watermark = source_semantic.get("watermark")
        model = source_semantic.get("model")
        if not isinstance(watermark, dict) or not isinstance(model, dict):
            raise PlanValidationError(
                "source Artifact does not declare watermark/model provenance"
            )
        if watermark.get("method") == "none":
            raise PlanValidationError(
                "detection cannot derive a detector from an unwatermarked "
                "source Artifact"
            )
        semantic = {
            **definition.semantic_settings,
            "watermark": watermark,
            "tokenizer": {
                key: model[key]
                for key in (
                    "tokenizer_checkpoint",
                    "tokenizer_revision",
                    "tokenizer_location",
                )
            },
        }
        return ResolvedStageDefinition(
            settings={
                **definition.settings,
                "derived_watermark": watermark,
                "derived_tokenizer": semantic["tokenizer"],
            },
            semantic_settings=semantic,
            execution_settings=definition.execution_settings,
            artifact_schema_revision=definition.artifact_schema_revision,
            resource_key=(
                "detector:"
                f"{semantic['tokenizer']['tokenizer_checkpoint']}@"
                f"{semantic['tokenizer']['tokenizer_revision']}:"
                f"{watermark['method']}"
            ),
        )

    def prepare(
        self, context: StageExecutionContext
    ) -> "_DetectionExecution":
        return _DetectionExecution(context)


class _DetectionExecution:
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
        }
        self.tokenizer = context.runtime.tokenizer(
            tokenizer_model, padding_side="left"
        )
        self.detector, self.detector_kwargs = detector_for(
            context.semantic_settings["watermark"],
            self.tokenizer,
            device=context.execution_settings["device"],
        )
        self.records = _artifact_records(context.inputs[0])

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.records,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item["sample_id"] for item in batch)
            identity = identity_for(
                {"ordinal": ordinal, "sample_ids": sample_ids},
                prefix="batch",
            )
            result.append(
                WorkItem(
                    identity=identity,
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=batch,
                )
            )
        return result

    def _detect(self, texts: list[str]) -> tuple[list[Any], list[Any] | None]:
        semantic = self.context.semantic_settings
        kwargs = {
            "token_num": semantic["token_num"],
        }
        signature_target = self.detector.batch_detect
        if (
            semantic["watermark"]["method"] == "vow"
            and semantic["use_local"]
        ):
            signature_target = self.detector.local_batch_detect
        parameters = inspect.signature(signature_target).parameters
        if "step_size" in parameters:
            kwargs["step_size"] = semantic["step_size"]
        if "include_p_values_per_token" in parameters:
            kwargs["include_p_values_per_token"] = (
                semantic["step_size"] is not None
            )
        if (
            semantic["watermark"]["method"] == "vow"
            and not semantic["use_local"]
        ):
            server = self.detector.voprf_server
            kwargs.update(
                {
                    "server_public_key": server.get_public_key(),
                    "server_interface": server.batch_blind_evaluate,
                }
            )
        output = signature_target(texts, **kwargs)
        if isinstance(output, tuple):
            return output
        return output, None

    def execute(self, item: WorkItem) -> WorkResult:
        target = self.context.semantic_settings["target_field"]
        missing = [
            record["sample_id"]
            for record in item.payload
            if target not in record
        ]
        if missing:
            raise PlanValidationError(
                f"source records lack target_field={target!r}: {missing[:5]}"
            )
        texts = [record[target] for record in item.payload]
        results, costs = self._detect(texts)
        records = []
        for index, (source, result) in enumerate(
            zip(item.payload, results)
        ):
            detection = _json_value(result)
            if costs is not None:
                detection["cost"] = _json_value(costs[index])
            record = {
                "sample_id": source["sample_id"],
                "source_artifact": self.context.inputs[0].identity.value,
                "prompt_text": source.get("prompt_text", ""),
                "generated_text": source[target],
                "detection": detection,
            }
            if "source_prompt" in source:
                record["source_prompt"] = source["source_prompt"]
            if "sample_metrics" in source:
                metrics = dict(source["sample_metrics"])
                effective = detection.get("effective_token_num")
                if effective is not None:
                    queries = metrics.get("oracle_query_count", 0)
                    metrics["detector_effective_token_num"] = effective
                    metrics["detector_token_num"] = detection[
                        "total_token_num"
                    ]
                    metrics["queries_per_honest_audit_query"] = (
                        queries / effective if effective else 0.0
                    )
                    metrics["detector_green_ratio"] = detection.get(
                        "green_ratio"
                    )
                record["sample_metrics"] = metrics
            records.append(record)
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        levels = self.context.semantic_settings["significance_levels"]
        p_values = [
            float(record["detection"]["p_value"]) for record in records
        ]
        total_tokens = sum(
            int(record["detection"]["total_token_num"])
            for record in records
        )
        rates = {
            f"{level:.0e}": (
                sum(value < level for value in p_values) / len(p_values)
                if p_values
                else 0.0
            )
            for level in levels
        }
        rate_name = (
            "tpr"
            if self.context.semantic_settings["watermark"]["enabled"]
            else "fpr"
        )
        summary: JsonObject = {
            "sample_num": len(records),
            "watermark": self.context.semantic_settings["watermark"],
            "p_value_median": (
                statistics.median(p_values) if p_values else None
            ),
            "detection_rate": rates,
            rate_name: rates,
            "total_token_num": total_tokens,
            "mean_token_num": (
                total_tokens / len(records) if records else 0.0
            ),
        }
        if records and all(
            "effective_token_num" in record["detection"]
            for record in records
        ):
            effective = sum(
                int(record["detection"]["effective_token_num"])
                for record in records
            )
            green = sum(
                int(record["detection"]["green_token_num"])
                for record in records
            )
            summary.update(
                {
                    "effective_token_num": effective,
                    "green_token_num": green,
                    "green_ratio": green / effective if effective else 0.0,
                }
            )
            if all("sample_metrics" in record for record in records):
                queries = sum(
                    record["sample_metrics"]["oracle_query_count"]
                    for record in records
                )
                summary["oracle_query_count"] = queries
                summary["query_overhead_vs_honest_audit"] = (
                    queries / effective if effective else 0.0
                )
        return summary

    def close(self) -> None:
        pass
