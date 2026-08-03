from __future__ import annotations

import inspect
import math
import statistics
import sys
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
import scipy.stats
import torch
from scipy import special

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import artifact_records, batches, require


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


def _distribution(values: list[float | int]) -> JsonObject:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p05": None,
            "p25": None,
            "p75": None,
            "p95": None,
        }
    ordered = sorted(finite)

    def percentile(fraction: float) -> float:
        index = max(
            0,
            min(
                len(ordered) - 1,
                math.ceil(fraction * len(ordered)) - 1,
            ),
        )
        return ordered[index]

    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p05": percentile(0.05),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
    }


def minimum_green_count(
    *,
    method: str,
    gamma: float,
    scored_pair_num: int,
    significance_level: float,
) -> int:
    """Smallest green count accepted by the configured detector test."""

    if method not in {"vow", "lefthash", "selfhash"}:
        raise ValueError(f"unsupported green-count detector {method!r}")
    for green_count in range(scored_pair_num + 1):
        if method == "vow":
            p_value = (
                1.0
                if green_count == 0
                else float(
                    special.betainc(
                        green_count,
                        scored_pair_num - green_count + 1,
                        gamma,
                    )
                )
            )
        else:
            z_score = (
                green_count - gamma * scored_pair_num
            ) / math.sqrt(scored_pair_num * gamma * (1.0 - gamma))
            p_value = float(scipy.stats.norm.sf(z_score))
        if p_value < significance_level:
            return green_count
    return scored_pair_num + 1


def adaptive_forgery_curve(
    records: list[JsonObject],
    significance_levels: list[float],
) -> list[JsonObject]:
    """Aggregate axes-ready query cost and attack success by token length."""

    by_milestone: dict[int, list[JsonObject]] = {}
    for record in records:
        forgery = record.get("adaptive_forgery")
        detection = record.get("detection", {})
        if not isinstance(forgery, dict):
            continue
        trace = forgery.get("trace")
        if not isinstance(trace, list):
            continue
        for milestone, p_value in zip(
            detection.get("milestones") or [],
            detection.get("step_p_values") or [],
        ):
            token_num = int(milestone)
            prefix = [
                step
                for step in trace
                if int(step.get("position", -1)) < token_num
            ]
            query_count = (
                int(prefix[-1]["cumulative_oracle_query_count"])
                if prefix
                else 0
            )
            scored = [
                step
                for step in prefix
                if step.get("selected_green") is not None
            ]
            green_count = sum(
                step.get("selected_green") is True for step in scored
            )
            by_milestone.setdefault(token_num, []).append(
                {
                    "query_count": query_count,
                    "green_count": green_count,
                    "green_ratio": (
                        green_count / len(scored) if scored else 0.0
                    ),
                    "p_value": float(p_value),
                }
            )
    return [
        {
            "token_num": token_num,
            "eligible_sample_num": len(values),
            "mean_oracle_query_count": statistics.mean(
                value["query_count"] for value in values
            ),
            "median_oracle_query_count": statistics.median(
                value["query_count"] for value in values
            ),
            "mean_queries_per_token": statistics.mean(
                value["query_count"] / token_num for value in values
            ),
            "mean_selected_green_token_count": statistics.mean(
                value["green_count"] for value in values
            ),
            "mean_selected_green_ratio": statistics.mean(
                value["green_ratio"] for value in values
            ),
            "median_p_value": statistics.median(
                value["p_value"] for value in values
            ),
            "attack_success_rate": {
                f"{level:.0e}": (
                    sum(value["p_value"] < level for value in values)
                    / len(values)
                )
                for level in significance_levels
            },
            "attack_success_counts": {
                f"{level:.0e}": {
                    "positive_num": sum(
                        value["p_value"] < level for value in values
                    ),
                    "sample_num": len(values),
                }
                for level in significance_levels
            },
        }
        for token_num, values in sorted(by_milestone.items())
    ]


class DetectionStageAdapter:
    kind = "detection"
    revision = "detection-v5"
    accepted_settings = {
        "batch_size",
        "target_field",
        "token_num",
        "significance_levels",
        "use_local",
        "step_size",
        "detector_watermark",
        "device",
    }
    _required = accepted_settings - {"detector_watermark"}

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
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
        detector_watermark = settings.get("detector_watermark")
        if detector_watermark is not None:
            detector_watermark = WATERMARK_SCHEMES.resolve(
                detector_watermark,
                context.repository,
            )
            if (
                detector_watermark["method"] == "none"
                or not detector_watermark["enabled"]
            ):
                raise PlanValidationError(
                    "detector_watermark must identify an enabled detector"
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
        semantic["detector_watermark"] = detector_watermark
        return ResolvedStageDefinition(
            settings={
                **settings,
                "detector_watermark": detector_watermark,
            },
            semantic_settings=semantic,
            execution_settings={"device": settings["device"]},
            artifact_schema_revision="watermark-detection-v4",
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
        source_watermark = source_semantic.get("watermark")
        model = source_semantic.get("model")
        if (
            not isinstance(source_watermark, dict)
            or not isinstance(model, dict)
        ):
            raise PlanValidationError(
                "source Artifact does not declare watermark/model provenance"
            )
        detector_watermark = definition.semantic_settings.get(
            "detector_watermark"
        )
        if detector_watermark is None:
            detector_watermark = source_watermark
        if detector_watermark.get("method") == "none":
            raise PlanValidationError(
                "detection cannot derive a detector from an unwatermarked "
                "source Artifact without detector_watermark"
            )
        semantic = {
            **definition.semantic_settings,
            "watermark": source_watermark,
            "detector_watermark": detector_watermark,
            "tokenizer": {
                key: model[key]
                for key in (
                    "tokenizer_checkpoint",
                    "tokenizer_revision",
                    "tokenizer_location",
                    "tokenizer_verification",
                )
            },
        }
        if detector_watermark.get("method") == "upv":
            # UPV exposes a classifier decision rather than p-values. The
            # fixed, calibrated operating point is carried by its detector
            # material, so generic p-value thresholds are not semantic inputs
            # to this bound Run.
            semantic["significance_levels"] = []
        return ResolvedStageDefinition(
            settings={
                **definition.settings,
                "derived_watermark": source_watermark,
                "derived_detector_watermark": detector_watermark,
                "derived_tokenizer": semantic["tokenizer"],
            },
            semantic_settings=semantic,
            execution_settings=definition.execution_settings,
            artifact_schema_revision=definition.artifact_schema_revision,
            resource_key=(
                "tokenizer:"
                f"{semantic['tokenizer']['tokenizer_checkpoint']}@"
                f"{semantic['tokenizer']['tokenizer_revision']}"
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
            "verification": context.semantic_settings["tokenizer"][
                "tokenizer_verification"
            ],
        }
        self.tokenizer = context.runtime.tokenizer(
            tokenizer_model, padding_side="left"
        )
        self.detector, self.detector_kwargs = WATERMARK_SCHEMES.detector(
            context.semantic_settings["detector_watermark"],
            self.tokenizer,
            device=context.execution_settings["device"],
        )
        self.records = artifact_records(context.inputs[0])

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
        detector_watermark = semantic["detector_watermark"]
        kwargs = {
            "token_num": semantic["token_num"],
        }
        signature_target = self.detector.batch_detect
        if (
            detector_watermark["method"] == "vow"
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
            detector_watermark["method"] == "vow"
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
            if "adaptive_forgery" in source:
                record["adaptive_forgery"] = source["adaptive_forgery"]
            records.append(record)
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        levels = self.context.semantic_settings["significance_levels"]
        watermark = self.context.semantic_settings["watermark"]
        detector_watermark = self.context.semantic_settings.get(
            "detector_watermark", watermark
        )
        total_tokens = sum(
            int(record["detection"]["total_token_num"])
            for record in records
        )
        rate_name = (
            "tpr"
            if watermark["enabled"]
            else "fpr"
        )
        summary: JsonObject = {
            "sample_num": len(records),
            "watermark": watermark,
            "detector_watermark": detector_watermark,
            "total_token_num": total_tokens,
            "mean_token_num": (
                total_tokens / len(records) if records else 0.0
            ),
        }
        if detector_watermark["method"] == "upv":
            operating_point = detector_watermark.get(
                "detection_operating_point"
            )
            if not isinstance(operating_point, dict):
                raise PlanValidationError(
                    "UPV detection requires a resolved operating point"
                )
            score_type = operating_point["score_type"]
            threshold = float(operating_point["decision_threshold"])
            operator = operating_point["decision_operator"]
            operating_point_id = f"{score_type}{operator}{threshold:g}"
            predictions = []
            confidences = []
            milestone_predictions: dict[int, list[bool]] = {}
            for record in records:
                detection = record["detection"]
                confidence = float(detection["confidence"])
                predicted = bool(detection["predicted"])
                if (
                    detection.get("p_value") is not None
                    or detection.get("score_type") != score_type
                    or float(detection["decision_threshold"]) != threshold
                    or predicted != (confidence > threshold)
                ):
                    raise PlanValidationError(
                        "UPV result disagrees with its classifier operating point"
                    )
                predictions.append(predicted)
                confidences.append(confidence)
                milestones = detection.get("milestones") or []
                step_scores = detection.get("step_scores") or []
                step_predictions = detection.get("step_predictions") or []
                if not (
                    len(milestones)
                    == len(step_scores)
                    == len(step_predictions)
                ):
                    raise PlanValidationError(
                        "UPV milestone scores and decisions must align"
                    )
                for milestone, score, decision in zip(
                    milestones,
                    step_scores,
                    step_predictions,
                ):
                    if bool(decision) != (float(score) > threshold):
                        raise PlanValidationError(
                            "UPV milestone decision disagrees with its score"
                        )
                    milestone_predictions.setdefault(
                        int(milestone), []
                    ).append(bool(decision))
            positive_num = sum(predictions)
            rates = {
                operating_point_id: (
                    positive_num / len(predictions) if predictions else 0.0
                )
            }
            counts = {
                operating_point_id: {
                    "positive_num": positive_num,
                    "sample_num": len(predictions),
                }
            }
            summary.update(
                {
                    "detection_operating_points": {
                        operating_point_id: operating_point,
                    },
                    "classifier_confidence_distribution": _distribution(
                        confidences
                    ),
                    "detection_rate": rates,
                    "detection_counts": counts,
                    rate_name: rates,
                }
            )
            if milestone_predictions:
                summary["milestone_detection_rate"] = [
                    {
                        "token_num": token_num,
                        "eligible_sample_num": len(values),
                        "detection_rate": {
                            operating_point_id: sum(values) / len(values)
                        },
                        "detection_counts": {
                            operating_point_id: {
                                "positive_num": sum(values),
                                "sample_num": len(values),
                            }
                        },
                    }
                    for token_num, values in sorted(
                        milestone_predictions.items()
                    )
                ]
        else:
            p_values = [
                float(record["detection"]["p_value"])
                for record in records
            ]
            rates = {
                f"{level:.0e}": (
                    sum(value < level for value in p_values) / len(p_values)
                    if p_values
                    else 0.0
                )
                for level in levels
            }
            counts = {
                f"{level:.0e}": {
                    "positive_num": sum(value < level for value in p_values),
                    "sample_num": len(p_values),
                }
                for level in levels
            }
            summary.update(
                {
                    "detection_operating_points": {
                        f"{level:.0e}": {
                            "kind": "p-value-threshold",
                            "score_type": "p_value",
                            "decision_operator": "<",
                            "decision_threshold": level,
                            "target_fpr": level,
                        }
                        for level in levels
                    },
                    "p_value_median": (
                        statistics.median(p_values) if p_values else None
                    ),
                    "p_value_distribution": _distribution(p_values),
                    "negative_log10_p_value_distribution": _distribution(
                        [
                            -math.log10(max(value, sys.float_info.min))
                            for value in p_values
                        ]
                    ),
                    "detection_rate": rates,
                    "detection_counts": counts,
                    rate_name: rates,
                }
            )
            milestone_values: dict[int, list[float]] = {}
            for record in records:
                detection = record["detection"]
                milestones = detection.get("milestones") or []
                step_p_values = detection.get("step_p_values") or []
                for milestone, p_value in zip(milestones, step_p_values):
                    milestone_values.setdefault(int(milestone), []).append(
                        float(p_value)
                    )
            if milestone_values:
                summary["milestone_detection_rate"] = [
                    {
                        "token_num": token_num,
                        "eligible_sample_num": len(values),
                        "detection_rate": {
                            f"{level:.0e}": (
                                sum(value < level for value in values)
                                / len(values)
                            )
                            for level in levels
                        },
                        "detection_counts": {
                            f"{level:.0e}": {
                                "positive_num": sum(
                                    value < level for value in values
                                ),
                                "sample_num": len(values),
                            }
                            for level in levels
                        },
                    }
                    for token_num, values in sorted(
                        milestone_values.items()
                    )
                ]
        if records and all(
            "adaptive_forgery" in record for record in records
        ):
            summary["attack_success_rate"] = rates
            summary["adaptive_forgery_curve"] = adaptive_forgery_curve(
                records,
                levels,
            )
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
                    "green_token_count_distribution": _distribution(
                        [
                            int(record["detection"]["green_token_num"])
                            for record in records
                        ]
                    ),
                    "effective_token_count_distribution": _distribution(
                        [
                            int(record["detection"]["effective_token_num"])
                            for record in records
                        ]
                    ),
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
                summary["oracle_query_count_distribution"] = _distribution(
                    [
                        record["sample_metrics"]["oracle_query_count"]
                        for record in records
                    ]
                )
            effective_counts = {
                int(record["detection"]["effective_token_num"])
                for record in records
            }
            method = detector_watermark["method"]
            if (
                len(effective_counts) == 1
                and method in {"vow", "lefthash", "selfhash"}
            ):
                scored_pair_num = next(iter(effective_counts))
                if scored_pair_num > 0:
                    summary["green_count_thresholds"] = {
                        f"{level:.0e}": minimum_green_count(
                            method=method,
                            gamma=float(detector_watermark["gamma"]),
                            scored_pair_num=scored_pair_num,
                            significance_level=level,
                        )
                        for level in levels
                    }
        return summary

    def close(self) -> None:
        pass
