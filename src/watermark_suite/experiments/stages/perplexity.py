from __future__ import annotations

import math
import statistics
from typing import Any

from datasets import Dataset

from watermark_suite.core.metrics import calculate_perplexities

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from .common import (
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)
from .detection import _artifact_records
from .adaptive_forgery import describe


def distinct_n(token_ids: list[int], n: int) -> float:
    count = len(token_ids) - n + 1
    if count <= 0:
        return 0.0
    return (
        len(
            {
                tuple(token_ids[index : index + n])
                for index in range(count)
            }
        )
        / count
    )


def correlation(
    left: list[float], right: list[float]
) -> float | None:
    pairs = [
        (x, y)
        for x, y in zip(left, right)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return None
    x_values, y_values = zip(*pairs)
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None
    return statistics.correlation(x_values, y_values)


class PerplexityStageAdapter:
    kind = "perplexity"
    revision = "perplexity-v1"
    accepted_settings = {
        "model",
        "batch_size",
        "max_length",
        "prompt_field",
        "target_field",
        "device",
        "dtype",
    }
    _required = accepted_settings

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        validate_execution_settings(settings)
        if settings["batch_size"] <= 0:
            raise PlanValidationError("batch_size must be positive")
        if settings["max_length"] < 2:
            raise PlanValidationError("max_length must be at least 2")
        model = resolve_model(
            settings["model"],
            models=context.models,
            repository=context.repository,
        )
        semantic = {
            "model": model,
            "batch_size": settings["batch_size"],
            "max_length": settings["max_length"],
            "prompt_field": settings["prompt_field"],
            "target_field": settings["target_field"],
            "tokenizer_policy": "evaluator-owned",
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
        }
        return ResolvedStageDefinition(
            settings={**settings, "model": model},
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision="conditional-perplexity-v1",
            resource_key=(
                f"causal:{model['checkpoint']}@{model['revision']}:"
                f"{settings['device']}:{settings['dtype']}"
            ),
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if len(inputs) != 1:
            raise PlanValidationError(
                "perplexity requires exactly one source Artifact"
            )
        return definition

    def prepare(
        self, context: StageExecutionContext
    ) -> "_PerplexityExecution":
        return _PerplexityExecution(context)


class _PerplexityExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        semantic = context.semantic_settings
        self.model, self.tokenizer = context.runtime.get(
            semantic["model"],
            device=context.execution_settings["device"],
            dtype=context.execution_settings["dtype"],
            padding_side="right",
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

    def execute(self, item: WorkItem) -> WorkResult:
        semantic = self.context.semantic_settings
        prompt_field = semantic["prompt_field"]
        target_field = semantic["target_field"]
        missing = [
            record["sample_id"]
            for record in item.payload
            if prompt_field not in record or target_field not in record
        ]
        if missing:
            raise PlanValidationError(
                "source records lack perplexity fields "
                f"{prompt_field!r}/{target_field!r}: {missing[:5]}"
            )
        dataset = Dataset.from_list(
            [
                {
                    "prompt": record[prompt_field],
                    "target": record[target_field],
                }
                for record in item.payload
            ]
        )
        evaluation = calculate_perplexities(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=dataset,
            prompt_column="prompt",
            target_column="target",
            batch_size=len(item.payload),
            max_length=semantic["max_length"],
            device=context_device(self.context.execution_settings["device"]),
        )
        records = []
        for index, source in enumerate(item.payload):
            text = source[target_field]
            token_ids = self.tokenizer.encode(
                text, add_special_tokens=False
            )
            record = {
                "sample_id": source["sample_id"],
                "source_artifact": self.context.inputs[0].identity.value,
                "prompt_text": source[prompt_field],
                "generated_text": text,
                "quality": {
                    "conditional_perplexity": (
                        evaluation.sample_perplexities[index]
                    ),
                    "mean_negative_log_likelihood": (
                        evaluation.sample_negative_log_likelihoods[index]
                    ),
                    "perplexity_target_token_num": (
                        evaluation.sample_target_token_nums[index]
                    ),
                    "evaluation_token_num": len(token_ids),
                    "character_num": len(text),
                    "characters_per_token": (
                        len(text) / len(token_ids) if token_ids else 0.0
                    ),
                    "distinct_1": distinct_n(token_ids, 1),
                    "distinct_2": distinct_n(token_ids, 2),
                    "distinct_4": distinct_n(token_ids, 4),
                },
            }
            if "sample_metrics" in source:
                record["sample_metrics"] = source["sample_metrics"]
            if "detection" in source:
                record["detection"] = source["detection"]
            records.append(record)
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        total_tokens = sum(
            record["quality"]["perplexity_target_token_num"]
            for record in records
        )
        total_nll = sum(
            record["quality"]["mean_negative_log_likelihood"]
            * record["quality"]["perplexity_target_token_num"]
            for record in records
            if record["quality"]["mean_negative_log_likelihood"] is not None
        )
        mean_nll = (
            total_nll / total_tokens if total_tokens else float("inf")
        )
        perplexity = (
            math.exp(mean_nll)
            if math.isfinite(mean_nll) and mean_nll < 709
            else float("inf")
        )
        sample_ppl = [
            record["quality"]["conditional_perplexity"]
            for record in records
            if math.isfinite(
                record["quality"]["conditional_perplexity"]
            )
        ]
        summary: JsonObject = {
            "sample_num": len(records),
            "evaluation_model": self.context.semantic_settings["model"],
            "conditional_perplexity": perplexity,
            "mean_negative_log_likelihood": mean_nll,
            "perplexity_target_token_num": total_tokens,
            "sample_conditional_perplexity": describe(sample_ppl),
            "mean_sample_conditional_perplexity": (
                statistics.mean(sample_ppl) if sample_ppl else None
            ),
            "evaluation_token_length": describe(
                [
                    record["quality"]["evaluation_token_num"]
                    for record in records
                ]
            ),
            "characters_per_token": describe(
                [
                    record["quality"]["characters_per_token"]
                    for record in records
                ]
            ),
            "distinct_1": describe(
                [
                    record["quality"]["distinct_1"]
                    for record in records
                ]
            ),
            "distinct_2": describe(
                [
                    record["quality"]["distinct_2"]
                    for record in records
                ]
            ),
            "distinct_4": describe(
                [
                    record["quality"]["distinct_4"]
                    for record in records
                ]
            ),
            "mean_distinct_1": (
                statistics.mean(
                    record["quality"]["distinct_1"]
                    for record in records
                )
                if records
                else None
            ),
            "mean_distinct_2": (
                statistics.mean(
                    record["quality"]["distinct_2"]
                    for record in records
                )
                if records
                else None
            ),
        }
        if records and all(
            "sample_metrics" in record for record in records
        ):
            finite_ppl = [
                float(record["quality"]["conditional_perplexity"])
                for record in records
            ]
            metrics = [record["sample_metrics"] for record in records]
            correlations = {
                "perplexity_vs_queries_per_scored_token": correlation(
                    finite_ppl,
                    [
                        float(item["queries_per_scored_token"])
                        for item in metrics
                    ],
                ),
                "perplexity_vs_selected_green_ratio": correlation(
                    finite_ppl,
                    [
                        float(item["selected_green_ratio"])
                        for item in metrics
                    ],
                ),
                "perplexity_vs_fallback_ratio": correlation(
                    finite_ppl,
                    [float(item["fallback_ratio"]) for item in metrics],
                ),
            }
            if all("detection" in record for record in records):
                correlations["perplexity_vs_log10_p_value"] = correlation(
                    finite_ppl,
                    [
                        math.log10(
                            max(
                                float(record["detection"]["p_value"]),
                                1e-300,
                            )
                        )
                        for record in records
                    ],
                )
            local_ppl = [
                (
                    float(item["local_model_perplexity"])
                    if item.get("local_model_perplexity") is not None
                    else float("nan")
                )
                for item in metrics
            ]
            correlations[
                "evaluator_perplexity_vs_local_model_perplexity"
            ] = correlation(finite_ppl, local_ppl)
            summary["quality_cost_correlations"] = correlations
        return summary

    def close(self) -> None:
        pass


def context_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"
