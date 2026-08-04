from __future__ import annotations

import math
import statistics
from collections import Counter
from typing import Any

from transformers import set_seed
from voprf_py import VoprfServer

from watermark_suite.attacks import (
    AdaptiveWatermarkForger,
    LocalColorOracle,
    VOPRFColorOracle,
    theoretical_green_probability,
    theoretical_queries_per_scored_token,
)

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..dataset_prompt import DATASET_PROMPTS, PromptSample
from ..errors import PlanValidationError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import (
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)


def describe(values: list[float | int]) -> JsonObject:
    finite = [
        value
        for value in values
        if not isinstance(value, float) or math.isfinite(value)
    ]
    if not finite:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    ordered = sorted(finite)
    p95_index = max(
        0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    )
    return {
        "mean": statistics.mean(finite),
        "median": statistics.median(finite),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_index],
    }


def build_sample_metrics(result: Any, detection: Any | None = None) -> JsonObject:
    effective_token_num = (
        detection.effective_token_num if detection is not None else None
    )
    green_ranks = [
        step.selected_rank
        for step in result.steps
        if step.selected_green is True
    ]
    metrics = {
        "generated_token_num": result.generated_token_num,
        "scored_position_num": result.scored_position_num,
        "oracle_query_count": result.oracle_query_count,
        "color_check_count": result.color_check_count,
        "oracle_round_count": (
            result.oracle_protocol_stats.server_round_count
            if result.oracle_protocol_stats is not None
            else result.oracle_query_count
        ),
        "queries_per_generated_token": result.queries_per_generated_token,
        "queries_per_scored_token": result.queries_per_scored_token,
        "color_checks_per_scored_token": (
            result.color_checks_per_scored_token
        ),
        "cache_hit_ratio": (
            result.cache_hit_count / result.color_check_count
            if result.color_check_count
            else 0.0
        ),
        "fallback_ratio": result.fallback_ratio,
        "selected_green_ratio": result.selected_green_ratio,
        "duplicate_selected_pair_ratio": (
            result.duplicate_selected_pair_num / result.scored_position_num
            if result.scored_position_num
            else 0.0
        ),
        "green_candidate_rank_histogram": {
            str(rank): count
            for rank, count in sorted(Counter(green_ranks).items())
        },
        "mean_green_candidate_rank": result.mean_green_candidate_rank,
        "local_model_perplexity": result.local_model_perplexity,
        "mean_selected_negative_log_likelihood": (
            result.mean_selected_negative_log_likelihood
        ),
        "mean_log_probability_gap": result.mean_log_probability_gap,
        "forgery_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "oracle_time_fraction": (
            result.oracle_time_seconds / result.elapsed_seconds
            if result.elapsed_seconds
            else 0.0
        ),
    }
    if detection is not None:
        metrics.update(
            {
                "detector_token_num": detection.total_token_num,
                "detector_effective_token_num": effective_token_num,
                "queries_per_honest_audit_query": (
                    result.oracle_query_count / effective_token_num
                    if effective_token_num
                    else 0.0
                ),
                "detector_green_ratio": detection.green_ratio,
                "effective_token_ratio": (
                    effective_token_num / detection.total_token_num
                    if detection.total_token_num
                    else 0.0
                ),
            }
        )
    return metrics


def _get(settings: Any, name: str) -> Any:
    return (
        settings[name]
        if isinstance(settings, dict)
        else getattr(settings, name)
    )


def build_summary(
    settings: Any,
    records: list[JsonObject],
    elapsed_seconds: float,
) -> JsonObject:
    forgery_records = [record["adaptive_forgery"] for record in records]
    sample_metrics = [record["sample_metrics"] for record in records]
    total_queries = sum(
        record["oracle_query_count"] for record in forgery_records
    )
    total_checks = sum(
        record["color_check_count"] for record in forgery_records
    )
    total_hits = sum(
        record["cache_hit_count"] for record in forgery_records
    )
    total_scored = sum(
        record["scored_token_num"] for record in forgery_records
    )
    total_scored_positions = sum(
        record["scored_position_num"] for record in forgery_records
    )
    total_green = sum(
        record["selected_green_token_num"] for record in forgery_records
    )
    total_fallbacks = sum(
        record["fallback_count"] for record in forgery_records
    )
    total_generated = sum(
        record["generated_token_num"] for record in forgery_records
    )
    total_duplicate = sum(
        record["duplicate_selected_pair_num"]
        for record in forgery_records
    )
    total_oracle_time = sum(
        record["oracle_time_seconds"] for record in forgery_records
    )
    total_forgery_time = sum(
        record["elapsed_seconds"] for record in forgery_records
    )
    protocol_stats = [
        record["oracle_protocol_stats"]
        for record in forgery_records
        if record["oracle_protocol_stats"] is not None
    ]
    communication_bytes = sum(
        record["total_communication_bytes"] for record in protocol_stats
    )
    gamma = _get(settings, "gamma")
    max_candidates = _get(settings, "max_candidates")
    theoretical_green = theoretical_green_probability(
        gamma, max_candidates
    )
    theoretical_queries = theoretical_queries_per_scored_token(
        gamma, max_candidates
    )
    observed_green = total_green / total_scored if total_scored else 0.0
    observed_queries = (
        total_queries / total_scored if total_scored else 0.0
    )
    summary: JsonObject = {
        "sample_num": len(records),
        "gamma": gamma,
        "window_size": _get(settings, "window_size"),
        "max_candidates": max_candidates,
        "theory_model": (
            "independent-bernoulli"
            if _get(settings, "method") == "vow"
            else "independent-bernoulli-approximation"
        ),
        "target_scored_pairs": _get(settings, "target_scored_pairs"),
        "theoretical_green_probability": theoretical_green,
        "theoretical_queries_per_scored_token": theoretical_queries,
        "observed_minus_theoretical_green_probability": (
            observed_green - theoretical_green
        ),
        "observed_to_theoretical_query_ratio": (
            observed_queries / theoretical_queries
            if theoretical_queries
            else None
        ),
        "oracle_query_count": total_queries,
        "color_check_count": total_checks,
        "cache_hit_count": total_hits,
        "oracle_round_count": sum(
            record["server_round_count"] for record in protocol_stats
        ),
        "generated_token_num": total_generated,
        "scored_token_num": total_scored,
        "scored_position_num": total_scored_positions,
        "duplicate_selected_pair_num": total_duplicate,
        "observed_queries_per_generated_token": (
            total_queries / total_generated if total_generated else 0.0
        ),
        "observed_queries_per_scored_token": observed_queries,
        "selected_green_ratio": observed_green,
        "fallback_ratio": (
            total_fallbacks / total_scored_positions
            if total_scored_positions
            else 0.0
        ),
        "cache_hit_ratio": (
            total_hits / total_checks if total_checks else 0.0
        ),
        "duplicate_selected_pair_ratio": (
            total_duplicate / total_scored_positions
            if total_scored_positions
            else 0.0
        ),
        "generated_token_length": describe(
            [record["generated_token_num"] for record in sample_metrics]
        ),
        "oracle_queries_per_sample": describe(
            [record["oracle_query_count"] for record in sample_metrics]
        ),
        "queries_per_scored_token_per_sample": describe(
            [record["queries_per_scored_token"] for record in sample_metrics]
        ),
        "local_model_perplexity": describe(
            [
                record["local_model_perplexity"]
                for record in sample_metrics
                if record["local_model_perplexity"] is not None
            ]
        ),
        "mean_log_probability_gap": describe(
            [
                record["mean_log_probability_gap"]
                for record in sample_metrics
                if record["mean_log_probability_gap"] is not None
            ]
        ),
        "total_oracle_time_seconds": total_oracle_time,
        "oracle_time_fraction": (
            total_oracle_time / total_forgery_time
            if total_forgery_time
            else 0.0
        ),
        "total_communication_bytes": communication_bytes,
        "communication_bytes_per_oracle_query": (
            communication_bytes / total_queries if total_queries else 0.0
        ),
        "total_elapsed_seconds": elapsed_seconds,
        "mean_forgery_seconds": (
            statistics.mean(
                record["elapsed_seconds"] for record in forgery_records
            )
            if forgery_records
            else 0.0
        ),
    }
    if records and all("detection" in record for record in records):
        detection_records = [record["detection"] for record in records]
        effective = sum(
            record["effective_token_num"] for record in detection_records
        )
        detected_green = sum(
            record["green_token_num"] for record in detection_records
        )
        p_values = [record["p_value"] for record in detection_records]
        summary.update(
            {
                "detector_token_num": sum(
                    record["total_token_num"]
                    for record in detection_records
                ),
                "detector_effective_token_num": effective,
                "query_overhead_vs_honest_audit": (
                    total_queries / effective if effective else 0.0
                ),
                "detector_green_ratio": (
                    detected_green / effective if effective else 0.0
                ),
                "median_p_value": (
                    statistics.median(p_values) if p_values else None
                ),
                "p_value": describe(p_values),
                "forgery_success_rate": {
                    f"{level:.0e}": (
                        sum(value < level for value in p_values)
                        / len(p_values)
                        if p_values
                        else 0.0
                    )
                    for level in _get(settings, "significance_levels")
                },
            }
        )
    return summary


class AdaptiveForgeryStageAdapter:
    kind = "adaptive-forgery"
    revision = "adaptive-forgery-v4"
    accepted_settings = {
        "model",
        "dataset",
        "num_samples",
        "batch_size",
        "seed",
        "max_new_tokens",
        "target_scored_pairs",
        "max_candidates",
        "watermark",
        "allow_special_tokens",
        "trace_level",
        "device",
        "dtype",
    }
    _required = accepted_settings - {"target_scored_pairs"}

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        validate_execution_settings(settings)
        for field in ("num_samples", "batch_size", "max_new_tokens"):
            if not isinstance(settings[field], int) or settings[field] <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if not isinstance(settings["seed"], int):
            raise PlanValidationError("seed must be an integer")
        if settings["max_candidates"] <= 0:
            raise PlanValidationError("max_candidates must be positive")
        target_scored_pairs = settings.get("target_scored_pairs")
        if (
            target_scored_pairs is not None
            and (
                not isinstance(target_scored_pairs, int)
                or target_scored_pairs <= 0
            )
        ):
            raise PlanValidationError(
                "target_scored_pairs must be a positive integer or null"
            )
        if settings["trace_level"] not in {"none", "compact", "full"}:
            raise PlanValidationError(
                "trace_level must be none, compact, or full"
            )
        model = resolve_model(
            settings["model"],
            models=context.models,
            repository=context.repository,
        )
        prompt_population = DATASET_PROMPTS.resolve(
            settings["dataset"],
            catalog=context.datasets,
            repository=context.repository,
            sample_num=settings["num_samples"],
        )
        watermark = WATERMARK_SCHEMES.resolve(
            settings["watermark"], context.repository
        )
        if watermark["method"] not in {"vow", "lefthash", "selfhash"}:
            raise PlanValidationError(
                "adaptive-forgery requires VOW, LeftHash, or SelfHash"
            )
        if not watermark["enabled"]:
            raise PlanValidationError(
                "adaptive-forgery requires watermark.enabled=true"
            )
        oracle_fields = (
            (
                "method",
                "enabled",
                "window_size",
                "gamma",
                "server_seed_path",
                "server_seed_sha256",
            )
            if watermark["method"] == "vow"
            else ("method", "enabled", "gamma")
        )
        oracle = {key: watermark[key] for key in oracle_fields}
        semantic = {
            "model": model,
            "prompt_population": prompt_population.to_dict(),
            "batch_size": settings["batch_size"],
            "seed": settings["seed"],
            "max_new_tokens": settings["max_new_tokens"],
            "target_scored_pairs": target_scored_pairs,
            "max_candidates": settings["max_candidates"],
            "watermark": oracle,
            "allow_special_tokens": settings["allow_special_tokens"],
            "trace_level": settings["trace_level"],
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
        }
        return ResolvedStageDefinition(
            settings={
                **settings,
                "model": model,
                "dataset": prompt_population.to_dict(),
            },
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision="adaptive-forgery-v4",
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
        if inputs:
            raise ValueError(
                "adaptive-forgery does not accept an input Artifact"
            )
        return definition

    def prepare(
        self, context: StageExecutionContext
    ) -> "_AdaptiveForgeryExecution":
        return _AdaptiveForgeryExecution(context)


class _AdaptiveForgeryExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        semantic = context.semantic_settings
        self.model, self.tokenizer = context.runtime.get(
            semantic["model"],
            device=context.execution_settings["device"],
            dtype=context.execution_settings["dtype"],
            padding_side="left",
        )
        watermark = semantic["watermark"]
        if watermark["method"] == "vow":
            server = VoprfServer(WATERMARK_SCHEMES.vow_seed(watermark))

            def server_interface(blinded_elements: list[Any]) -> Any:
                return server.batch_blind_evaluate(blinded_elements)

            oracle = VOPRFColorOracle(
                server_public_key=server.get_public_key(),
                server_interface=server_interface,
                gamma=watermark["gamma"],
            )
            prior_context_width = watermark["window_size"]
        else:
            kgw_detector, _ = WATERMARK_SCHEMES.detector(
                watermark,
                self.tokenizer,
                device=context.execution_settings["device"],
            )

            def color_interface(
                context_ids: tuple[int, ...], token_id: int
            ) -> bool:
                prefix = (
                    (*context_ids, token_id)
                    if kgw_detector.self_salt
                    else context_ids
                )
                return kgw_detector.is_green(prefix, token_id)

            oracle = LocalColorOracle(color_interface)
            prior_context_width = (
                kgw_detector.context_width - int(kgw_detector.self_salt)
            )
        self.forger = AdaptiveWatermarkForger(
            model=self.model,
            tokenizer=self.tokenizer,
            oracle=oracle,
            window_size=prior_context_width,
            max_candidates=semantic["max_candidates"],
        )
        self.samples = list(
            DATASET_PROMPTS.materialize(
                semantic["prompt_population"],
                tokenizer=self.tokenizer,
            ).samples
        )

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.samples,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item.sample_id for item in batch)
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
        records = []
        samples: list[PromptSample] = item.payload
        for sample in samples:
            sample_seed = derived_seed(
                semantic["seed"], sample.sample_id
            )
            set_seed(sample_seed)
            result = self.forger.forge(
                prompt=sample.model_prompt,
                max_new_tokens=semantic["max_new_tokens"],
                target_scored_pairs=semantic["target_scored_pairs"],
                suppress_token_ids=(
                    None
                    if semantic["allow_special_tokens"]
                    else self.tokenizer.all_special_ids
                ),
                stop_on_eos=True,
            )
            records.append(
                {
                    "sample_id": sample.sample_id,
                    "source_prompt": sample.source_prompt,
                    "prompt_text": sample.model_prompt,
                    "generated_text": result.text,
                    "adaptive_forgery": result.to_dict(
                        trace_level=semantic["trace_level"]
                    ),
                    "sample_metrics": build_sample_metrics(result),
                }
            )
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        semantic = self.context.semantic_settings
        settings = {
            "method": semantic["watermark"]["method"],
            "gamma": semantic["watermark"]["gamma"],
            "window_size": self.forger.window_size,
            "max_candidates": semantic["max_candidates"],
            "target_scored_pairs": semantic["target_scored_pairs"],
            "significance_levels": [0.00001],
        }
        elapsed = sum(
            record["adaptive_forgery"]["elapsed_seconds"]
            for record in records
        )
        return build_summary(settings, records, elapsed)

    def close(self) -> None:
        pass
