from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

from ..identity import identity_for
from ..models import ArtifactIdentity, ArtifactRef, JsonObject
from ..scheme_registry import WATERMARK_SCHEMES
from ._core import (
    Distribution,
    ExactCount,
    InterpretationIssue,
    MetricFact,
    MetricScope,
    ScientificDimensions,
    _ParsedArtifact,
    _RawMetricFact,
    _SchemaAdapter,
)


_DISTRIBUTION_FIELDS = frozenset(
    {
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "p05",
        "p25",
        "p75",
        "p95",
    }
)


def _model(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    checkpoint = value.get("checkpoint")
    revision = value.get("revision")
    if not isinstance(checkpoint, str) or not isinstance(revision, str):
        return None
    result = {"checkpoint": checkpoint, "revision": revision}
    tokenizer_checkpoint = value.get("tokenizer_checkpoint")
    tokenizer_revision = value.get("tokenizer_revision")
    if isinstance(tokenizer_checkpoint, str):
        result["tokenizer_checkpoint"] = tokenizer_checkpoint
    if isinstance(tokenizer_revision, str):
        result["tokenizer_revision"] = tokenizer_revision
    return result


def _population(value: object) -> JsonObject | None:
    if not isinstance(value, dict):
        return None
    selection = value.get("selection")
    if not isinstance(selection, dict):
        return None
    sample_ids = selection.get("sample_ids")
    count = selection.get("count")
    if not isinstance(sample_ids, list) or not all(
        isinstance(item, str) for item in sample_ids
    ):
        return None
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("dataset Sample identities must be unique")
    sample_num = int(count) if count is not None else len(sample_ids)
    if sample_num != len(sample_ids):
        raise ValueError(
            "dataset Sample count disagrees with selected Sample identities"
        )
    document = {
        "kind": value.get("kind") or value.get("name"),
        "split": value.get("split"),
        "snapshot": value.get("snapshot"),
        "sample_ids": sample_ids,
    }
    return {
        "identity": identity_for(document, prefix="population"),
        "kind": document["kind"],
        "split": document["split"],
        "sample_num": sample_num,
    }


def _transformation(value: object) -> JsonObject | None:
    if not isinstance(value, dict) or not isinstance(value.get("method"), str):
        return None
    parameters = {
        key: item
        for key, item in value.items()
        if key not in {"method", "instruction"}
    }
    if isinstance(value.get("instruction"), str):
        parameters["instruction_identity"] = identity_for(
            value["instruction"],
            prefix="instruction",
        )
    return {
        "method": value["method"],
        "parameters": parameters,
        "identity": identity_for(value, prefix="transformation"),
    }


def _generation(
    semantic: JsonObject,
    population: JsonObject | None,
    prompt_population: JsonObject | None,
) -> JsonObject:
    return {
        "model": _model(semantic.get("model")),
        "decoding": semantic.get("generation") or semantic.get("decoding"),
        "prompt_policy": (
            prompt_population.get("prompt_policy")
            if prompt_population is not None
            else None
        ),
        "prompt_generation": (
            prompt_population.get("generation")
            if prompt_population is not None
            else None
        ),
        "seed": semantic.get("seed"),
        "repetitions": semantic.get("repetitions"),
        "max_new_tokens": semantic.get("max_new_tokens"),
        "target_scored_pairs": semantic.get("target_scored_pairs"),
        "max_candidates": semantic.get("max_candidates"),
        "allow_special_tokens": semantic.get("allow_special_tokens"),
        "task_evaluation": semantic.get("task_evaluation"),
        "population_identity": (
            population["identity"] if population is not None else None
        ),
    }


def _scheme(value: object) -> JsonObject | None:
    return (
        WATERMARK_SCHEMES.analysis_dimensions(value)
        if isinstance(value, dict)
        else None
    )


def _distribution(
    value: object,
    *,
    default_count: int | None = None,
) -> Distribution | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("distribution must be a mapping")
    unknown = set(value) - _DISTRIBUTION_FIELDS
    if unknown:
        raise ValueError(
            "distribution has unknown fields: " + ", ".join(sorted(unknown))
        )
    if "count" not in value and default_count is None:
        return None
    return Distribution(
        count=int(value.get("count", default_count)),
        mean=_optional_float(value.get("mean")),
        median=_optional_float(value.get("median")),
        std=_optional_float(value.get("std")),
        minimum=_optional_float(value.get("min")),
        maximum=_optional_float(value.get("max")),
        p05=_optional_float(value.get("p05")),
        p25=_optional_float(value.get("p25")),
        p75=_optional_float(value.get("p75")),
        p95=_optional_float(value.get("p95")),
    )


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)


def _summary_count(summary: JsonObject) -> int | None:
    value = summary.get("sample_num")
    return int(value) if value is not None else None


def _facts_from_counts(
    values: object,
    *,
    metric: str,
    token_num: int | None = None,
    rates: object = None,
    operating_points: object = None,
) -> list[_RawMetricFact]:
    if not isinstance(values, dict):
        raise ValueError(f"{metric} counts must be a mapping")
    result = []
    if rates is not None and not isinstance(rates, dict):
        raise ValueError(f"{metric} rates must be a mapping")
    if isinstance(rates, dict) and set(rates) != set(values):
        raise ValueError(
            f"{metric} rate/count thresholds disagree: "
            f"rates={sorted(rates)}, counts={sorted(values)}"
        )
    if operating_points is not None:
        if not isinstance(operating_points, dict):
            raise ValueError("detection_operating_points must be a mapping")
        if set(operating_points) != set(values):
            raise ValueError(
                f"{metric} operating-point/count identifiers disagree: "
                f"operating_points={sorted(operating_points)}, "
                f"counts={sorted(values)}"
            )
    for threshold, count_value in values.items():
        if not isinstance(count_value, dict):
            raise ValueError(f"{metric} count for {threshold} must be a mapping")
        count = ExactCount(
            sample_num=int(count_value["sample_num"]),
            positive_num=int(count_value["positive_num"]),
        )
        operating_point = None
        if isinstance(operating_points, dict):
            raw_operating_point = operating_points[threshold]
            if not isinstance(raw_operating_point, dict):
                raise ValueError(
                    f"detection operating point {threshold!r} must be a mapping"
                )
            operating_point = dict(raw_operating_point)
            kind = operating_point.get("kind")
            score_type = operating_point.get("score_type")
            operator = operating_point.get("decision_operator")
            decision_threshold = operating_point.get("decision_threshold")
            if (
                kind not in {"p-value-threshold", "classifier-threshold"}
                or not isinstance(score_type, str)
                or operator not in {"<", ">"}
                or not isinstance(decision_threshold, (int, float))
                or isinstance(decision_threshold, bool)
                or not math.isfinite(float(decision_threshold))
            ):
                raise ValueError(
                    f"detection operating point {threshold!r} is invalid"
                )
            target_value = operating_point.get("target_fpr")
            target_fpr = (
                float(target_value) if target_value is not None else None
            )
            if kind == "p-value-threshold":
                if (
                    score_type != "p_value"
                    or operator != "<"
                    or target_fpr is None
                    or not 0 < target_fpr < 1
                ):
                    raise ValueError(
                        "p-value operating points require a target FPR"
                    )
            else:
                if target_fpr is not None:
                    raise ValueError(
                        "classifier operating points cannot claim target FPR"
                    )
                empirical = operating_point.get("empirical_fpr")
                if not isinstance(empirical, dict):
                    raise ValueError(
                        "classifier operating points require empirical_fpr"
                    )
                empirical_count = ExactCount(
                    sample_num=int(empirical["sample_num"]),
                    positive_num=int(empirical["positive_num"]),
                )
                empirical_rate = (
                    empirical_count.positive_num / empirical_count.sample_num
                    if empirical_count.sample_num
                    else -1.0
                )
                if not math.isclose(
                    float(empirical["rate"]),
                    empirical_rate,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                ):
                    raise ValueError(
                        "empirical FPR disagrees with its exact counts"
                    )
        else:
            target_fpr = float(threshold)
            if not 0 < target_fpr < 1:
                raise ValueError("target FPR must be between zero and one")
        value = (
            count.positive_num / count.sample_num
            if count.sample_num
            else 0.0
        )
        if rates is not None:
            stored = rates.get(threshold)
            if stored is None or not math.isclose(
                float(stored),
                value,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"{metric} rate at {threshold} disagrees with exact "
                    f"count ratio: stored={stored!r}, expected={value!r}"
                )
        result.append(
            _RawMetricFact(
                metric=metric,
                value=value,
                token_num=token_num,
                target_fpr=target_fpr,
                detection_operating_point=operating_point,
                count=count,
            )
        )
    return result


def _claims(
    semantic: JsonObject,
    *,
    producer: bool,
    transformation: bool = False,
    task: bool = False,
) -> dict[str, object]:
    prompt_population_value = semantic.get("prompt_population")
    prompt_population = (
        prompt_population_value
        if isinstance(prompt_population_value, dict)
        else None
    )
    population = _population(
        prompt_population.get("dataset")
        if prompt_population is not None
        else None
    )
    return {
        "population": population,
        "generation": (
            _generation(semantic, population, prompt_population)
            if producer
            else None
        ),
        "scheme": _scheme(semantic.get("watermark")),
        "transformation": (
            _transformation(semantic.get("transformation"))
            if transformation
            else None
        ),
        "task": semantic.get("task") if task else None,
    }


def _base(
    ref: ArtifactRef,
    summary: JsonObject,
    *,
    producer: bool,
    transformation: bool = False,
    task: bool = False,
    inherited_model: bool = False,
    detector_scheme: bool = False,
    evaluation_model_field: str | None = None,
    facts: tuple[_RawMetricFact, ...] = (),
) -> _ParsedArtifact:
    semantic = ref.manifest.get("semantic_settings", {})
    if not isinstance(semantic, dict):
        raise ValueError("semantic_settings must be a mapping")
    evaluation_model = (
        _model(semantic.get(evaluation_model_field))
        if evaluation_model_field
        else None
    )
    detector_value = semantic.get("detector_watermark")
    if not isinstance(detector_value, dict):
        detector_value = semantic.get("watermark")
    return _ParsedArtifact(
        claims=_claims(
            semantic,
            producer=producer,
            transformation=transformation,
            task=task,
        ),
        evaluation_model=evaluation_model,
        detector_scheme=(
            _scheme(detector_value) if detector_scheme else None
        ),
        inherited_model_claim=(
            _model(semantic.get("model")) if inherited_model else None
        ),
        raw_facts=facts,
        summary_sample_num=_summary_count(summary),
    )


def _parse_generation(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    facts = tuple(
        _RawMetricFact(metric=metric, value=float(summary[metric]))
        for metric in (
            "source_sample_num",
            "repetitions",
            "generated_token_num",
            "mean_generated_token_num",
        )
        if summary.get(metric) is not None
    )
    return _base(ref, summary, producer=True, facts=facts)


def _parse_forgery(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    metrics = (
        "theoretical_green_probability",
        "theoretical_queries_per_scored_token",
        "observed_minus_theoretical_green_probability",
        "observed_to_theoretical_query_ratio",
        "oracle_query_count",
        "color_check_count",
        "cache_hit_count",
        "oracle_round_count",
        "generated_token_num",
        "scored_position_num",
        "scored_token_num",
        "duplicate_selected_pair_num",
        "observed_queries_per_generated_token",
        "observed_queries_per_scored_token",
        "selected_green_ratio",
        "fallback_ratio",
        "cache_hit_ratio",
        "duplicate_selected_pair_ratio",
        "query_overhead_vs_honest_audit",
        "detector_green_ratio",
        "median_p_value",
        "communication_bytes_per_oracle_query",
        "tokenization_preserved_ratio",
        "mean_forgery_seconds",
        "total_oracle_time_seconds",
        "oracle_time_fraction",
        "total_communication_bytes",
        "total_elapsed_seconds",
    )
    facts = [
        _RawMetricFact(metric=name, value=float(summary[name]))
        for name in metrics
        if summary.get(name) is not None
    ]
    sample_num = int(summary["sample_num"])
    for metric, field in (
        ("generated_token_length", "generated_token_length"),
        ("oracle_queries_per_sample", "oracle_queries_per_sample"),
        (
            "queries_per_scored_token_per_sample",
            "queries_per_scored_token_per_sample",
        ),
        ("local_model_perplexity", "local_model_perplexity"),
        ("mean_log_probability_gap", "mean_log_probability_gap"),
        ("p_value", "p_value"),
    ):
        if summary.get(field) is None:
            continue
        distribution = _distribution(
            summary[field],
            default_count=sample_num,
        )
        if distribution is not None:
            facts.append(
                _RawMetricFact(
                    metric=metric,
                    value=distribution.mean,
                    count=ExactCount(distribution.count),
                    distribution=distribution,
                )
            )
    return _base(ref, summary, producer=True, facts=tuple(facts))


def _parse_robustness(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    facts = []
    if summary.get("mean_character_ratio") is not None:
        facts.append(
            _RawMetricFact(
                metric="mean_character_ratio",
                value=float(summary["mean_character_ratio"]),
            )
        )
    remote = summary.get("openai")
    if remote is not None:
        if not isinstance(remote, dict):
            raise ValueError("openai robustness summary must be a mapping")
        for metric, value in (
            ("openai_request_num", remote.get("request_num")),
            (
                "openai_mean_latency_seconds",
                remote.get("mean_latency_seconds"),
            ),
        ):
            if value is not None:
                facts.append(
                    _RawMetricFact(metric=metric, value=float(value))
                )
        usage = remote.get("usage", {})
        if not isinstance(usage, dict):
            raise ValueError("openai usage summary must be a mapping")
        facts.extend(
            _RawMetricFact(
                metric=f"openai_{field}",
                value=float(value),
            )
            for field, value in usage.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
        )
    return _base(
        ref,
        summary,
        producer=False,
        transformation=True,
        inherited_model=True,
        facts=tuple(facts),
    )


def _parse_detection(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    operating_points = summary.get("detection_operating_points")
    facts = _facts_from_counts(
        summary.get("detection_counts"),
        metric="detection_rate",
        rates=summary.get("detection_rate"),
        operating_points=operating_points,
    )
    thresholds = summary.get("green_count_thresholds", {})
    if not isinstance(thresholds, dict):
        raise ValueError("green_count_thresholds must be a mapping")
    for level_text, value in thresholds.items():
        level = float(level_text)
        facts.append(
            _RawMetricFact(
                metric="green_count_threshold",
                value=int(value),
                target_fpr=level,
                detection_operating_point=(
                    operating_points.get(level_text)
                    if isinstance(operating_points, dict)
                    else None
                ),
            )
        )
    sample_num = int(summary["sample_num"])
    for metric, field in (
        ("total_token_num", "total_token_num"),
        ("mean_token_num", "mean_token_num"),
        ("effective_token_num", "effective_token_num"),
        ("green_token_num", "green_token_num"),
        ("green_ratio", "green_ratio"),
        ("total_oracle_query_count", "oracle_query_count"),
        (
            "query_overhead_vs_honest_audit",
            "query_overhead_vs_honest_audit",
        ),
    ):
        if summary.get(field) is not None:
            facts.append(
                _RawMetricFact(metric=metric, value=float(summary[field]))
            )
    milestones = summary.get("milestone_detection_rate", [])
    if not isinstance(milestones, list):
        raise ValueError("milestone_detection_rate must be a list")
    seen_milestones: set[int] = set()
    for milestone in milestones:
        if not isinstance(milestone, dict):
            raise ValueError("detection milestone must be a mapping")
        token_num = int(milestone["token_num"])
        if token_num in seen_milestones:
            raise ValueError(
                f"duplicate detection milestone token_num {token_num}"
            )
        seen_milestones.add(token_num)
        eligible = int(milestone["eligible_sample_num"])
        if eligible < 0 or eligible > sample_num:
            raise ValueError(
                "milestone eligible_sample_num must be between zero and "
                "summary sample_num"
            )
        milestone_facts = _facts_from_counts(
            milestone.get("detection_counts"),
            metric="detection_rate",
            token_num=token_num,
            rates=milestone.get("detection_rate"),
            operating_points=operating_points,
        )
        if any(
            fact.count is not None
            and fact.count.sample_num != eligible
            for fact in milestone_facts
        ):
            raise ValueError(
                "milestone detection count population disagrees with "
                "eligible_sample_num"
            )
        facts.extend(
            milestone_facts
        )
    scalar_distributions = {
        "p_value": "p_value_distribution",
        "negative_log10_p_value": "negative_log10_p_value_distribution",
        "classifier_confidence": "classifier_confidence_distribution",
        "green_token_count": "green_token_count_distribution",
        "effective_token_count": "effective_token_count_distribution",
        "oracle_query_count": "oracle_query_count_distribution",
    }
    for metric, field in scalar_distributions.items():
        if field in summary:
            distribution = _distribution(summary[field])
            if distribution is not None:
                facts.append(
                    _RawMetricFact(
                        metric=metric,
                        value=distribution.mean,
                        distribution=distribution,
                        count=ExactCount(distribution.count),
                    )
                )
    curve = summary.get("adaptive_forgery_curve", [])
    if curve is not None and not isinstance(curve, list):
        raise ValueError("adaptive_forgery_curve must be a list")
    seen_curve_tokens: set[int] = set()
    for point in curve or []:
        token_num = int(point["token_num"])
        if token_num in seen_curve_tokens:
            raise ValueError(
                f"duplicate forgery curve token_num {token_num}"
            )
        seen_curve_tokens.add(token_num)
        sample_num = int(point["eligible_sample_num"])
        if sample_num < 0 or sample_num > int(summary["sample_num"]):
            raise ValueError(
                "forgery curve eligible_sample_num must be between zero "
                "and summary sample_num"
            )
        for field in (
            "mean_oracle_query_count",
            "median_oracle_query_count",
            "mean_queries_per_token",
            "mean_selected_green_token_count",
            "mean_selected_green_ratio",
            "median_p_value",
        ):
            facts.append(
                _RawMetricFact(
                    metric=field,
                    value=float(point[field]),
                    token_num=token_num,
                    count=ExactCount(sample_num),
                )
            )
        attack_facts = _facts_from_counts(
            point.get("attack_success_counts"),
            metric="attack_success_rate",
            token_num=token_num,
            rates=point.get("attack_success_rate"),
        )
        if any(
            fact.count is not None
            and fact.count.sample_num != sample_num
            for fact in attack_facts
        ):
            raise ValueError(
                "forgery attack count population disagrees with "
                "eligible_sample_num"
            )
        facts.extend(attack_facts)
    return _base(
        ref,
        summary,
        producer=False,
        inherited_model=True,
        detector_scheme=True,
        evaluation_model_field="model",
        facts=tuple(facts),
    )


def _parse_perplexity(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    sample_num = int(summary["sample_num"])
    distribution = _distribution(
        summary.get("sample_conditional_perplexity"),
        default_count=sample_num,
    )
    facts = [
        _RawMetricFact(
            metric="conditional_perplexity",
            value=float(summary["conditional_perplexity"]),
            count=ExactCount(sample_num),
            distribution=distribution,
        ),
    ]
    for metric in (
        "mean_negative_log_likelihood",
        "perplexity_target_token_num",
        "mean_sample_conditional_perplexity",
        "mean_distinct_1",
        "mean_distinct_2",
    ):
        if summary.get(metric) is not None:
            facts.append(
                _RawMetricFact(metric=metric, value=float(summary[metric]))
            )
    for metric in (
        "evaluation_token_length",
        "characters_per_token",
        "distinct_1",
        "distinct_2",
        "distinct_4",
    ):
        if summary.get(metric) is None:
            continue
        metric_distribution = _distribution(
            summary[metric],
            default_count=sample_num,
        )
        if metric_distribution is not None:
            facts.append(
                _RawMetricFact(
                    metric=metric,
                    value=metric_distribution.mean,
                    count=ExactCount(metric_distribution.count),
                    distribution=metric_distribution,
                )
            )
    correlations = summary.get("quality_cost_correlations", {})
    if not isinstance(correlations, dict):
        raise ValueError("quality_cost_correlations must be a mapping")
    facts.extend(
        _RawMetricFact(metric=metric, value=float(value))
        for metric, value in correlations.items()
        if value is not None
    )
    return _base(
        ref,
        summary,
        producer=False,
        evaluation_model_field="model",
        facts=tuple(facts),
    )


def _parse_text_evaluation(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    facts = []
    similarity = summary.get("similarity")
    if similarity is not None:
        distribution = _distribution(similarity)
        if distribution is None:
            raise ValueError("similarity distribution requires count")
        facts.append(
            _RawMetricFact(
                metric="cosine_similarity",
                value=distribution.mean,
                count=ExactCount(distribution.count),
                distribution=distribution,
            )
        )
    diversity = summary.get("diversity")
    if diversity is not None:
        if not isinstance(diversity, dict) or not isinstance(
            diversity.get("aggregate"), dict
        ):
            raise ValueError("diversity aggregate must be a mapping")
        group_num = int(diversity["group_num"])
        for metric in (
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "self_bleu_4",
            "vendi_score",
        ):
            facts.append(
                _RawMetricFact(
                    metric=metric,
                    value=float(diversity["aggregate"][metric]),
                    count=ExactCount(group_num),
                )
            )
    return _base(
        ref,
        summary,
        producer=False,
        evaluation_model_field="embedding_model",
        facts=tuple(facts),
    )


def _parse_downstream(
    ref: ArtifactRef,
    summary: JsonObject,
    issues: list[InterpretationIssue],
    root: str,
    path: tuple[str, ...],
) -> _ParsedArtifact:
    del issues, root, path
    sample_num = int(summary["sample_num"])
    task = summary["task"]
    if task == "gsm8k":
        metric = "accuracy"
        positive = int(summary["correct_num"])
    elif task == "humaneval":
        metric = "pass_at_1"
        positive = int(summary["passed_num"])
    else:
        raise ValueError(f"unknown downstream task {task!r}")
    count = ExactCount(sample_num=sample_num, positive_num=positive)
    stored_rate = summary.get(metric)
    expected_rate = positive / sample_num if sample_num else 0.0
    if stored_rate is not None and not math.isclose(
        float(stored_rate),
        expected_rate,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"{metric} disagrees with exact count ratio: "
            f"stored={stored_rate!r}, expected={expected_rate!r}"
        )
    fact = _RawMetricFact(
        metric=metric,
        value=expected_rate,
        count=count,
    )
    return _base(
        ref,
        summary,
        producer=True,
        task=True,
        evaluation_model_field="model",
        facts=(fact,),
    )


def _jsonlines(path: Path) -> Iterator[JsonObject]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield value


def _selected(metric: str, selected: frozenset[str] | None) -> bool:
    return selected is None or metric in selected


def _sample_fact(
    *,
    metric: str,
    value: object,
    sample_id: str,
    dimensions: ScientificDimensions,
    ref: ArtifactRef,
) -> MetricFact:
    number = float(value)
    if metric in {
        "p_value",
        "classifier_confidence",
        "selected_green_ratio",
        "fallback_ratio",
        "cache_hit_ratio",
        "duplicate_selected_pair_ratio",
        "detector_green_ratio",
        "tokenization_preserved_ratio",
        "accuracy",
        "pass_at_1",
    } and not 0 <= number <= 1:
        raise ValueError(
            f"Sample metric {metric!r} must be between zero and one"
        )
    if (
        "query" in metric
        or "token_num" in metric
        or metric.endswith("_count")
        or metric.endswith("_seconds")
    ) and number < 0:
        raise ValueError(f"Sample metric {metric!r} must be non-negative")
    return MetricFact(
        metric=metric,
        scope=MetricScope.SAMPLE,
        value=number,
        dimensions=dimensions,
        source_artifact=ref.identity,
        sample_identity=sample_id,
    )


def _stream_none(
    ref: ArtifactRef,
    dimensions: ScientificDimensions,
    selected: frozenset[str] | None,
) -> Iterator[MetricFact]:
    del ref, dimensions, selected
    return
    yield


def _stream_records(
    ref: ArtifactRef,
    dimensions: ScientificDimensions,
    selected: frozenset[str] | None,
) -> Iterator[MetricFact]:
    seen: set[str] = set()
    count = 0
    for record in _jsonlines(ref.path / "records.jsonl"):
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("Sample record lacks a stable sample_id")
        if sample_id in seen:
            raise ValueError(f"duplicate Sample identity {sample_id!r}")
        seen.add(sample_id)
        count += 1
        kind = str(ref.manifest["kind"])
        values: list[tuple[str, object]] = []
        if kind == "adaptive-forgery":
            values.extend(
                (metric, value)
                for metric, value in record.get("sample_metrics", {}).items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
            )
            detection = record.get("detection", {})
            if isinstance(detection, dict):
                for metric, field in (
                    ("p_value", "p_value"),
                    ("classifier_confidence", "confidence"),
                    ("green_token_count", "green_token_num"),
                    ("effective_token_count", "effective_token_num"),
                ):
                    if detection.get(field) is not None:
                        values.append((metric, detection[field]))
        elif kind == "robustness":
            robustness = record.get("robustness", {})
            if "character_ratio" in robustness:
                values.append(
                    ("character_ratio", robustness["character_ratio"])
                )
        elif kind == "detection":
            detection = record.get("detection", {})
            for metric, field in (
                ("p_value", "p_value"),
                ("classifier_confidence", "confidence"),
                ("green_token_count", "green_token_num"),
                ("effective_token_count", "effective_token_num"),
            ):
                if detection.get(field) is not None:
                    values.append((metric, detection[field]))
        elif kind == "perplexity":
            quality = record.get("quality", {})
            if "conditional_perplexity" in quality:
                values.append(
                    (
                        "conditional_perplexity",
                        quality["conditional_perplexity"],
                    )
                )
        elif kind == "text-evaluation":
            evaluation = record.get("text_evaluation", {})
            if "cosine_similarity" in evaluation:
                values.append(
                    ("cosine_similarity", evaluation["cosine_similarity"])
                )
        elif kind == "downstream":
            if "is_correct" in record:
                values.append(("accuracy", int(bool(record["is_correct"]))))
            if "passed" in record:
                values.append(("pass_at_1", int(bool(record["passed"]))))
        for metric, value in values:
            if _selected(metric, selected):
                yield _sample_fact(
                    metric=metric,
                    value=value,
                    sample_id=sample_id,
                    dimensions=dimensions,
                    ref=ref,
                )
    expected = int(ref.manifest["record_count"])
    if count != expected:
        raise ValueError(
            f"records.jsonl has {count} records, manifest declares {expected}"
        )


schema_registry: dict[tuple[str, str], _SchemaAdapter] = {
    ("generation", "generated-text-v3"): _SchemaAdapter(
        "generation",
        "generated-text-v3",
        _parse_generation,
        _stream_none,
    ),
    ("adaptive-forgery", "adaptive-forgery-v2"): _SchemaAdapter(
        "adaptive-forgery",
        "adaptive-forgery-v2",
        _parse_forgery,
        _stream_records,
    ),
    ("adaptive-forgery", "adaptive-forgery-v3"): _SchemaAdapter(
        "adaptive-forgery",
        "adaptive-forgery-v3",
        _parse_forgery,
        _stream_records,
    ),
    ("robustness", "robustness-text-v2"): _SchemaAdapter(
        "robustness",
        "robustness-text-v2",
        _parse_robustness,
        _stream_records,
    ),
    ("detection", "watermark-detection-v4"): _SchemaAdapter(
        "detection",
        "watermark-detection-v4",
        _parse_detection,
        _stream_records,
    ),
    ("perplexity", "conditional-perplexity-v2"): _SchemaAdapter(
        "perplexity",
        "conditional-perplexity-v2",
        _parse_perplexity,
        _stream_records,
    ),
    ("text-evaluation", "text-evaluation-v1"): _SchemaAdapter(
        "text-evaluation",
        "text-evaluation-v1",
        _parse_text_evaluation,
        _stream_records,
    ),
    ("downstream", "gsm8k-evaluation-v2"): _SchemaAdapter(
        "downstream",
        "gsm8k-evaluation-v2",
        _parse_downstream,
        _stream_records,
    ),
    ("downstream", "humaneval-evaluation-v2"): _SchemaAdapter(
        "downstream",
        "humaneval-evaluation-v2",
        _parse_downstream,
        _stream_records,
    ),
}
