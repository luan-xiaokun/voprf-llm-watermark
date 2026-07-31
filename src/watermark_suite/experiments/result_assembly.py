from __future__ import annotations

import json
import math
from statistics import NormalDist
from typing import Callable

from .artifact_interpretation import (
    ARTIFACT_INTERPRETATION_REVISION,
    ArtifactInterpretation,
    InterpretationSet,
    MetricFact,
)
from .errors import PlanValidationError
from .identity import identity_for
from .models import JsonObject


REPORT_RECIPES = frozenset(
    {
        "tpr-vs-token-length",
        "tpr-vs-ppl",
        "downstream-performance",
        "robustness",
        "adaptive-forgery",
        "diversity",
    }
)


def _wilson(
    positive: int,
    total: int,
    confidence_level: float,
) -> JsonObject | None:
    if total <= 0:
        return None
    z_value = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    observed = positive / total
    denominator = 1.0 + z_value * z_value / total
    center = (
        observed + z_value * z_value / (2.0 * total)
    ) / denominator
    margin = (
        z_value
        * math.sqrt(
            observed * (1.0 - observed) / total
            + z_value * z_value / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "method": "wilson",
        "confidence_level": confidence_level,
        "lower": max(0.0, center - margin),
        "upper": min(1.0, center + margin),
    }


class ResultAssembler:
    """Join interpreted scientific facts into recipe-specific report rows."""

    revision = f"result-assembly-v2+{ARTIFACT_INTERPRETATION_REVISION}"

    def __init__(
        self,
        interpretations: InterpretationSet,
        *,
        confidence_level: float,
        threshold_policy: JsonObject,
    ) -> None:
        self.interpretations = interpretations
        self.confidence_level = confidence_level
        self.threshold_policy = threshold_policy

    def _threshold(self, artifact: ArtifactInterpretation) -> float:
        method = artifact.dimensions.scheme
        value = self.threshold_policy.get(
            method, self.threshold_policy.get("default")
        )
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 < float(value) < 1
        ):
            raise PlanValidationError(
                f"threshold_policy has no valid threshold for scheme {method!r}"
            )
        return float(value)

    @staticmethod
    def _facts(
        artifact: ArtifactInterpretation,
        metric: str,
        *,
        token_axis: bool | None = None,
        target_fpr: float | None = None,
    ) -> tuple[MetricFact, ...]:
        result = []
        for fact in artifact.facts(metric):
            token_num = fact.dimensions.token_num
            if token_axis is True and token_num is None:
                continue
            if token_axis is False and token_num is not None:
                continue
            if target_fpr is not None and (
                fact.dimensions.target_fpr is None
                or not math.isclose(
                    fact.dimensions.target_fpr,
                    target_fpr,
                    rel_tol=0.0,
                    abs_tol=1e-15,
                )
            ):
                continue
            result.append(fact)
        return tuple(result)

    @staticmethod
    def _one_fact(
        artifact: ArtifactInterpretation,
        metric: str,
        *,
        token_axis: bool | None = None,
        target_fpr: float | None = None,
    ) -> MetricFact:
        facts = ResultAssembler._facts(
            artifact,
            metric,
            token_axis=token_axis,
            target_fpr=target_fpr,
        )
        if len(facts) != 1:
            raise PlanValidationError(
                f"Artifact {artifact.identity.value} must provide exactly one "
                f"{metric!r} fact for token_axis={token_axis!r}, "
                f"target_fpr={target_fpr!r}; found {len(facts)}"
            )
        return facts[0]

    def _row(
        self,
        *,
        recipe: str,
        metric: str,
        fact: MetricFact,
        sources: tuple[ArtifactInterpretation, ...],
    ) -> JsonObject:
        source_ids = [source.identity.value for source in sources]
        dimensions = fact.dimensions.to_dict()
        identity_document = {
            "recipe": recipe,
            "metric": metric,
            "dimensions": dimensions,
            "source_artifacts": source_ids,
        }
        count = fact.count
        population = count.to_dict() if count is not None else {}
        positive = count.positive_num if count is not None else None
        total = count.sample_num if count is not None else None
        row: JsonObject = {
            "sample_id": identity_for(identity_document, prefix="metric"),
            "recipe": recipe,
            "metric": metric,
            "value": fact.value,
            "dimensions": dimensions,
            "population": population,
            "uncertainty": (
                _wilson(positive, total, self.confidence_level)
                if positive is not None and total is not None
                else None
            ),
            "source_artifacts": source_ids,
        }
        if fact.distribution is not None:
            row["distribution"] = fact.distribution.to_dict()
        return row

    @staticmethod
    def _require_kinds(
        artifacts: tuple[ArtifactInterpretation, ...],
        allowed: set[str],
        *,
        recipe: str,
    ) -> None:
        unexpected = [
            f"{artifact.identity.value}:{artifact.kind}"
            for artifact in artifacts
            if artifact.kind not in allowed
        ]
        if unexpected:
            raise PlanValidationError(
                f"{recipe} received unsupported source kinds: "
                + ", ".join(unexpected)
            )

    @staticmethod
    def _require_comparable_population(
        artifacts: tuple[ArtifactInterpretation, ...],
        *,
        recipe: str,
    ) -> None:
        populations = {
            artifact.dimensions.population_identity
            for artifact in artifacts
            if artifact.dimensions.population_identity is not None
        }
        if len(populations) != 1 or any(
            artifact.dimensions.population_identity is None
            for artifact in artifacts
        ):
            details = {
                artifact.identity.value:
                    artifact.dimensions.population_identity
                for artifact in artifacts
            }
            raise PlanValidationError(
                f"{recipe} requires one sample population, found {details}"
            )

    @staticmethod
    def _require_comparable_generation(
        artifacts: tuple[ArtifactInterpretation, ...],
        *,
        recipe: str,
        group_by_task: bool = False,
    ) -> None:
        groups: dict[str, list[ArtifactInterpretation]] = {}
        for artifact in artifacts:
            task = artifact.dimensions.task if group_by_task else None
            group = str(task) if task is not None else "all"
            groups.setdefault(group, []).append(artifact)
        for group, members in groups.items():
            contexts = {
                json.dumps(
                    member.dimensions.generation,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for member in members
            }
            if len(contexts) != 1 or any(
                member.dimensions.generation is None for member in members
            ):
                details = {
                    member.identity.value: member.dimensions.generation
                    for member in members
                }
                suffix = f" for task {group!r}" if group_by_task else ""
                raise PlanValidationError(
                    f"{recipe} requires one generation context{suffix}, "
                    f"found {details}"
                )

    def _tpr_token(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "tpr-vs-token-length"
        self._require_kinds(artifacts, {"detection"}, recipe=recipe)
        self._require_comparable_population(artifacts, recipe=recipe)
        self._require_comparable_generation(artifacts, recipe=recipe)
        rows = []
        for artifact in artifacts:
            threshold = self._threshold(artifact)
            facts = self._facts(
                artifact,
                "detection_rate",
                token_axis=True,
                target_fpr=threshold,
            )
            if not facts:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} has no token-level "
                    f"detection facts at target FPR {threshold}"
                )
            rows.extend(
                self._row(
                    recipe=recipe,
                    metric="true_positive_rate",
                    fact=fact,
                    sources=(artifact,),
                )
                for fact in facts
            )
        return rows

    def _tpr_ppl(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "tpr-vs-ppl"
        self._require_kinds(
            artifacts, {"detection", "perplexity"}, recipe=recipe
        )
        self._require_comparable_population(artifacts, recipe=recipe)
        self._require_comparable_generation(artifacts, recipe=recipe)
        detections = {
            artifact.identity.value: artifact
            for artifact in artifacts
            if artifact.kind == "detection"
        }
        perplexities = tuple(
            artifact
            for artifact in artifacts
            if artifact.kind == "perplexity"
        )
        baseline = [
            artifact
            for artifact in perplexities
            if artifact.dimensions.scheme == "none"
        ]
        if len(baseline) != 1:
            raise PlanValidationError(
                f"{recipe} requires exactly one unwatermarked PPL baseline"
            )

        rows = []
        matched: set[str] = set()
        for ppl in perplexities:
            ppl_fact = self._one_fact(
                ppl, "conditional_perplexity", token_axis=False
            )
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="conditional_perplexity",
                    fact=ppl_fact,
                    sources=(ppl,),
                )
            )
            if ppl.dimensions.scheme == "none":
                continue
            source_matches = [
                detections[source.value]
                for source in ppl.direct_source_identities
                if source.value in detections
            ]
            if len(source_matches) != 1:
                raise PlanValidationError(
                    f"PPL Artifact {ppl.identity.value} must have exactly one "
                    "collected detection parent"
                )
            detection = source_matches[0]
            if detection.identity.value in matched:
                raise PlanValidationError(
                    f"Detection Artifact {detection.identity.value} has more "
                    "than one collected PPL descendant"
                )
            matched.add(detection.identity.value)
            if (
                detection.dimensions.scheme_identity
                != ppl.dimensions.scheme_identity
            ):
                raise PlanValidationError(
                    f"PPL Artifact {ppl.identity.value} and detection "
                    f"{detection.identity.value} disagree on scheme"
                )
            fact = self._one_fact(
                detection,
                "detection_rate",
                token_axis=False,
                target_fpr=self._threshold(detection),
            )
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="true_positive_rate",
                    fact=fact,
                    sources=(detection, ppl),
                )
            )
        if matched != set(detections):
            missing = sorted(set(detections) - matched)
            raise PlanValidationError(
                f"{recipe} did not pair detection Artifacts {missing}"
            )
        return rows

    def _downstream(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "downstream-performance"
        self._require_kinds(artifacts, {"downstream"}, recipe=recipe)
        self._require_comparable_generation(
            artifacts,
            recipe=recipe,
            group_by_task=True,
        )
        rows = []
        for artifact in artifacts:
            metric = {
                "gsm8k": "accuracy",
                "humaneval": "pass_at_1",
            }.get(artifact.dimensions.task)
            if metric is None:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} has unknown task "
                    f"{artifact.dimensions.task!r}"
                )
            rows.append(
                self._row(
                    recipe=recipe,
                    metric=metric,
                    fact=self._one_fact(
                        artifact, metric, token_axis=False
                    ),
                    sources=(artifact,),
                )
            )
        return rows

    def _robustness(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "robustness"
        self._require_kinds(
            artifacts, {"detection", "text-evaluation"}, recipe=recipe
        )
        self._require_comparable_population(artifacts, recipe=recipe)
        self._require_comparable_generation(artifacts, recipe=recipe)

        evaluations_by_source: dict[str, ArtifactInterpretation] = {}
        for artifact in artifacts:
            if artifact.kind != "text-evaluation":
                continue
            sources = artifact.direct_source_identities
            if len(sources) != 1:
                raise PlanValidationError(
                    f"Similarity Artifact {artifact.identity.value} must have "
                    "exactly one transformed-text parent"
                )
            source = sources[0].value
            if source in evaluations_by_source:
                raise PlanValidationError(
                    f"Transformed Artifact {source} has more than one "
                    "similarity evaluation"
                )
            evaluations_by_source[source] = artifact

        rows = []
        transformed_detections: dict[str, ArtifactInterpretation] = {}
        for artifact in artifacts:
            if artifact.kind != "detection":
                continue
            if artifact.dimensions.transformation is not None:
                sources = artifact.direct_source_identities
                if len(sources) != 1:
                    raise PlanValidationError(
                        f"Detection Artifact {artifact.identity.value} must "
                        "have exactly one transformed-text parent"
                    )
                source = sources[0].value
                if source in transformed_detections:
                    raise PlanValidationError(
                        f"Transformed Artifact {source} has more than one "
                        "detection evaluation"
                    )
                transformed_detections[source] = artifact
            threshold = self._threshold(artifact)
            facts = self._facts(
                artifact,
                "detection_rate",
                token_axis=True,
                target_fpr=threshold,
            )
            if not facts:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} has no token-level "
                    f"detection facts at target FPR {threshold}"
                )
            rows.extend(
                self._row(
                    recipe=recipe,
                    metric="detection_rate",
                    fact=fact,
                    sources=(artifact,),
                )
                for fact in facts
            )

        detection_sources = set(transformed_detections)
        similarity_sources = set(evaluations_by_source)
        if detection_sources != similarity_sources:
            raise PlanValidationError(
                f"{recipe} requires one detection and one similarity "
                "evaluation per transformed Artifact; missing similarity "
                f"for {sorted(detection_sources - similarity_sources)}, "
                "missing detection for "
                f"{sorted(similarity_sources - detection_sources)}"
            )
        for source, evaluation in evaluations_by_source.items():
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="cosine_similarity",
                    fact=self._one_fact(
                        evaluation,
                        "cosine_similarity",
                        token_axis=False,
                    ),
                    sources=(transformed_detections[source], evaluation),
                )
            )
        return rows

    def _adaptive_forgery(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "adaptive-forgery"
        self._require_kinds(
            artifacts, {"detection", "perplexity"}, recipe=recipe
        )
        detections = tuple(
            artifact
            for artifact in artifacts
            if artifact.kind == "detection"
        )
        perplexities = tuple(
            artifact
            for artifact in artifacts
            if artifact.kind == "perplexity"
        )
        if len(detections) != 1:
            raise PlanValidationError(
                f"{recipe} requires exactly one detection Artifact"
            )
        if len(perplexities) != 1:
            raise PlanValidationError(
                f"{recipe} requires exactly one PPL Artifact"
            )
        self._require_comparable_population(artifacts, recipe=recipe)
        self._require_comparable_generation(artifacts, recipe=recipe)
        detection = detections[0]
        ppl = perplexities[0]

        curve_metrics = {
            "mean_oracle_query_count",
            "median_oracle_query_count",
            "mean_queries_per_token",
            "mean_selected_green_token_count",
            "mean_selected_green_ratio",
            "median_p_value",
            "attack_success_rate",
        }
        curve_facts = tuple(
            fact
            for fact in detection.aggregate_facts
            if fact.metric in curve_metrics
            and fact.dimensions.token_num is not None
        )
        if not curve_facts:
            raise PlanValidationError(
                f"Artifact {detection.identity.value} has no forgery curve"
            )
        rows = [
            self._row(
                recipe=recipe,
                metric=fact.metric,
                fact=fact,
                sources=(detection,),
            )
            for fact in curve_facts
        ]
        overall_metrics = {
            "p_value",
            "negative_log10_p_value",
            "green_token_count",
            "effective_token_count",
            "oracle_query_count",
            "total_oracle_query_count",
            "green_ratio",
            "query_overhead_vs_honest_audit",
        }
        rows.extend(
            self._row(
                recipe=recipe,
                metric=fact.metric,
                fact=fact,
                sources=(detection,),
            )
            for fact in detection.aggregate_facts
            if fact.metric in overall_metrics
            and fact.dimensions.token_num is None
        )
        if detection.identity not in ppl.direct_source_identities:
            raise PlanValidationError(
                f"PPL Artifact {ppl.identity.value} is not a direct "
                f"descendant of detection Artifact {detection.identity.value}"
            )
        if detection.dimensions.scheme_identity != ppl.dimensions.scheme_identity:
            raise PlanValidationError(
                f"PPL Artifact {ppl.identity.value} and detection "
                f"{detection.identity.value} disagree on scheme"
            )
        rows.append(
            self._row(
                recipe=recipe,
                metric="conditional_perplexity",
                fact=self._one_fact(
                    ppl, "conditional_perplexity", token_axis=False
                ),
                sources=(detection, ppl),
            )
        )
        return rows

    def _diversity(
        self,
        artifacts: tuple[ArtifactInterpretation, ...],
    ) -> list[JsonObject]:
        recipe = "diversity"
        self._require_kinds(artifacts, {"text-evaluation"}, recipe=recipe)
        self._require_comparable_population(artifacts, recipe=recipe)
        self._require_comparable_generation(artifacts, recipe=recipe)
        rows = []
        metrics = (
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "self_bleu_4",
            "vendi_score",
        )
        for artifact in artifacts:
            for metric in metrics:
                rows.append(
                    self._row(
                        recipe=recipe,
                        metric=metric,
                        fact=self._one_fact(
                            artifact, metric, token_axis=False
                        ),
                        sources=(artifact,),
                    )
                )
        return rows

    def assemble(
        self,
        *,
        recipe: str,
    ) -> tuple[list[JsonObject], JsonObject]:
        if recipe not in REPORT_RECIPES:
            raise PlanValidationError(f"unknown result recipe {recipe!r}")
        artifacts = self.interpretations.root_artifacts
        method: Callable[
            [tuple[ArtifactInterpretation, ...]],
            list[JsonObject],
        ] = {
            "tpr-vs-token-length": self._tpr_token,
            "tpr-vs-ppl": self._tpr_ppl,
            "downstream-performance": self._downstream,
            "robustness": self._robustness,
            "adaptive-forgery": self._adaptive_forgery,
            "diversity": self._diversity,
        }[recipe]
        rows = method(artifacts)
        rows.sort(
            key=lambda row: (
                json.dumps(
                    row["dimensions"],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                row["metric"],
                row["sample_id"],
            )
        )
        consumed = {
            source
            for row in rows
            for source in row["source_artifacts"]
        }
        expected = {
            artifact.identity.value for artifact in artifacts
        }
        if consumed != expected:
            raise PlanValidationError(
                f"{recipe} did not account for source Artifacts "
                f"{sorted(expected - consumed)}"
            )
        schema_counts: dict[str, int] = {}
        for artifact in artifacts:
            schema_counts[artifact.schema_revision] = (
                schema_counts.get(artifact.schema_revision, 0) + 1
            )
        summary = {
            "recipe": recipe,
            "recipe_revision": self.revision,
            "artifact_interpretation_revision":
                ARTIFACT_INTERPRETATION_REVISION,
            "source_artifacts": [
                artifact.identity.value for artifact in artifacts
            ],
            "source_schema_counts": schema_counts,
            "metric_row_num": len(rows),
            "metrics": sorted({row["metric"] for row in rows}),
            "threshold_policy": self.threshold_policy,
            "confidence_level": self.confidence_level,
            "compatibility": {"status": "passed"},
        }
        return rows, summary
