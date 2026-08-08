from __future__ import annotations

import json
import math
from dataclasses import replace
from statistics import NormalDist
from typing import Callable

from .artifact_interpretation import (
    ARTIFACT_INTERPRETATION_REVISION,
    ArtifactInterpretation,
    ExactCount,
    InterpretationSet,
    MetricFact,
    MetricScope,
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

    revision = f"result-assembly-v4+{ARTIFACT_INTERPRETATION_REVISION}"

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
        method = (
            artifact.dimensions.detector_scheme
            or artifact.dimensions.scheme
        )
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
    def _roc_auc(
        positive_scores: list[float],
        negative_scores: list[float],
    ) -> float:
        if not positive_scores or not negative_scores:
            raise PlanValidationError(
                "AUC requires non-empty positive and negative samples"
            )
        ranked = sorted(
            [(value, True) for value in positive_scores]
            + [(value, False) for value in negative_scores]
        )
        rank_sum = 0.0
        index = 0
        while index < len(ranked):
            end = index + 1
            while end < len(ranked) and ranked[end][0] == ranked[index][0]:
                end += 1
            average_rank = (index + 1 + end) / 2.0
            rank_sum += average_rank * sum(
                is_positive for _, is_positive in ranked[index:end]
            )
            index = end
        positive_num = len(positive_scores)
        negative_num = len(negative_scores)
        return (
            rank_sum - positive_num * (positive_num + 1) / 2.0
        ) / (positive_num * negative_num)

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

    def _detection_facts(
        self,
        artifact: ArtifactInterpretation,
        *,
        token_axis: bool,
    ) -> tuple[MetricFact, ...]:
        if artifact.dimensions.detector_scheme != "upv":
            return self._facts(
                artifact,
                "detection_rate",
                token_axis=token_axis,
                target_fpr=self._threshold(artifact),
            )
        return tuple(
            fact
            for fact in self._facts(
                artifact,
                "detection_rate",
                token_axis=token_axis,
            )
            if (
                fact.dimensions.detection_operating_point or {}
            ).get("kind")
            == "classifier-threshold"
        )

    def _one_detection_fact(
        self,
        artifact: ArtifactInterpretation,
        *,
        token_axis: bool,
    ) -> MetricFact:
        facts = self._detection_facts(
            artifact,
            token_axis=token_axis,
        )
        if len(facts) != 1:
            raise PlanValidationError(
                f"Artifact {artifact.identity.value} must provide exactly "
                "one selected detection operating point for "
                f"token_axis={token_axis!r}; found {len(facts)}"
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
            facts = self._detection_facts(artifact, token_axis=True)
            if not facts:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} has no token-level "
                    "detection facts at its selected operating point"
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
        if len(baseline) > 1:
            raise PlanValidationError(
                f"{recipe} accepts at most one unwatermarked PPL baseline"
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
            fact = self._one_detection_fact(
                detection,
                token_axis=False,
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

        detections = tuple(
            artifact for artifact in artifacts if artifact.kind == "detection"
        )
        negative_detections: dict[str, ArtifactInterpretation] = {}
        for artifact in detections:
            detector_identity = artifact.dimensions.detector_scheme_identity
            if detector_identity is None:
                raise PlanValidationError(
                    f"Detection Artifact {artifact.identity.value} lacks a "
                    "detector scheme"
                )
            if artifact.dimensions.scheme != "none":
                continue
            if detector_identity in negative_detections:
                raise PlanValidationError(
                    f"Detector {detector_identity} has more than one "
                    "unwatermarked reference Artifact"
                )
            negative_detections[detector_identity] = artifact
        auc_enabled = bool(negative_detections)

        rows = []
        transformed_detections: dict[str, ArtifactInterpretation] = {}
        for artifact in detections:
            detector_identity = artifact.dimensions.detector_scheme_identity
            scalar_facts = self._detection_facts(
                artifact,
                token_axis=False,
            )
            token_facts = self._detection_facts(
                artifact,
                token_axis=True,
            )
            if not scalar_facts:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} has no selected "
                    "detection fact at its selected operating point"
                )
            facts = scalar_facts + token_facts
            if artifact.dimensions.scheme == "none":
                rows.extend(
                    self._row(
                        recipe=recipe,
                        metric="false_positive_rate",
                        fact=fact,
                        sources=(artifact,),
                    )
                    for fact in facts
                )
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
                if auc_enabled:
                    negative = negative_detections.get(detector_identity)
                    if negative is None:
                        raise PlanValidationError(
                            f"Detector {detector_identity} has no "
                            "unwatermarked reference Artifact"
                        )
                    positive_scores = self._robustness_scores_by_token(
                        artifact
                    )
                    negative_scores = self._robustness_scores_by_token(
                        negative
                    )
                    self._validate_robustness_score_curve(
                        artifact,
                        facts,
                        positive_scores,
                    )
                    negative_facts = (
                        self._detection_facts(negative, token_axis=False)
                        + self._detection_facts(negative, token_axis=True)
                    )
                    self._validate_robustness_score_curve(
                        negative,
                        negative_facts,
                        negative_scores,
                    )
                    for token_num in sorted(
                        set(positive_scores) & set(negative_scores),
                        key=lambda value: (
                            value is not None,
                            value if value is not None else -1,
                        ),
                    ):
                        positive_values = positive_scores[token_num]
                        negative_values = negative_scores[token_num]
                        auc_fact = MetricFact(
                            metric="roc_auc",
                            scope=MetricScope.AGGREGATE,
                            value=self._roc_auc(
                                positive_values,
                                negative_values,
                            ),
                            dimensions=replace(
                                artifact.dimensions,
                                token_num=token_num,
                                target_fpr=None,
                                detection_operating_point=None,
                            ),
                            source_artifact=artifact.identity,
                            count=ExactCount(
                                len(positive_values) + len(negative_values)
                            ),
                        )
                        rows.append(
                            self._row(
                                recipe=recipe,
                                metric="roc_auc",
                                fact=auc_fact,
                                sources=(artifact, negative),
                            )
                        )
            if artifact.dimensions.detector_scheme_identity != (
                artifact.dimensions.scheme_identity
            ):
                raise PlanValidationError(
                    f"Positive Artifact {artifact.identity.value} was scored "
                    "with a detector for another watermark scheme"
                )
            rows.extend(
                self._row(
                    recipe=recipe,
                    metric=(
                        "true_positive_rate"
                        if auc_enabled
                        else "detection_rate"
                    ),
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

    def _robustness_scores_by_token(
        self,
        artifact: ArtifactInterpretation,
    ) -> dict[int | None, list[float]]:
        confidence = artifact.dimensions.detector_scheme == "upv"
        metric = "classifier_confidence" if confidence else "p_value"
        facts = self.interpretations.stream_sample_facts(
            artifact.identity,
            metrics=(metric,),
        )
        result: dict[int | None, list[float]] = {}
        for fact in facts:
            if fact.metric != metric or fact.value is None:
                continue
            value = float(fact.value)
            result.setdefault(fact.dimensions.token_num, []).append(
                value if confidence else -value
            )
        final_values = result.get(None, [])
        if len(final_values) != artifact.record_count:
            raise PlanValidationError(
                f"Artifact {artifact.identity.value} provides "
                f"{len(final_values)} final {metric} scores for "
                f"{artifact.record_count} records"
            )
        return result

    @staticmethod
    def _validate_robustness_score_curve(
        artifact: ArtifactInterpretation,
        facts: tuple[MetricFact, ...],
        scores: dict[int | None, list[float]],
    ) -> None:
        fact_counts = {
            fact.dimensions.token_num: fact.count.sample_num
            for fact in facts
            if fact.count is not None
        }
        score_counts = {
            token_num: len(values) for token_num, values in scores.items()
        }
        if fact_counts != score_counts:
            raise PlanValidationError(
                f"Artifact {artifact.identity.value} milestone detection "
                f"counts disagree with Sample scores: facts={fact_counts}, "
                f"scores={score_counts}"
            )

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
        self._require_comparable_population(artifacts, recipe=recipe)
        generation_models = {
            json.dumps(
                artifact.dimensions.generation_model,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for artifact in artifacts
        }
        if len(generation_models) != 1:
            raise PlanValidationError(
                f"{recipe} requires one generation model"
            )

        baseline = tuple(
            artifact
            for artifact in perplexities
            if artifact.dimensions.scheme == "none"
        )
        if len(baseline) != 1:
            raise PlanValidationError(
                f"{recipe} requires exactly one unwatermarked PPL baseline"
            )
        detection_ids = {
            detection.identity.value for detection in detections
        }
        ppl_by_detection: dict[str, ArtifactInterpretation] = {}
        honest_ppl_by_scheme: dict[str, ArtifactInterpretation] = {}
        for ppl in perplexities:
            if ppl in baseline:
                continue
            parents = [
                source.value
                for source in ppl.direct_source_identities
                if source.value in detection_ids
            ]
            if not parents:
                if len(ppl.direct_source_identities) != 1:
                    raise PlanValidationError(
                        f"PPL Artifact {ppl.identity.value} must have one "
                        "direct source"
                    )
                source = self.interpretations.artifact(
                    ppl.direct_source_identities[0]
                )
                scheme_identity = ppl.dimensions.scheme_identity
                if (
                    source.kind != "generation"
                    or ppl.dimensions.scheme in {None, "none"}
                    or scheme_identity is None
                    or scheme_identity in honest_ppl_by_scheme
                ):
                    raise PlanValidationError(
                        f"PPL Artifact {ppl.identity.value} is not a unique "
                        "honest-watermarked generation control"
                    )
                honest_ppl_by_scheme[scheme_identity] = ppl
                continue
            if len(parents) != 1 or parents[0] in ppl_by_detection:
                raise PlanValidationError(
                    f"PPL Artifact {ppl.identity.value} must uniquely pair "
                    "with one forgery detection Artifact"
                )
            ppl_by_detection[parents[0]] = ppl
        missing_ppl = sorted(
            detection.identity.value
            for detection in detections
            if detection.identity.value not in ppl_by_detection
        )
        if missing_ppl:
            raise PlanValidationError(
                f"{recipe} lacks PPL Artifacts for {missing_ppl}"
            )

        rows = []
        overall_metrics = {
            "p_value",
            "negative_log10_p_value",
            "green_token_count",
            "effective_token_count",
            "oracle_query_count",
            "total_oracle_query_count",
            "green_ratio",
            "green_count_threshold",
            "query_overhead_vs_honest_audit",
        }
        for detection in detections:
            if len(detection.direct_source_identities) != 1:
                raise PlanValidationError(
                    f"Detection Artifact {detection.identity.value} must "
                    "have one forgery parent"
                )
            forgery = self.interpretations.artifact(
                detection.direct_source_identities[0]
            )
            if forgery.kind != "adaptive-forgery":
                raise PlanValidationError(
                    f"Detection Artifact {detection.identity.value} is not "
                    "derived from an adaptive forgery"
                )
            ppl = ppl_by_detection[detection.identity.value]
            if (
                detection.dimensions.scheme_identity
                != ppl.dimensions.scheme_identity
            ):
                raise PlanValidationError(
                    f"PPL Artifact {ppl.identity.value} and detection "
                    f"{detection.identity.value} disagree on scheme"
                )
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="attack_success_rate",
                    fact=self._one_detection_fact(
                        detection,
                        token_axis=False,
                    ),
                    sources=(detection,),
                )
            )
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
            for metric in (
                "theoretical_green_probability",
                "theoretical_queries_per_scored_token",
            ):
                rows.append(
                    self._row(
                        recipe=recipe,
                        metric=metric,
                        fact=self._one_fact(
                            forgery,
                            metric,
                            token_axis=False,
                        ),
                        sources=(detection,),
                    )
                )
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="conditional_perplexity",
                    fact=self._one_fact(
                        ppl, "conditional_perplexity", token_axis=False
                    ),
                    sources=(ppl,),
                )
            )

        for control_ppl in (
            *honest_ppl_by_scheme.values(),
            baseline[0],
        ):
            rows.append(
                self._row(
                    recipe=recipe,
                    metric="conditional_perplexity",
                    fact=self._one_fact(
                        control_ppl,
                        "conditional_perplexity",
                        token_axis=False,
                    ),
                    sources=(control_ppl,),
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
        operating_points: dict[str, JsonObject] = {}
        for row in rows:
            value = row.get("dimensions", {}).get(
                "detection_operating_point"
            )
            if isinstance(value, dict):
                key = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                operating_points[key] = value
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
            "detection_operating_points": [
                operating_points[key] for key in sorted(operating_points)
            ],
            "confidence_level": self.confidence_level,
            "compatibility": {"status": "passed"},
        }
        return rows, summary
