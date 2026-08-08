from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from ..errors import PlanValidationError
from ..identity import canonical_json
from ..models import ArtifactIdentity, ArtifactRef, JsonObject
from ..workspace import ExperimentWorkspace


ARTIFACT_INTERPRETATION_REVISION = "artifact-interpretation-v4"


class MetricScope(StrEnum):
    AGGREGATE = "aggregate"
    SAMPLE = "sample"


@dataclass(frozen=True)
class ExactCount:
    sample_num: int
    positive_num: int | None = None

    def __post_init__(self) -> None:
        if self.sample_num < 0:
            raise ValueError("sample_num must be non-negative")
        if self.positive_num is not None and not (
            0 <= self.positive_num <= self.sample_num
        ):
            raise ValueError(
                "positive_num must be between zero and sample_num"
            )

    def to_dict(self) -> JsonObject:
        result: JsonObject = {"sample_num": self.sample_num}
        if self.positive_num is not None:
            result["positive_num"] = self.positive_num
        return result


@dataclass(frozen=True)
class Distribution:
    count: int
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    p05: float | None = None
    p25: float | None = None
    p75: float | None = None
    p95: float | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("distribution count must be non-negative")
        for value in (
            self.mean,
            self.median,
            self.std,
            self.minimum,
            self.maximum,
            self.p05,
            self.p25,
            self.p75,
            self.p95,
        ):
            if value is not None and not math.isfinite(value):
                raise ValueError("distribution values must be finite")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("distribution minimum exceeds maximum")

    def to_dict(self) -> JsonObject:
        values = {
            "count": self.count,
            "mean": self.mean,
            "median": self.median,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "p05": self.p05,
            "p25": self.p25,
            "p75": self.p75,
            "p95": self.p95,
        }
        return {
            key: value
            for key, value in values.items()
            if value is not None or key in {"count", "mean", "median"}
        }


@dataclass(frozen=True)
class ScientificDimensions:
    scheme: str | None = None
    scheme_identity: str | None = None
    watermark_parameters: JsonObject | None = None
    detector_scheme: str | None = None
    detector_scheme_identity: str | None = None
    detector_watermark_parameters: JsonObject | None = None
    population_identity: str | None = None
    generation_model: JsonObject | None = None
    generation: JsonObject | None = None
    evaluation_model: JsonObject | None = None
    transformation: JsonObject | None = None
    task: str | None = None
    token_num: int | None = None
    target_fpr: float | None = None
    detection_operating_point: JsonObject | None = None

    def with_metric_axes(
        self,
        *,
        token_num: int | None = None,
        target_fpr: float | None = None,
        detection_operating_point: JsonObject | None = None,
    ) -> "ScientificDimensions":
        return replace(
            self,
            token_num=token_num,
            target_fpr=target_fpr,
            detection_operating_point=detection_operating_point,
        )

    def to_dict(self) -> JsonObject:
        return {
            "scheme": self.scheme,
            "scheme_identity": self.scheme_identity,
            "watermark_parameters": dict(
                self.watermark_parameters or {}
            ),
            "detector_scheme": self.detector_scheme,
            "detector_scheme_identity": self.detector_scheme_identity,
            "detector_watermark_parameters": dict(
                self.detector_watermark_parameters or {}
            ),
            "population_identity": self.population_identity,
            "generation_model": self.generation_model,
            "generation": self.generation,
            "evaluation_model": self.evaluation_model,
            "transformation": self.transformation,
            "task": self.task,
            "token_num": self.token_num,
            "target_fpr": self.target_fpr,
            "detection_operating_point": self.detection_operating_point,
        }


@dataclass(frozen=True)
class MetricFact:
    metric: str
    scope: MetricScope
    value: float | int | None
    dimensions: ScientificDimensions
    source_artifact: ArtifactIdentity
    count: ExactCount | None = None
    distribution: Distribution | None = None
    sample_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("metric name must be non-empty")
        if self.scope == MetricScope.SAMPLE and not self.sample_identity:
            raise ValueError("Sample Metric Facts require a Sample identity")
        if self.scope == MetricScope.AGGREGATE and self.sample_identity:
            raise ValueError(
                "aggregate Metric Facts cannot have a Sample identity"
            )
        if isinstance(self.value, float) and not math.isfinite(self.value):
            raise ValueError("Metric Fact values must be finite")


@dataclass(frozen=True)
class ArtifactRelation:
    artifact_identity: ArtifactIdentity
    source_identity: ArtifactIdentity
    direct: bool
    path: tuple[ArtifactIdentity, ...]
    roles: tuple[str, ...]


@dataclass(frozen=True)
class ArtifactInterpretation:
    identity: ArtifactIdentity
    kind: str
    schema_revision: str
    is_root: bool
    dimensions: ScientificDimensions
    relations: tuple[ArtifactRelation, ...]
    aggregate_facts: tuple[MetricFact, ...]
    record_count: int

    @property
    def direct_source_identities(self) -> tuple[ArtifactIdentity, ...]:
        return tuple(
            relation.source_identity
            for relation in self.relations
            if relation.direct
        )

    def facts(self, metric: str | None = None) -> tuple[MetricFact, ...]:
        return tuple(
            fact
            for fact in self.aggregate_facts
            if metric is None or fact.metric == metric
        )


@dataclass(frozen=True)
class InterpretationIssue:
    root_identity: str
    artifact_identity: str
    path: tuple[str, ...]
    code: str
    message: str
    expected: object | None = None
    observed: object | None = None

    def sort_key(self) -> tuple[object, ...]:
        return (
            self.root_identity,
            self.path,
            self.artifact_identity,
            self.code,
            self.message,
        )


class ArtifactInterpretationError(PlanValidationError):
    def __init__(self, issues: Iterable[InterpretationIssue]) -> None:
        self.issues = tuple(sorted(issues, key=InterpretationIssue.sort_key))
        lines = ["Artifact Interpretation preflight failed:"]
        for issue in self.issues:
            path = " -> ".join(issue.path)
            detail = ""
            if issue.expected is not None or issue.observed is not None:
                detail = (
                    f" (expected={issue.expected!r}, "
                    f"observed={issue.observed!r})"
                )
            lines.append(
                f"  [{issue.code}] root={issue.root_identity} "
                f"path={path}: {issue.message}{detail}"
            )
        super().__init__("\n".join(lines))


@dataclass(frozen=True)
class _RawMetricFact:
    metric: str
    value: float | int | None
    token_num: int | None = None
    target_fpr: float | None = None
    detection_operating_point: JsonObject | None = None
    count: ExactCount | None = None
    distribution: Distribution | None = None


@dataclass(frozen=True)
class _ParsedArtifact:
    claims: Mapping[str, object]
    evaluation_model: JsonObject | None
    detector_scheme: JsonObject | None
    inherited_model_claim: JsonObject | None
    raw_facts: tuple[_RawMetricFact, ...]
    summary_sample_num: int | None


SampleFactReader = Callable[
    [ArtifactRef, ScientificDimensions, frozenset[str] | None],
    Iterator[MetricFact],
]


@dataclass(frozen=True)
class _SchemaAdapter:
    kind: str
    schema_revision: str
    parse: Callable[
        [ArtifactRef, JsonObject, list[InterpretationIssue], str, tuple[str, ...]],
        _ParsedArtifact,
    ]
    sample_facts: SampleFactReader


class InterpretationSet:
    def __init__(
        self,
        *,
        roots: tuple[ArtifactIdentity, ...],
        artifacts: tuple[ArtifactInterpretation, ...],
        refs: Mapping[str, ArtifactRef],
        sample_readers: Mapping[str, SampleFactReader],
    ) -> None:
        self.roots = roots
        self.artifacts = artifacts
        self._by_identity = {
            artifact.identity.value: artifact for artifact in artifacts
        }
        self._refs = dict(refs)
        self._sample_readers = dict(sample_readers)

    @property
    def root_artifacts(self) -> tuple[ArtifactInterpretation, ...]:
        return tuple(
            self._by_identity[identity.value] for identity in self.roots
        )

    def artifact(
        self,
        identity: ArtifactIdentity | str,
    ) -> ArtifactInterpretation:
        value = identity.value if isinstance(identity, ArtifactIdentity) else identity
        try:
            return self._by_identity[value]
        except KeyError as error:
            raise KeyError(f"unknown interpreted Artifact {value}") from error

    def stream_sample_facts(
        self,
        identity: ArtifactIdentity | str,
        *,
        metrics: Iterable[str] | None = None,
    ) -> Iterator[MetricFact]:
        value = identity.value if isinstance(identity, ArtifactIdentity) else identity
        interpretation = self.artifact(value)
        selected = frozenset(metrics) if metrics is not None else None
        reader = self._sample_readers[value]

        def validated() -> Iterator[MetricFact]:
            counts: dict[tuple[str, int | None], int] = {}
            totals: dict[tuple[str, int | None], float] = {}
            for fact in reader(
                self._refs[value],
                interpretation.dimensions,
                selected,
            ):
                key = (fact.metric, fact.dimensions.token_num)
                counts[key] = counts.get(key, 0) + 1
                if fact.value is not None:
                    totals[key] = (
                        totals.get(key, 0.0) + float(fact.value)
                    )
                yield fact
            for aggregate in interpretation.aggregate_facts:
                if selected is not None and aggregate.metric not in selected:
                    continue
                key = (aggregate.metric, aggregate.dimensions.token_num)
                sample_count = counts.get(key, 0)
                if aggregate.distribution is not None:
                    expected = aggregate.distribution.count
                    if sample_count != expected:
                        raise PlanValidationError(
                            f"Sample Metric Facts for {value}:"
                            f"{aggregate.metric} contain {sample_count} "
                            f"values, aggregate distribution declares "
                            f"{expected}"
                        )
                    if (
                        expected
                        and aggregate.distribution.mean is not None
                        and not math.isclose(
                            totals[key] / expected,
                            aggregate.distribution.mean,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    ):
                        raise PlanValidationError(
                            f"Sample Metric Facts for {value}:"
                            f"{aggregate.metric} disagree with aggregate mean"
                        )
                if (
                    aggregate.count is not None
                    and aggregate.count.positive_num is not None
                    and aggregate.metric in {"accuracy", "pass_at_1"}
                    and (
                        sample_count != aggregate.count.sample_num
                        or not math.isclose(
                            totals.get(key, 0.0),
                            aggregate.count.positive_num,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        )
                    )
                ):
                    raise PlanValidationError(
                        f"Sample Metric Facts for {value}:"
                        f"{aggregate.metric} disagree with exact count"
                    )
                if (
                    aggregate.metric == "mean_character_ratio"
                    and aggregate.value is not None
                    and (
                        sample_count != interpretation.record_count
                        or (
                            sample_count > 0
                            and not math.isclose(
                                totals.get(key, 0.0) / sample_count,
                                float(aggregate.value),
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            )
                        )
                    )
                ):
                    raise PlanValidationError(
                        f"Sample Metric Facts for {value}:"
                        "character_ratio disagrees with aggregate mean"
                    )

        return validated()


def _read_object(
    path: Path,
    *,
    issues: list[InterpretationIssue],
    root: str,
    artifact: str,
    lineage_path: tuple[str, ...],
    code: str,
) -> JsonObject | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        issues.append(
            InterpretationIssue(
                root_identity=root,
                artifact_identity=artifact,
                path=lineage_path,
                code=code,
                message=f"{path.name} is not valid JSON: {error}",
            )
        )
        return None
    if not isinstance(value, dict):
        issues.append(
            InterpretationIssue(
                root_identity=root,
                artifact_identity=artifact,
                path=lineage_path,
                code=code,
                message=f"{path.name} must contain a JSON object",
            )
        )
        return None
    return value


def _canonical(value: object) -> str:
    return canonical_json(value)


def _relation_paths(
    identity: str,
    refs: Mapping[str, ArtifactRef],
) -> tuple[tuple[str, ...], ...]:
    result: list[tuple[str, ...]] = []

    def visit(current: str, path: tuple[str, ...]) -> None:
        ref = refs[current]
        for source in ref.source_artifacts:
            next_path = (*path, source.value)
            result.append(next_path)
            if (
                source.value in refs
                and source.value not in path
                and source.value != current
            ):
                visit(source.value, next_path)

    visit(identity, (identity,))
    return tuple(result)


def _ancestor_ids(
    identity: str,
    refs: Mapping[str, ArtifactRef],
) -> tuple[str, ...]:
    result = [identity]
    seen = {identity}
    for path in _relation_paths(identity, refs):
        candidate = path[-1]
        if candidate not in seen:
            seen.add(candidate)
            result.append(candidate)
    return tuple(result)


def _effective_dimensions(
    *,
    identity: str,
    refs: Mapping[str, ArtifactRef],
    parsed: Mapping[str, _ParsedArtifact],
    roots_and_paths: tuple[tuple[str, tuple[str, ...]], ...],
    issues: list[InterpretationIssue],
) -> ScientificDimensions:
    lineage = _ancestor_ids(identity, refs)
    effective: dict[str, object | None] = {}
    for name in ("population", "generation", "scheme", "transformation", "task"):
        claims = [
            (artifact_id, parsed[artifact_id].claims[name])
            for artifact_id in lineage
            if artifact_id in parsed
            and parsed[artifact_id].claims.get(name) is not None
        ]
        unique = {_canonical(value) for _, value in claims}
        if len(unique) > 1:
            for root, root_path in roots_and_paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=root,
                        artifact_identity=identity,
                        path=root_path,
                        code="lineage-invariant-conflict",
                        message=f"Lineage Invariant {name!r} conflicts",
                        expected=claims[0][1],
                        observed=claims[1][1],
                    )
                )
        effective[name] = claims[0][1] if claims else None

    generation = effective["generation"]
    generation_model = (
        generation.get("model")
        if isinstance(generation, dict)
        else None
    )
    for artifact_id in lineage:
        if artifact_id not in parsed:
            continue
        claim = parsed[artifact_id].inherited_model_claim
        if (
            claim is not None
            and generation_model is not None
            and _canonical(claim) != _canonical(generation_model)
        ):
            for root, root_path in roots_and_paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=root,
                        artifact_identity=artifact_id,
                        path=root_path,
                        code="lineage-invariant-conflict",
                        message="inherited generation model conflicts",
                        expected=generation_model,
                        observed=claim,
                    )
                )
    scheme = effective["scheme"]
    detector_scheme = parsed[identity].detector_scheme
    return ScientificDimensions(
        scheme=(
            str(scheme["method"]) if isinstance(scheme, dict) else None
        ),
        scheme_identity=(
            str(scheme["identity"]) if isinstance(scheme, dict) else None
        ),
        watermark_parameters=(
            dict(scheme.get("parameters", {}))
            if isinstance(scheme, dict)
            else {}
        ),
        detector_scheme=(
            str(detector_scheme["method"])
            if isinstance(detector_scheme, dict)
            else None
        ),
        detector_scheme_identity=(
            str(detector_scheme["identity"])
            if isinstance(detector_scheme, dict)
            else None
        ),
        detector_watermark_parameters=(
            dict(detector_scheme.get("parameters", {}))
            if isinstance(detector_scheme, dict)
            else {}
        ),
        population_identity=(
            str(effective["population"]["identity"])
            if isinstance(effective["population"], dict)
            else None
        ),
        generation_model=(
            dict(generation_model)
            if isinstance(generation_model, dict)
            else None
        ),
        generation=(
            dict(generation) if isinstance(generation, dict) else None
        ),
        evaluation_model=parsed[identity].evaluation_model,
        transformation=(
            dict(effective["transformation"])
            if isinstance(effective["transformation"], dict)
            else None
        ),
        task=(
            str(effective["task"])
            if isinstance(effective["task"], str)
            else None
        ),
    )


def _missing_dimension_roles(
    ref: ArtifactRef,
    dimensions: ScientificDimensions,
) -> tuple[str, ...]:
    required = {"population", "generation", "generation_model", "scheme"}
    kind = str(ref.manifest.get("kind"))
    if kind == "robustness":
        required.add("transformation")
    if kind == "detection":
        required.add("detector_scheme")
    if kind in {"perplexity", "text-evaluation", "downstream"}:
        required.add("evaluation_model")
    if kind == "downstream":
        required.add("task")
    values = {
        "population": dimensions.population_identity,
        "generation": dimensions.generation,
        "generation_model": dimensions.generation_model,
        "scheme": dimensions.scheme,
        "detector_scheme": dimensions.detector_scheme,
        "transformation": dimensions.transformation,
        "evaluation_model": dimensions.evaluation_model,
        "task": dimensions.task,
    }
    return tuple(sorted(name for name in required if values[name] is None))


def _fact_issue(
    fact: _RawMetricFact,
    *,
    record_count: int,
) -> str | None:
    if fact.token_num is not None and fact.token_num < 0:
        return "token_num must be non-negative"
    if fact.target_fpr is not None and not 0 < fact.target_fpr < 1:
        return "target_fpr must be between zero and one"
    if (
        fact.detection_operating_point is not None
        and not isinstance(fact.detection_operating_point, dict)
    ):
        return "detection_operating_point must be a mapping"
    if fact.metric in {
        "detection_rate",
        "attack_success_rate",
        "accuracy",
        "pass_at_1",
        "selected_green_ratio",
        "fallback_ratio",
        "cache_hit_ratio",
        "duplicate_selected_pair_ratio",
        "detector_green_ratio",
        "tokenization_preserved_ratio",
        "mean_selected_green_ratio",
        "theoretical_green_probability",
        "oracle_time_fraction",
    } and fact.value is not None and not 0 <= float(fact.value) <= 1:
        return f"{fact.metric} must be between zero and one"
    if (
        fact.metric in {"median_p_value"}
        and fact.value is not None
        and not 0 <= float(fact.value) <= 1
    ):
        return f"{fact.metric} must be between zero and one"
    if fact.value is not None and (
        "query" in fact.metric
        or "token_num" in fact.metric
        or fact.metric.endswith("_count")
        or fact.metric.endswith("_seconds")
        or fact.metric.endswith("_bytes")
        or fact.metric in {
            "conditional_perplexity",
            "local_model_perplexity",
            "vendi_score",
            "openai_mean_latency_seconds",
        }
    ) and float(fact.value) < 0:
        return f"{fact.metric} must be non-negative"
    if fact.metric in {
        "p_value",
        "classifier_confidence",
    } and fact.distribution is not None:
        if (
            fact.distribution.minimum is not None
            and fact.distribution.minimum < 0
        ) or (
            fact.distribution.maximum is not None
            and fact.distribution.maximum > 1
        ):
            return f"{fact.metric} distribution must lie between zero and one"
    if fact.count is not None and fact.count.sample_num > record_count:
        return (
            f"fact population {fact.count.sample_num} exceeds manifest "
            f"record_count {record_count}"
        )
    if (
        fact.distribution is not None
        and fact.distribution.count > record_count
    ):
        return (
            f"distribution population {fact.distribution.count} exceeds "
            f"manifest record_count {record_count}"
        )
    if (
        fact.distribution is not None
        and fact.count is not None
        and fact.distribution.count > fact.count.sample_num
    ):
        return (
            f"distribution population {fact.distribution.count} exceeds "
            f"fact population {fact.count.sample_num}"
        )
    if (
        fact.metric
        in {
            "detection_rate",
            "conditional_perplexity",
            "cosine_similarity",
            "p_value",
            "accuracy",
            "pass_at_1",
        }
        and fact.token_num is None
        and fact.count is not None
        and fact.count.sample_num != record_count
    ):
        return (
            f"{fact.metric} population {fact.count.sample_num} differs "
            f"from manifest record_count {record_count}"
        )
    return None


def interpret_artifacts(
    workspace: Path | ExperimentWorkspace,
    roots: tuple[ArtifactRef, ...],
) -> InterpretationSet:
    if not roots:
        raise ArtifactInterpretationError(
            [
                InterpretationIssue(
                    root_identity="<none>",
                    artifact_identity="<none>",
                    path=(),
                    code="empty-input",
                    message="Artifact Interpretation requires root Artifacts",
                )
            ]
        )
    experiment_workspace = (
        workspace
        if isinstance(workspace, ExperimentWorkspace)
        else ExperimentWorkspace(workspace)
    )
    from ._schemas import schema_registry

    issues: list[InterpretationIssue] = []
    refs: dict[str, ArtifactRef] = {}
    root_paths: dict[tuple[str, str], tuple[str, ...]] = {}
    supplied_roots = {
        root.identity.value: root for root in roots
    }
    if len(supplied_roots) != len(roots):
        duplicate_values = sorted(
            {
                root.identity.value
                for root in roots
                if sum(
                    candidate.identity == root.identity
                    for candidate in roots
                )
                > 1
            }
        )
        issues.append(
            InterpretationIssue(
                root_identity="<multiple>",
                artifact_identity="<multiple>",
                path=tuple(duplicate_values),
                code="duplicate-root",
                message="root Artifact identities must be unique",
                observed=duplicate_values,
            )
        )

    def load(
        identity: ArtifactIdentity,
        *,
        root: str,
        path: tuple[str, ...],
        active: frozenset[str],
    ) -> None:
        value = identity.value
        if value in active:
            issues.append(
                InterpretationIssue(
                    root_identity=root,
                    artifact_identity=value,
                    path=(*path, value),
                    code="lineage-cycle",
                    message="Artifact lineage contains a cycle",
                )
            )
            return
        root_paths.setdefault((root, value), (*path, value))
        if value not in refs:
            try:
                refs[value] = experiment_workspace.artifact(identity)
            except KeyError:
                issues.append(
                    InterpretationIssue(
                        root_identity=root,
                        artifact_identity=value,
                        path=(*path, value),
                        code="missing-artifact",
                        message="Artifact is absent from the Experiment Workspace",
                    )
                )
                return
            except Exception as error:
                issues.append(
                    InterpretationIssue(
                        root_identity=root,
                        artifact_identity=value,
                        path=(*path, value),
                        code="artifact-integrity",
                        message=str(error),
                    )
                )
                return
        ref = refs[value]
        manifest_sources = ref.manifest.get("source_artifacts", [])
        observed_sources = [item.value for item in ref.source_artifacts]
        supplied = supplied_roots.get(value)
        if supplied is not None:
            observed_sources = [
                item.value for item in supplied.source_artifacts
            ]
        if manifest_sources != observed_sources:
            issues.append(
                InterpretationIssue(
                    root_identity=root,
                    artifact_identity=value,
                    path=(*path, value),
                    code="source-relation-mismatch",
                    message="ArtifactRef and manifest sources disagree",
                    expected=manifest_sources,
                    observed=observed_sources,
                )
            )
        next_active = active | {value}
        for source in ref.source_artifacts:
            load(
                source,
                root=root,
                path=(*path, value),
                active=next_active,
            )

    for supplied in roots:
        load(
            supplied.identity,
            root=supplied.identity.value,
            path=(),
            active=frozenset(),
        )

    parsed: dict[str, _ParsedArtifact] = {}
    adapters: dict[str, _SchemaAdapter] = {}
    for identity, ref in sorted(refs.items()):
        paths = [
            (root, path)
            for (root, artifact), path in root_paths.items()
            if artifact == identity
        ]
        root, lineage_path = paths[0]
        adapter = schema_registry.get(
            (
                str(ref.manifest.get("kind")),
                ref.schema_revision,
            )
        )
        if adapter is None:
            schema_kinds = {
                kind
                for kind, revision in schema_registry
                if revision == ref.schema_revision
            }
            code = (
                "kind-schema-mismatch"
                if schema_kinds
                else "unknown-schema"
            )
            for issue_root, issue_path in paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code=code,
                        message=(
                            f"unsupported Artifact pair "
                            f"({ref.manifest.get('kind')!r}, "
                            f"{ref.schema_revision!r})"
                        ),
                        expected=sorted(schema_kinds) or None,
                        observed=ref.manifest.get("kind"),
                    )
                )
            continue
        summary = _read_object(
            ref.path / "summary.json",
            issues=issues,
            root=root,
            artifact=identity,
            lineage_path=lineage_path,
            code="invalid-summary",
        )
        if summary is None:
            for issue_root, issue_path in paths[1:]:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code="invalid-summary",
                        message="summary is invalid for this lineage",
                    )
                )
            continue
        try:
            parsed[identity] = adapter.parse(
                ref,
                summary,
                issues,
                root,
                lineage_path,
            )
            adapters[identity] = adapter
        except Exception as error:
            for issue_root, issue_path in paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code="schema-interpretation",
                        message=str(error),
                    )
                )

    interpretations: list[ArtifactInterpretation] = []
    root_ids = tuple(root.identity for root in roots)
    root_values = {identity.value for identity in root_ids}
    for identity in sorted(parsed):
        ref = refs[identity]
        roots_for_artifact = sorted(
            root
            for root, artifact in root_paths
            if artifact == identity
        )
        root = roots_for_artifact[0]
        roots_and_paths = tuple(
            (candidate, root_paths[(candidate, identity)])
            for candidate in roots_for_artifact
        )
        dimensions = _effective_dimensions(
            identity=identity,
            refs=refs,
            parsed=parsed,
            roots_and_paths=roots_and_paths,
            issues=issues,
        )
        relations = []
        for path in _relation_paths(identity, refs):
            roles = tuple(
                (
                    (
                        f"{refs[item].manifest.get('kind')}:"
                        f"{refs[item].schema_revision}"
                    )
                    if item in refs
                    else "<missing>"
                )
                for item in path
            )
            relations.append(
                ArtifactRelation(
                    artifact_identity=ArtifactIdentity(identity),
                    source_identity=ArtifactIdentity(path[-1]),
                    direct=len(path) == 2,
                    path=tuple(ArtifactIdentity(item) for item in path),
                    roles=roles,
                )
            )
        raw = parsed[identity]
        try:
            record_count = int(ref.manifest.get("record_count", -1))
        except (TypeError, ValueError):
            record_count = -1
        if record_count < 0:
            for issue_root, issue_path in roots_and_paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code="invalid-record-count",
                        message="manifest record_count must be non-negative",
                        observed=ref.manifest.get("record_count"),
                    )
                )
            record_count = 0
        if (
            raw.summary_sample_num is not None
            and raw.summary_sample_num != record_count
        ):
            for issue_root, issue_path in roots_and_paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code="population-count-mismatch",
                        message=(
                            "summary sample count differs from manifest records"
                        ),
                        expected=record_count,
                        observed=raw.summary_sample_num,
                    )
                )
        kind = str(ref.manifest.get("kind"))
        if kind in {
            "robustness",
            "detection",
            "perplexity",
            "text-evaluation",
        }:
            if len(ref.source_artifacts) != 1:
                for issue_root, issue_path in roots_and_paths:
                    issues.append(
                        InterpretationIssue(
                            root_identity=issue_root,
                            artifact_identity=identity,
                            path=issue_path,
                            code="source-cardinality",
                            message=(
                                f"{kind} Artifact requires exactly one "
                                "direct source"
                            ),
                            expected=1,
                            observed=len(ref.source_artifacts),
                        )
                    )
            elif ref.source_artifacts[0].value in refs:
                source_ref = refs[ref.source_artifacts[0].value]
                try:
                    source_count = int(
                        source_ref.manifest.get("record_count", -1)
                    )
                except (TypeError, ValueError):
                    source_count = -1
                if source_count >= 0 and source_count != record_count:
                    for issue_root, issue_path in roots_and_paths:
                        issues.append(
                            InterpretationIssue(
                                root_identity=issue_root,
                                artifact_identity=identity,
                                path=issue_path,
                                code="population-count-mismatch",
                                message=(
                                    f"{kind} record count differs from its "
                                    "one-to-one source Artifact"
                                ),
                                expected=source_count,
                                observed=record_count,
                            )
                        )
        missing_roles = _missing_dimension_roles(ref, dimensions)
        if missing_roles:
            for issue_root, issue_path in roots_and_paths:
                issues.append(
                    InterpretationIssue(
                        root_identity=issue_root,
                        artifact_identity=identity,
                        path=issue_path,
                        code="missing-scientific-role",
                        message=(
                            "Artifact has incomplete scientific provenance: "
                            + ", ".join(missing_roles)
                        ),
                        observed=list(missing_roles),
                    )
                )
        facts = []
        for raw_fact in raw.raw_facts:
            validation = _fact_issue(
                raw_fact,
                record_count=record_count,
            )
            if validation is not None:
                for issue_root, issue_path in roots_and_paths:
                    issues.append(
                        InterpretationIssue(
                            root_identity=issue_root,
                            artifact_identity=identity,
                            path=issue_path,
                            code="invalid-metric-fact",
                            message=validation,
                            observed=raw_fact.value,
                        )
                    )
                continue
            try:
                facts.append(
                    MetricFact(
                        metric=raw_fact.metric,
                        scope=MetricScope.AGGREGATE,
                        value=raw_fact.value,
                        dimensions=dimensions.with_metric_axes(
                            token_num=raw_fact.token_num,
                            target_fpr=raw_fact.target_fpr,
                            detection_operating_point=(
                                raw_fact.detection_operating_point
                            ),
                        ),
                        source_artifact=ArtifactIdentity(identity),
                        count=raw_fact.count,
                        distribution=raw_fact.distribution,
                    )
                )
            except (TypeError, ValueError) as error:
                for issue_root, issue_path in roots_and_paths:
                    issues.append(
                        InterpretationIssue(
                            root_identity=issue_root,
                            artifact_identity=identity,
                            path=issue_path,
                            code="invalid-metric-fact",
                            message=f"{raw_fact.metric}: {error}",
                            observed=raw_fact.value,
                        )
                    )
        interpretations.append(
            ArtifactInterpretation(
                identity=ArtifactIdentity(identity),
                kind=str(ref.manifest["kind"]),
                schema_revision=ref.schema_revision,
                is_root=identity in root_values,
                dimensions=dimensions,
                relations=tuple(relations),
                aggregate_facts=tuple(facts),
                record_count=record_count,
            )
        )

    if issues:
        raise ArtifactInterpretationError(issues)
    return InterpretationSet(
        roots=root_ids,
        artifacts=tuple(interpretations),
        refs=refs,
        sample_readers={
            identity: adapter.sample_facts
            for identity, adapter in adapters.items()
        },
    )
