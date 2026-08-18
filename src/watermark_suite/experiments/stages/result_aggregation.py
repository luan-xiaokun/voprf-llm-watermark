from __future__ import annotations

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..artifact_interpretation import (
    ARTIFACT_INTERPRETATION_REVISION,
    interpret_artifacts,
)
from ..errors import PlanValidationError
from ..fpr_calibration import FprCalibrationAssembler
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..result_assembly import REPORT_RECIPES, ResultAssembler
from .common import require


class ResultAggregationStageAdapter:
    kind = "result-aggregation"
    revision = (
        f"result-aggregation-v6+{ARTIFACT_INTERPRETATION_REVISION}"
    )
    accepted_settings = {
        "recipe",
        "confidence_level",
        "threshold_policy",
        "significance_levels",
    }
    _required = {"recipe", "confidence_level"}

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition:
        del context
        require(settings, self._required, kind=self.kind)
        recipe = settings["recipe"]
        if recipe not in REPORT_RECIPES:
            raise PlanValidationError(
                "recipe must be one of: " + ", ".join(sorted(REPORT_RECIPES))
            )
        confidence = settings["confidence_level"]
        if (
            not isinstance(confidence, (int, float))
            or not 0 < float(confidence) < 1
        ):
            raise PlanValidationError(
                "confidence_level must be between zero and one"
            )
        policy = settings.get("threshold_policy")
        significance_levels = settings.get("significance_levels")
        if recipe == "fpr-calibration":
            if (
                not isinstance(significance_levels, list)
                or not significance_levels
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not 0 < float(value) < 1
                    for value in significance_levels
                )
            ):
                raise PlanValidationError(
                    "fpr-calibration requires significance_levels between zero and one"
                )
            if policy is not None:
                raise PlanValidationError(
                    "fpr-calibration does not use threshold_policy"
                )
            resolved_policy = {}
            resolved_levels = sorted(
                {float(value) for value in significance_levels},
                reverse=True,
            )
        else:
            if (
                not isinstance(policy, dict)
                or "default" not in policy
                or any(
                    not isinstance(name, str)
                    or not isinstance(value, (int, float))
                    or not 0 < float(value) < 1
                    for name, value in policy.items()
                )
            ):
                raise PlanValidationError(
                    "threshold_policy must map scheme names to thresholds "
                    "between zero and one and include default"
                )
            if significance_levels is not None:
                raise PlanValidationError(
                    "significance_levels is only valid for fpr-calibration"
                )
            resolved_policy = {
                name: float(value) for name, value in policy.items()
            }
            resolved_levels = []
        semantic = {
            "recipe": recipe,
            "confidence_level": float(confidence),
            "threshold_policy": resolved_policy,
            "significance_levels": resolved_levels,
        }
        return ResolvedStageDefinition(
            settings=dict(settings),
            semantic_settings=semantic,
            execution_settings={},
            artifact_schema_revision="experiment-report-v2",
            resource_key="cpu:result-aggregation",
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if not inputs:
            raise PlanValidationError(
                "result-aggregation requires source Artifacts"
            )
        return definition

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> "_ResultAggregationExecution":
        return _ResultAggregationExecution(context)


class _ResultAggregationExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        self._summary: JsonObject | None = None

    def work_items(self) -> list[WorkItem]:
        source_ids = tuple(
            artifact.identity.value for artifact in self.context.inputs
        )
        return [
            WorkItem(
                identity=identity_for(
                    {
                        "recipe": self.context.semantic_settings["recipe"],
                        "sources": source_ids,
                    },
                    prefix="assembly",
                ),
                ordinal=0,
                sample_identities=source_ids,
                payload=None,
            )
        ]

    def execute(self, item: WorkItem) -> WorkResult:
        del item
        rows, self._summary = self._assemble()
        return WorkResult(records=tuple(rows))

    def _assemble(self) -> tuple[list[JsonObject], JsonObject]:
        semantic = self.context.semantic_settings
        if semantic["recipe"] == "fpr-calibration":
            return FprCalibrationAssembler(
                self.context.inputs,
                significance_levels=semantic["significance_levels"],
                confidence_level=semantic["confidence_level"],
            ).assemble()
        interpretations = interpret_artifacts(
            self.context.workspace,
            self.context.inputs,
        )
        assembler = ResultAssembler(
            interpretations,
            confidence_level=semantic["confidence_level"],
            threshold_policy=semantic["threshold_policy"],
        )
        return assembler.assemble(recipe=semantic["recipe"])

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        if self._summary is None:
            expected, self._summary = self._assemble()
            if records != expected:
                raise PlanValidationError(
                    "resumed result-aggregation records differ from sources"
                )
        if len(records) != self._summary["metric_row_num"]:
            raise PlanValidationError(
                "result-aggregation record count changed before finalization"
            )
        return self._summary

    def close(self) -> None:
        pass
