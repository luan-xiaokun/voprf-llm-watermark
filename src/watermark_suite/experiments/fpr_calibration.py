from __future__ import annotations

import json
import math

from scipy.stats import beta

from .errors import PlanValidationError
from .identity import identity_for
from .models import ArtifactRef, JsonObject


def _summary(artifact: ArtifactRef) -> JsonObject:
    try:
        value = json.loads(
            (artifact.path / "summary.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise PlanValidationError(
            f"Artifact {artifact.identity.value} has an invalid summary"
        ) from error
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"Artifact {artifact.identity.value} summary must be a mapping"
        )
    return value


def _clopper_pearson(
    positive_num: int,
    sample_num: int,
    confidence_level: float,
) -> JsonObject:
    tail = (1.0 - confidence_level) / 2.0
    lower = (
        0.0
        if positive_num == 0
        else float(beta.ppf(tail, positive_num, sample_num - positive_num + 1))
    )
    upper = (
        1.0
        if positive_num == sample_num
        else float(
            beta.ppf(
                1.0 - tail,
                positive_num + 1,
                sample_num - positive_num,
            )
        )
    )
    return {
        "method": "clopper-pearson",
        "confidence_level": confidence_level,
        "lower": lower,
        "upper": upper,
    }


class FprCalibrationAssembler:
    """Validate null-detection Artifacts and build an empirical FPR table."""

    revision = "fpr-calibration-v1"

    def __init__(
        self,
        artifacts: tuple[ArtifactRef, ...],
        *,
        significance_levels: list[float],
        confidence_level: float,
    ) -> None:
        self.artifacts = artifacts
        self.significance_levels = significance_levels
        self.confidence_level = confidence_level

    def assemble(self) -> tuple[list[JsonObject], JsonObject]:
        if not self.artifacts:
            raise PlanValidationError(
                "fpr-calibration requires null-detection Artifacts"
            )
        summaries: list[tuple[ArtifactRef, JsonObject]] = []
        for artifact in self.artifacts:
            if (
                artifact.manifest.get("kind") != "null-detection"
                or artifact.schema_revision != "null-detection-v1"
            ):
                raise PlanValidationError(
                    "fpr-calibration accepts only null-detection-v1 Artifacts"
                )
            summary = _summary(artifact)
            if int(summary.get("sample_num", -1)) != int(
                artifact.manifest.get("record_count", -2)
            ):
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} record count disagrees "
                    "with its null-detection summary"
                )
            summaries.append((artifact, summary))

        population_ids = {
            summary.get("population", {}).get("identity")
            for _, summary in summaries
        }
        window_documents = {
            json.dumps(
                summary.get("window"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for _, summary in summaries
        }
        source_ids = {summary.get("source_artifact") for _, summary in summaries}
        if len(population_ids) != 1 or None in population_ids:
            raise PlanValidationError(
                "fpr-calibration inputs do not share one population"
            )
        if len(window_documents) != 1 or len(source_ids) != 1:
            raise PlanValidationError(
                "fpr-calibration inputs do not score the same token windows"
            )

        seen_schemes: set[str] = set()
        rows: list[JsonObject] = []
        for artifact, summary in summaries:
            scheme = summary.get("detector_scheme", {}).get("method")
            if not isinstance(scheme, str) or not scheme:
                raise PlanValidationError(
                    f"Artifact {artifact.identity.value} lacks detector scheme"
                )
            if scheme in seen_schemes:
                raise PlanValidationError(
                    f"fpr-calibration has more than one {scheme} Artifact"
                )
            seen_schemes.add(scheme)
            measured = summary.get("false_positive_counts", {})
            minimum = summary.get("p_value", {}).get("minimum_attainable")
            for level in self.significance_levels:
                level_key = f"{level:.0e}"
                count = measured.get(level_key)
                status = "measured"
                reason = None
                value = None
                uncertainty = None
                population: JsonObject = {
                    "sample_num": int(summary["sample_num"])
                }
                if count is not None:
                    positive = int(count["positive_num"])
                    total = int(count["sample_num"])
                    if total != int(summary["sample_num"]):
                        raise PlanValidationError(
                            f"Artifact {artifact.identity.value} has inconsistent "
                            f"counts at alpha={level:g}"
                        )
                    value = positive / total
                    population = {
                        "positive_num": positive,
                        "sample_num": total,
                    }
                    uncertainty = _clopper_pearson(
                        positive, total, self.confidence_level
                    )
                elif (
                    isinstance(minimum, (int, float))
                    and math.isfinite(float(minimum))
                    and float(minimum) >= level
                ):
                    status = "unsupported"
                    reason = (
                        f"minimum attainable p-value is {float(minimum):.6g}"
                    )
                else:
                    raise PlanValidationError(
                        f"Artifact {artifact.identity.value} did not measure "
                        f"alpha={level:g}"
                    )
                identity_document = {
                    "recipe": "fpr-calibration",
                    "scheme": scheme,
                    "nominal_alpha": level,
                    "source_artifact": artifact.identity.value,
                }
                rows.append(
                    {
                        "sample_id": identity_for(
                            identity_document, prefix="metric"
                        ),
                        "recipe": "fpr-calibration",
                        "metric": "false_positive_rate",
                        "value": value,
                        "status": status,
                        "reason": reason,
                        "dimensions": {
                            "scheme": scheme,
                            "nominal_alpha": level,
                            "window_token_num": summary["window"]["token_num"],
                            "population_identity": summary["population"][
                                "identity"
                            ],
                        },
                        "population": population,
                        "uncertainty": uncertainty,
                        "source_artifacts": [artifact.identity.value],
                    }
                )
        rows.sort(
            key=lambda row: (
                -row["dimensions"]["nominal_alpha"],
                row["dimensions"]["scheme"],
            )
        )
        summary = {
            "recipe": "fpr-calibration",
            "recipe_revision": self.revision,
            "source_artifacts": [
                artifact.identity.value for artifact in self.artifacts
            ],
            "metric_row_num": len(rows),
            "metrics": ["false_positive_rate"],
            "significance_levels": self.significance_levels,
            "confidence_level": self.confidence_level,
            "interval_method": "clopper-pearson",
            "decision_rule": "p_value < nominal_alpha",
            "population_identity": next(iter(population_ids)),
            "compatibility": {"status": "passed"},
        }
        return rows, summary
