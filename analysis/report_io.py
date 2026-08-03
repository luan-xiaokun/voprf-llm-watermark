from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ReportArtifact:
    path: Path
    manifest: dict
    summary: dict
    records: tuple[dict, ...]

    @property
    def recipe(self) -> str:
        return str(self.summary["recipe"])


def _object(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _records(path: Path) -> tuple[dict, ...]:
    result = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            result.append(value)
    return tuple(result)


def resolve_report_path(
    value: str | Path,
    *,
    workspace: Path,
) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        return candidate.resolve()
    identity = str(value)
    if not identity.startswith("artifact_"):
        raise ValueError(
            f"{value!r} is neither an Artifact directory nor identity"
        )
    path = workspace.expanduser().resolve() / "artifacts" / identity
    if not path.is_dir():
        raise ValueError(f"unknown report Artifact {identity} in {workspace}")
    return path


def load_report(
    value: str | Path,
    *,
    workspace: Path = Path("output/experiments"),
) -> ReportArtifact:
    path = resolve_report_path(value, workspace=workspace)
    manifest = _object(path / "manifest.json")
    if manifest.get("artifact_schema_revision") != "experiment-report-v2":
        raise ValueError(
            f"{path} is not an experiment-report-v2 Artifact"
        )
    summary = _object(path / "summary.json")
    records = _records(path / "records.jsonl")
    if len(records) != summary.get("metric_row_num"):
        raise ValueError(
            f"{path} has {len(records)} records but summary declares "
            f"{summary.get('metric_row_num')}"
        )
    if any(record.get("recipe") != summary.get("recipe") for record in records):
        raise ValueError(f"{path} mixes report recipes")
    return ReportArtifact(
        path=path,
        manifest=manifest,
        summary=summary,
        records=records,
    )


def _json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def flattened_rows(records: Iterable[dict]) -> list[dict]:
    result = []
    for record in records:
        dimensions = record.get("dimensions") or {}
        population = record.get("population") or {}
        uncertainty = record.get("uncertainty") or {}
        operating_point = dimensions.get("detection_operating_point") or {}
        empirical_fpr = operating_point.get("empirical_fpr") or {}
        calibration = operating_point.get("calibration") or {}
        result.append(
            {
                "sample_id": record.get("sample_id"),
                "recipe": record.get("recipe"),
                "metric": record.get("metric"),
                "value": record.get("value"),
                "scheme": dimensions.get("scheme"),
                "scheme_identity": dimensions.get("scheme_identity"),
                "watermark_parameters": _json(
                    dimensions.get("watermark_parameters") or {}
                ),
                "population_identity": dimensions.get(
                    "population_identity"
                ),
                "generation_model": _json(
                    dimensions.get("generation_model")
                ),
                "generation": _json(dimensions.get("generation")),
                "evaluation_model": _json(
                    dimensions.get("evaluation_model")
                ),
                "transformation": _json(
                    dimensions.get("transformation")
                ),
                "task": dimensions.get("task"),
                "token_num": dimensions.get("token_num"),
                "target_fpr": dimensions.get("target_fpr"),
                "detection_operating_point": _json(operating_point),
                "score_type": operating_point.get("score_type"),
                "decision_operator": operating_point.get(
                    "decision_operator"
                ),
                "decision_threshold": operating_point.get(
                    "decision_threshold"
                ),
                "empirical_fpr": empirical_fpr.get("rate"),
                "empirical_fpr_sample_num": empirical_fpr.get(
                    "sample_num"
                ),
                "empirical_fpr_positive_num": empirical_fpr.get(
                    "positive_num"
                ),
                "empirical_fpr_calibration_token_num": calibration.get(
                    "token_num"
                ),
                "sample_num": population.get("sample_num"),
                "positive_num": population.get("positive_num"),
                "uncertainty_method": uncertainty.get("method"),
                "confidence_level": uncertainty.get("confidence_level"),
                "uncertainty_lower": uncertainty.get("lower"),
                "uncertainty_upper": uncertainty.get("upper"),
                "distribution": _json(record.get("distribution")),
                "source_artifacts": _json(
                    record.get("source_artifacts") or []
                ),
            }
        )
    return result


def export_csv(report: ReportArtifact, path: Path) -> None:
    rows = flattened_rows(report.records)
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def export_markdown(report: ReportArtifact, path: Path) -> None:
    rows = flattened_rows(report.records)
    selected = [
        {
            "scheme": row["scheme"],
            "watermark_parameters": row["watermark_parameters"],
            "task": row["task"],
            "metric": row["metric"],
            "value": row["value"],
            "token_num": row["token_num"],
            "target_fpr": row["target_fpr"],
            "decision_threshold": row["decision_threshold"],
            "empirical_fpr": row["empirical_fpr"],
            "empirical_fpr_sample_num": row[
                "empirical_fpr_sample_num"
            ],
            "empirical_fpr_calibration_token_num": row[
                "empirical_fpr_calibration_token_num"
            ],
            "sample_num": row["sample_num"],
            "positive_num": row["positive_num"],
        }
        for row in rows
    ]
    columns = list(selected[0]) if selected else []
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| "
        + " | ".join(str(row[column]) for column in columns)
        + " |"
        for row in selected
    )
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
