from __future__ import annotations

import csv
import json
from pathlib import Path

from analysis.render_report import main as render_report
from analysis.report_io import export_csv, flattened_rows, load_report


def write_report(path: Path) -> Path:
    path.mkdir(parents=True)
    record = {
        "sample_id": "metric_1",
        "recipe": "tpr-vs-token-length",
        "metric": "true_positive_rate",
        "value": 0.5,
        "dimensions": {
            "scheme": "vow",
            "scheme_identity": "scheme_1",
            "watermark_parameters": {"gamma": 0.5},
            "population_identity": "population_1",
            "generation_model": {
                "checkpoint": "Qwen/Qwen2.5-7B",
                "revision": "commit",
            },
            "generation": {
                "model": {
                    "checkpoint": "Qwen/Qwen2.5-7B",
                    "revision": "commit",
                },
                "decoding": {"temperature": 0.7},
                "prompt_policy": {
                    "name": "c4-plain-text",
                    "revision": "c4-plain-text-v1",
                },
                "population_identity": "population_1",
            },
            "evaluation_model": {
                "checkpoint": "Qwen/Qwen2.5-14B",
                "revision": "eval-commit",
            },
            "transformation": None,
            "task": None,
            "token_num": 100,
            "target_fpr": 0.00001,
        },
        "population": {"sample_num": 2, "positive_num": 1},
        "uncertainty": {
            "method": "wilson",
            "confidence_level": 0.95,
            "lower": 0.1,
            "upper": 0.9,
        },
        "source_artifacts": ["artifact_source"],
    }
    (path / "manifest.json").write_text(
        json.dumps(
            {"artifact_schema_revision": "experiment-report-v2"}
        ),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "recipe": "tpr-vs-token-length",
                "metric_row_num": 1,
            }
        ),
        encoding="utf-8",
    )
    (path / "records.jsonl").write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )
    return path


def test_report_loader_resolves_path_or_workspace_identity(tmp_path):
    report_path = write_report(
        tmp_path
        / "workspace"
        / "artifacts"
        / "artifact_report"
    )

    direct = load_report(report_path)
    by_identity = load_report(
        "artifact_report",
        workspace=tmp_path / "workspace",
    )

    assert direct.recipe == "tpr-vs-token-length"
    assert by_identity.records == direct.records


def test_report_csv_export_flattens_scientific_dimensions(tmp_path):
    report = load_report(write_report(tmp_path / "artifact_report"))
    output = tmp_path / "report.csv"

    export_csv(report, output)

    with output.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    assert rows[0]["scheme"] == "vow"
    assert rows[0]["token_num"] == "100"
    assert rows[0]["positive_num"] == "1"
    assert "Qwen/Qwen2.5-14B" in rows[0]["evaluation_model"]
    assert rows[0]["source_artifacts"] == '["artifact_source"]'


def test_report_exports_empirical_detection_operating_point():
    row = {
        "sample_id": "metric_upv",
        "recipe": "tpr-vs-token-length",
        "metric": "true_positive_rate",
        "value": 0.5,
        "dimensions": {
            "scheme": "upv",
            "detection_operating_point": {
                "kind": "classifier-threshold",
                "score_type": "classifier_confidence",
                "decision_operator": ">",
                "decision_threshold": 0.5,
                "empirical_fpr": {
                    "positive_num": 7920,
                    "sample_num": 1_000_000,
                    "rate": 0.00792,
                },
                "calibration": {"token_num": 255},
            },
        },
        "population": {"sample_num": 2, "positive_num": 1},
    }

    flattened = flattened_rows([row])[0]

    assert flattened["target_fpr"] is None
    assert flattened["decision_threshold"] == 0.5
    assert flattened["empirical_fpr"] == 0.00792
    assert flattened["empirical_fpr_sample_num"] == 1_000_000
    assert flattened["empirical_fpr_calibration_token_num"] == 255


def test_report_renderer_consumes_artifact_without_path_discovery(tmp_path):
    report_path = write_report(tmp_path / "artifact_report")
    csv_path = tmp_path / "exports" / "report.csv"
    figure_path = tmp_path / "exports" / "report.pdf"

    exit_code = render_report(
        [
            str(report_path),
            "--csv",
            str(csv_path),
            "--figure",
            str(figure_path),
        ]
    )

    assert exit_code == 0
    assert csv_path.is_file()
    assert figure_path.stat().st_size > 0
