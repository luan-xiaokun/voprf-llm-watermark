from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.plot.vow_robustness_report import (
    ATTACKS,
    SCHEME_ORDER,
    main,
    robustness_values,
)
from analysis.report_io import load_report


def _scheme_dimensions(label: str) -> dict:
    if label.startswith("VOW"):
        return {
            "scheme": "vow",
            "watermark_parameters": {
                "window_size": int(label.split("=")[1].split("$")[0])
            },
        }
    return {
        "scheme": {
            "LeftHash": "lefthash",
            "SelfHash": "selfhash",
            "RDF": "rdf",
            "PDW": "pdw",
            "UPV": "upv",
        }[label],
        "watermark_parameters": {},
    }


def _transformation(attack: str) -> dict:
    if attack == "synonym":
        return {
            "method": "masked-lm-replacement",
            "parameters": {"replacement_rate": 0.3, "top_k": 15},
        }
    return {
        "method": "openai-paraphrase",
        "parameters": {"model": attack},
    }


def _write_report(path: Path) -> Path:
    path.mkdir(parents=True)
    records = []
    for scheme_index, scheme in enumerate(SCHEME_ORDER):
        for attack_index, (attack, _) in enumerate(ATTACKS):
            for metric, base in (
                ("roc_auc", 0.5),
                ("true_positive_rate", 0.0),
            ):
                records.append(
                    {
                        "sample_id": (
                            f"metric_{scheme_index}_{attack_index}_{metric}"
                        ),
                        "recipe": "robustness",
                        "metric": metric,
                        "value": min(
                            1.0,
                            base + 0.03 * scheme_index + 0.01 * attack_index,
                        ),
                        "dimensions": {
                            **_scheme_dimensions(scheme),
                            "transformation": _transformation(attack),
                        },
                        "population": {"sample_num": 500},
                        "source_artifacts": ["artifact_source"],
                    }
                )
    (path / "manifest.json").write_text(
        json.dumps({"artifact_schema_revision": "experiment-report-v2"}),
        encoding="utf-8",
    )
    (path / "summary.json").write_text(
        json.dumps(
            {
                "recipe": "robustness",
                "metric_row_num": len(records),
            }
        ),
        encoding="utf-8",
    )
    (path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_robustness_plot_requires_the_complete_scheme_attack_matrix(tmp_path):
    report = load_report(_write_report(tmp_path / "artifact_report"))

    values = robustness_values(report)

    assert len(values) == 48
    assert values[
        ("VOW ($h=4$)", "gpt-5.6-luna", "roc_auc")
    ] == pytest.approx(0.58)


def test_robustness_plot_renders_report_artifact(tmp_path):
    report_path = _write_report(tmp_path / "artifact_report")
    output = tmp_path / "figures" / "robustness.pdf"

    exit_code = main([str(report_path), "--output", str(output)])

    assert exit_code == 0
    assert output.stat().st_size > 0
