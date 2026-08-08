from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.plot.vow_robustness_report import (
    ATTACKS,
    SCHEME_ORDER,
    main,
    robustness_curves,
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
                for token_num in (20, 40, 60):
                    records.append(
                        {
                            "sample_id": (
                                f"metric_{scheme_index}_{attack_index}_{metric}_"
                                f"{token_num}"
                            ),
                            "recipe": "robustness",
                            "metric": metric,
                            "value": min(
                                1.0,
                                base
                                + 0.02 * scheme_index
                                + 0.01 * attack_index
                                + 0.0005 * token_num,
                            ),
                            "dimensions": {
                                **_scheme_dimensions(scheme),
                                "transformation": _transformation(attack),
                                "token_num": token_num,
                            },
                            "population": {"sample_num": 500},
                            "source_artifacts": ["artifact_source"],
                        }
                    )
    records.append(
        {
            "sample_id": "sparse_rdf_endpoint",
            "recipe": "robustness",
            "metric": "roc_auc",
            "value": 0.99,
            "dimensions": {
                **_scheme_dimensions("RDF"),
                "transformation": _transformation("synonym"),
                "token_num": 57,
            },
            "population": {"sample_num": 10},
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


def test_robustness_plot_requires_complete_token_curves(tmp_path):
    report = load_report(_write_report(tmp_path / "artifact_report"))

    curves = robustness_curves(report)

    assert len(curves) == 42
    assert curves[
        ("VOW ($h=4$)", "gpt-5.6-luna", "roc_auc")
    ][-1].value == pytest.approx(0.59)
    assert [
        point.token_num
        for point in curves[("RDF", "synonym", "roc_auc")]
    ] == [20, 40, 60]


def test_robustness_plot_renders_lines_not_bars(tmp_path, monkeypatch):
    from matplotlib.axes import Axes

    def reject_bar(*args, **kwargs):
        del args, kwargs
        raise AssertionError("robustness curves must not use bars")

    monkeypatch.setattr(Axes, "bar", reject_bar)
    report_path = _write_report(tmp_path / "artifact_report")
    output = tmp_path / "figures" / "robustness.pdf"

    exit_code = main([str(report_path), "--output", str(output)])

    assert exit_code == 0
    assert output.stat().st_size > 0
