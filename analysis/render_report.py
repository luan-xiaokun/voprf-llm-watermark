from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable

try:
    from .report_io import (
        ReportArtifact,
        export_csv,
        export_markdown,
        load_report,
    )
except ImportError:
    from report_io import (
        ReportArtifact,
        export_csv,
        export_markdown,
        load_report,
    )


def _scheme_label(row: dict) -> str:
    dimensions = row["dimensions"]
    method = dimensions.get("scheme") or "unknown"
    parameters = dimensions.get("watermark_parameters") or {}
    if method == "vow":
        return (
            f"VOW h={parameters.get('window_size')}, "
            f"γ={parameters.get('gamma')}, δ={parameters.get('delta')}"
        )
    return str(method).replace("hash", "Hash").upper() if method == "rdf" else (
        str(method).replace("lefthash", "LeftHash")
        .replace("selfhash", "SelfHash")
        .replace("upv", "UPV")
        .replace("pdw", "PDW")
    )


def _pyplot():
    import matplotlib.pyplot as plt

    return plt


def _render_tpr_token(report: ReportArtifact, output: Path) -> None:
    plt = _pyplot()
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in report.records:
        if row["metric"] == "true_positive_rate":
            grouped[_scheme_label(row)].append(row)
    fig, ax = plt.subplots(figsize=(3.2, 1.8), layout="constrained")
    for label, rows in sorted(grouped.items()):
        rows.sort(key=lambda row: row["dimensions"]["token_num"])
        ax.plot(
            [row["dimensions"]["token_num"] for row in rows],
            [row["value"] for row in rows],
            label=label,
            linewidth=1.0,
        )
    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("True Positive Rate")
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _render_tpr_ppl(report: ReportArtifact, output: Path) -> None:
    plt = _pyplot()
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in report.records:
        grouped[row["dimensions"]["scheme_identity"]][row["metric"]] = row
    fig, ax = plt.subplots(figsize=(3.2, 1.8), layout="constrained")
    for metrics in grouped.values():
        ppl = metrics.get("conditional_perplexity")
        tpr = metrics.get("true_positive_rate")
        if ppl is None:
            continue
        if tpr is None:
            ax.axvline(
                ppl["value"],
                color="r",
                linestyle="--",
                label="Baseline PPL",
            )
            continue
        ax.scatter(
            ppl["value"],
            tpr["value"],
            label=_scheme_label(tpr),
            s=30,
        )
    ax.set_xlabel("Perplexity (better →)")
    ax.set_ylabel("True Positive Rate")
    ax.invert_xaxis()
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _transformation_label(row: dict) -> str:
    transformation = row["dimensions"].get("transformation")
    if transformation is None:
        return "Clean"
    parameters = transformation.get("parameters") or {}
    method = transformation.get("method")
    if method == "openai-paraphrase":
        return f"Paraphrase ({parameters.get('model')})"
    if method == "word-deletion":
        return f"Word deletion ({parameters.get('rate')})"
    return str(method)


def _render_robustness(report: ReportArtifact, output: Path) -> None:
    plt = _pyplot()
    detection = [
        row for row in report.records if row["metric"] == "detection_rate"
    ]
    transformations = sorted({_transformation_label(row) for row in detection})
    fig, axes = plt.subplots(
        1,
        len(transformations),
        figsize=(max(3.2, 2.4 * len(transformations)), 2.0),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for axis, transformation in zip(axes[0], transformations):
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in detection:
            if _transformation_label(row) == transformation:
                grouped[_scheme_label(row)].append(row)
        for label, rows in sorted(grouped.items()):
            rows.sort(key=lambda row: row["dimensions"]["token_num"])
            axis.plot(
                [row["dimensions"]["token_num"] for row in rows],
                [row["value"] for row in rows],
                label=label,
                linewidth=1.0,
            )
        axis.set_title(transformation)
        axis.set_xlabel("Number of Tokens")
        axis.grid(which="major", linestyle=":", alpha=0.7)
    axes[0][0].set_ylabel("Detection Rate")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _render_adaptive_forgery(
    report: ReportArtifact,
    output: Path,
) -> None:
    plt = _pyplot()
    query_rows = [
        row
        for row in report.records
        if row["metric"] == "mean_oracle_query_count"
    ]
    asr_rows = [
        row
        for row in report.records
        if row["metric"] == "attack_success_rate"
        and row["dimensions"]["target_fpr"] == 0.00001
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(6.4, 2.2), layout="constrained"
    )
    query_rows.sort(key=lambda row: row["dimensions"]["token_num"])
    asr_rows.sort(key=lambda row: row["dimensions"]["token_num"])
    axes[0].plot(
        [row["dimensions"]["token_num"] for row in query_rows],
        [row["value"] for row in query_rows],
    )
    axes[1].plot(
        [row["dimensions"]["token_num"] for row in asr_rows],
        [row["value"] for row in asr_rows],
    )
    axes[0].set_ylabel("Mean Oracle Queries")
    axes[1].set_ylabel("Attack Success Rate")
    for axis in axes:
        axis.set_xlabel("Number of Tokens")
        axis.grid(which="major", linestyle=":", alpha=0.7)
    fig.savefig(output, dpi=300)
    plt.close(fig)


_RENDERERS: dict[str, Callable[[ReportArtifact, Path], None]] = {
    "tpr-vs-token-length": _render_tpr_token,
    "tpr-vs-ppl": _render_tpr_ppl,
    "robustness": _render_robustness,
    "adaptive-forgery": _render_adaptive_forgery,
}


def main(
    argv: list[str] | None = None,
    *,
    expected_recipe: str | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        description="Render or export an experiment-report-v1 Artifact."
    )
    parser.add_argument("artifact")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("output/experiments"),
    )
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--figure", type=Path)
    parser.add_argument(
        "--include-pdw",
        action="store_true",
        help="Include PDW in the TPR/PPL figure; report exports always retain it.",
    )
    args = parser.parse_args(argv)
    if not any((args.csv, args.markdown, args.figure)):
        parser.error("at least one of --csv, --markdown, or --figure is required")
    report = load_report(args.artifact, workspace=args.workspace)
    if expected_recipe is not None and report.recipe != expected_recipe:
        parser.error(
            f"expected recipe {expected_recipe!r}, found {report.recipe!r}"
        )
    if args.csv is not None:
        export_csv(report, args.csv)
    if args.markdown is not None:
        export_markdown(report, args.markdown)
    if args.figure is not None:
        renderer = _RENDERERS.get(report.recipe)
        if renderer is None:
            parser.error(
                f"recipe {report.recipe!r} has no figure renderer; "
                "use --csv or --markdown"
            )
        figure_report = report
        if report.recipe == "tpr-vs-ppl" and not args.include_pdw:
            figure_report = replace(
                report,
                records=tuple(
                    row
                    for row in report.records
                    if row["dimensions"].get("scheme") != "pdw"
                ),
            )
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        renderer(figure_report, args.figure)
    print(
        json.dumps(
            {
                "artifact": str(report.path),
                "recipe": report.recipe,
                "metric_rows": len(report.records),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
