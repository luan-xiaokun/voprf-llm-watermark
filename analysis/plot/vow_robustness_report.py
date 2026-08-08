from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.report_io import ReportArtifact, load_report  # noqa: E402


COLORS = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#000000",
)

SCHEME_ORDER = (
    "VOW ($h=1$)",
    "VOW ($h=2$)",
    "VOW ($h=4$)",
    "LeftHash",
    "SelfHash",
    "RDF",
    "PDW",
    "UPV",
)

ATTACKS = (
    ("synonym", "Synonym Replacement"),
    ("gpt-3.5-turbo-0125", "Paraphrase (GPT-3.5 Turbo)"),
    ("gpt-5.6-luna", "Paraphrase (GPT-5.6 Luna)"),
)

METRICS = (
    ("roc_auc", "AUC", (0.45, 1.02)),
    ("true_positive_rate", "True Positive Rate", (0.0, 1.02)),
)

STYLE = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7,
    "axes.labelsize": 7,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
}


def _scheme_label(row: dict[str, Any]) -> str:
    dimensions = row.get("dimensions") or {}
    method = dimensions.get("scheme")
    if method == "vow":
        parameters = dimensions.get("watermark_parameters") or {}
        window_size = parameters.get("window_size")
        if window_size not in {1, 2, 4}:
            raise ValueError(
                "robustness report contains an unsupported VOW window: "
                f"{window_size!r}"
            )
        return f"VOW ($h={window_size}$)"
    labels = {
        "lefthash": "LeftHash",
        "selfhash": "SelfHash",
        "rdf": "RDF",
        "pdw": "PDW",
        "upv": "UPV",
    }
    try:
        return labels[str(method)]
    except KeyError as error:
        raise ValueError(
            f"robustness report contains an unsupported scheme: {method!r}"
        ) from error


def _attack_key(row: dict[str, Any]) -> str:
    dimensions = row.get("dimensions") or {}
    transformation = dimensions.get("transformation") or {}
    method = transformation.get("method")
    if method == "masked-lm-replacement":
        return "synonym"
    if method == "openai-paraphrase":
        parameters = transformation.get("parameters") or {}
        model = parameters.get("model")
        if model in {attack for attack, _ in ATTACKS}:
            return str(model)
        raise ValueError(
            "robustness report contains an unsupported paraphrase model: "
            f"{model!r}"
        )
    raise ValueError(
        "robustness report contains an unsupported transformation: "
        f"{method!r}"
    )


def robustness_values(
    report: ReportArtifact,
) -> dict[tuple[str, str, str], float]:
    if report.recipe != "robustness":
        raise ValueError(
            f"expected robustness report, found {report.recipe!r}"
        )
    selected_metrics = {metric for metric, _, _ in METRICS}
    values: dict[tuple[str, str, str], float] = {}
    for row in report.records:
        metric = row.get("metric")
        if metric not in selected_metrics:
            continue
        key = (_scheme_label(row), _attack_key(row), str(metric))
        if key in values:
            raise ValueError(f"robustness report contains duplicate row {key}")
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"robustness row {key} has invalid value {value!r}")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"robustness row {key} is outside [0, 1]: {value}")
        values[key] = value

    expected = {
        (scheme, attack, metric)
        for scheme in SCHEME_ORDER
        for attack, _ in ATTACKS
        for metric, _, _ in METRICS
    }
    missing = sorted(expected - set(values))
    unexpected = sorted(set(values) - expected)
    if missing or unexpected:
        raise ValueError(
            "robustness report does not match the expected 8×3×2 matrix; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return values


def _style_subplot_frame(axis: Any) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
    axis.tick_params(width=0.4, length=2.5)


def render_robustness(report: ReportArtifact, output: Path) -> None:
    values = robustness_values(report)
    with mpl.rc_context(STYLE):
        from matplotlib import pyplot as plt

        figure, axes = plt.subplots(
            2,
            3,
            figsize=(6.8, 2.4),
            sharex=True,
            sharey="row",
            layout="constrained",
        )
        x_positions = tuple(range(len(SCHEME_ORDER)))
        for column, (attack, attack_label) in enumerate(ATTACKS):
            axes[0][column].set_title(
                attack_label,
                fontsize=6,
                fontweight="bold",
            )
            for row_index, (metric, ylabel, limits) in enumerate(METRICS):
                axis = axes[row_index][column]
                heights = [
                    values[(scheme, attack, metric)]
                    for scheme in SCHEME_ORDER
                ]
                axis.bar(
                    x_positions,
                    heights,
                    width=0.78,
                    color=COLORS,
                    edgecolor="black",
                    linewidth=0.3,
                    zorder=3,
                )
                axis.set_ylim(*limits)
                axis.set_xlim(-0.65, len(SCHEME_ORDER) - 0.35)
                axis.set_xticks([])
                axis.grid(
                    axis="y",
                    which="major",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=0.5,
                    zorder=0,
                )
                _style_subplot_frame(axis)
                if column == 0:
                    axis.set_ylabel(ylabel)

        handles = [axes[0][0].patches[index] for index in x_positions]
        figure.suptitle(" ")
        figure.legend(
            handles,
            SCHEME_ORDER,
            loc="upper center",
            ncol=8,
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            columnspacing=0.8,
            handlelength=1.1,
            handletextpad=0.35,
        )
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render the aggregate AUC and TPR matrix from a USENIX "
            "robustness experiment-report-v2 Artifact."
        )
    )
    parser.add_argument(
        "artifact",
        help="Report Artifact identity or Artifact directory",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("output/experiments"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = load_report(args.artifact, workspace=args.workspace)
    render_robustness(report, args.output)
    print(
        json.dumps(
            {
                "artifact": str(report.path),
                "output": str(args.output.expanduser().resolve()),
                "recipe": report.recipe,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
