from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib as mpl


REPOSITORY = Path(__file__).resolve().parents[2]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from analysis.report_io import ReportArtifact, load_report  # noqa: E402


class CurvePoint(NamedTuple):
    token_num: int
    value: float
    sample_num: int


SCHEME_ORDER = (
    "VOW ($h=1$)",
    "VOW ($h=2$)",
    "VOW ($h=4$)",
    "LeftHash",
    "SelfHash",
    "RDF",
    "UPV",
)

SCHEME_STYLES = {
    "VOW ($h=1$)": ("#E69F00", "-"),
    "VOW ($h=2$)": ("#56B4E9", (0, (5, 1.5))),
    "VOW ($h=4$)": ("#009E73", (0, (3, 1.2, 1, 1.2))),
    "LeftHash": ("#F0E442", (0, (7, 2))),
    "SelfHash": ("#0072B2", (0, (3, 1.5, 1.2, 1.5))),
    "RDF": ("#D55E00", "--"),
    "UPV": ("#CC79A7", (0, (1.5, 1.5))),
}

ATTACKS = (
    ("synonym", "Synonym Replacement"),
    ("gpt-3.5-turbo-0125", "Paraphrase (GPT-3.5 Turbo)"),
    ("gpt-5.6-luna", "Paraphrase (GPT-5.6 Luna)"),
)

METRICS = (
    ("roc_auc", "AUC", (0.38, 1.02), 0.5),
    ("true_positive_rate", "True Positive Rate", (0.0, 1.02), 0.0),
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
        "upv": "UPV",
    }
    try:
        return labels[str(method)]
    except KeyError as error:
        raise ValueError(
            "robustness token curve contains an unsupported scheme: "
            f"{method!r}"
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


def robustness_curves(
    report: ReportArtifact,
    *,
    min_eligible_fraction: float = 0.2,
) -> dict[tuple[str, str, str], tuple[CurvePoint, ...]]:
    if report.recipe != "robustness":
        raise ValueError(
            f"expected robustness report, found {report.recipe!r}"
        )
    if not 0.0 <= min_eligible_fraction <= 1.0:
        raise ValueError("min_eligible_fraction must be between zero and one")

    selected_metrics = {metric for metric, _, _, _ in METRICS}
    grouped: dict[tuple[str, str, str], list[CurvePoint]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    for row in report.records:
        metric = row.get("metric")
        if metric not in selected_metrics:
            continue
        dimensions = row.get("dimensions") or {}
        token_num = dimensions.get("token_num")
        if token_num is None:
            continue
        if (
            not isinstance(token_num, int)
            or isinstance(token_num, bool)
            or token_num <= 0
        ):
            raise ValueError(
                f"robustness curve row has invalid token_num {token_num!r}"
            )
        key = (_scheme_label(row), _attack_key(row), str(metric))
        point_key = (*key, token_num)
        if point_key in seen:
            raise ValueError(
                f"robustness report contains duplicate curve point {point_key}"
            )
        seen.add(point_key)
        value = row.get("value")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"robustness curve point {point_key} has invalid value "
                f"{value!r}"
            )
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"robustness curve point {point_key} is outside [0, 1]: "
                f"{value}"
            )
        population = row.get("population") or {}
        sample_num = population.get("sample_num")
        if (
            not isinstance(sample_num, int)
            or isinstance(sample_num, bool)
            or sample_num <= 0
        ):
            raise ValueError(
                f"robustness curve point {point_key} has invalid population"
            )
        grouped.setdefault(key, []).append(
            CurvePoint(token_num, value, sample_num)
        )

    expected = {
        (scheme, attack, metric)
        for scheme in SCHEME_ORDER
        for attack, _ in ATTACKS
        for metric, _, _, _ in METRICS
    }
    missing = sorted(expected - set(grouped))
    unexpected = sorted(set(grouped) - expected)
    if missing or unexpected:
        raise ValueError(
            "robustness report does not contain the expected 7×3×2 token "
            f"curves; missing={missing}, unexpected={unexpected}"
        )

    curves = {}
    for key, points in grouped.items():
        maximum_population = max(point.sample_num for point in points)
        minimum_population = maximum_population * min_eligible_fraction
        eligible = tuple(
            sorted(
                (
                    point
                    for point in points
                    if point.sample_num >= minimum_population
                ),
                key=lambda point: point.token_num,
            )
        )
        if not eligible:
            raise ValueError(
                f"robustness curve {key} has no sufficiently populated points"
            )
        curves[key] = eligible
    return curves


def _style_subplot_frame(axis: Any) -> None:
    for spine in axis.spines.values():
        spine.set_linewidth(0.4)
    axis.tick_params(width=0.4, length=2.5)


def render_robustness(
    report: ReportArtifact,
    output: Path,
    *,
    min_eligible_fraction: float = 0.2,
) -> None:
    curves = robustness_curves(
        report,
        min_eligible_fraction=min_eligible_fraction,
    )
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
        maximum_token_num = max(
            point.token_num
            for points in curves.values()
            for point in points
        )
        for column, (attack, attack_label) in enumerate(ATTACKS):
            axes[0][column].set_title(
                attack_label,
                fontsize=6,
                fontweight="bold",
            )
            for row_index, (
                metric,
                ylabel,
                limits,
                origin,
            ) in enumerate(METRICS):
                axis = axes[row_index][column]
                for scheme in SCHEME_ORDER:
                    points = curves[(scheme, attack, metric)]
                    color, linestyle = SCHEME_STYLES[scheme]
                    axis.plot(
                        [0, *(point.token_num for point in points)],
                        [origin, *(point.value for point in points)],
                        label=scheme,
                        color=color,
                        linestyle=linestyle,
                        linewidth=0.75,
                    )
                axis.set_ylim(*limits)
                axis.set_xlim(0, maximum_token_num)
                axis.grid(
                    which="major",
                    linestyle=":",
                    alpha=0.5,
                    linewidth=0.5,
                )
                _style_subplot_frame(axis)
                if column == 0:
                    axis.set_ylabel(ylabel)

        handles, labels = axes[0][0].get_legend_handles_labels()
        figure.suptitle(" ")
        figure.supxlabel("Number of Tokens", fontsize=7)
        figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=7,
            bbox_to_anchor=(0.5, 1.0),
            frameon=False,
            columnspacing=0.8,
            handlelength=1.7,
            handletextpad=0.35,
        )
        destination = output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300)
        plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render token-level AUC and TPR robustness curves from a USENIX "
            "experiment-report-v2 Artifact."
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
    parser.add_argument(
        "--min-eligible-fraction",
        type=float,
        default=0.2,
        help=(
            "Drop sparse endpoint milestones below this fraction of each "
            "curve's largest eligible population (default: 0.2)."
        ),
    )
    args = parser.parse_args(argv)

    report = load_report(args.artifact, workspace=args.workspace)
    render_robustness(
        report,
        args.output,
        min_eligible_fraction=args.min_eligible_fraction,
    )
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
