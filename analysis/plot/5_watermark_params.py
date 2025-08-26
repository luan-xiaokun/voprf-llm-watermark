import itertools
from collections import defaultdict

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

from setting import *

plt.rcParams["figure.figsize"] = (6.6, 3.0)

record_file_path = "data/plot_data/ppl_against_tpr.md"
gammas = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
deltas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
window_sizes = [4]
top_ks = ["--", 50]

shapes = ["o", "s", "d", "v", "^", "p", "P"]

delta_colors = plt.cm.viridis(np.linspace(0, 1, len(deltas)))
delta_cmap = mcolors.ListedColormap(delta_colors)


def plot_perplexity_against_tpr(
    baselines: dict, all_records: dict, window_size: int, top_k: int | str = "--"
):
    baseline_ppl = baselines[top_k]

    fig, ax = plt.subplots(layout="constrained")
    ax.axvline(x=baseline_ppl, color="r", linestyle="--", label="Baseline PPL")

    params = list(itertools.product(gammas, deltas))
    for gamma, delta in params:
        gamma_idx = gammas.index(gamma)
        delta_idx = deltas.index(delta)
        records = all_records[top_k][(gamma, delta, window_size)]
        perplexity = records["perplexity"]
        tpr_values = records["tpr_values"]
        tpr = tpr_values[1]

        ax.scatter(
            perplexity,
            tpr,
            color=delta_cmap(delta_idx),
            marker=shapes[gamma_idx],
            s=30,
            edgecolors="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Perplexity (better $\\rightarrow$)")
    ax.set_ylabel("TPR @ $10^{-5}$ FPR")

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=shapes[i],
            color="w",
            label=f"γ: {gammas[i]:.3f}",
            markerfacecolor="gray",
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        for i in range(len(gammas))
    ]

    legend1 = ax.legend(
        handles=legend_elements,
        title="Gamma",
        loc="lower left",
        bbox_to_anchor=(0.01, 0.01),
        frameon=True,
        framealpha=0.9,
    )
    ax.add_artist(legend1)

    sm = ScalarMappable(
        cmap=delta_cmap, norm=mcolors.Normalize(vmin=0, vmax=len(deltas) - 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.05, ticks=range(len(deltas)))
    cbar.set_ticklabels([f"{d:.1f}" for d in deltas])
    cbar.set_label("Delta")

    if top_k == "--":
        ax.set_xlim(5.0, 8.25)
    else:
        ax.set_xlim(3.5, 6.75)
    ax.invert_xaxis()

    ax.grid(True, linestyle="--", alpha=0.7, linewidth=0.5)
    ax.set_axisbelow(True)
    # plt.tight_layout()

    fig_name = "ppl_vs_tpr"
    if top_k != "--":
        fig_name += f"_top_{top_k}"
    fig_name += f"_w{window_size}.pdf"
    plt.savefig(f"figures/appendix/{fig_name}", dpi=300)
    plt.close()


def main():
    all_records = defaultdict(dict)
    baselines = {}
    with open(record_file_path, "r") as file:
        for line in file.readlines():
            if not line.startswith("|"):
                continue

            components = [c.strip() for c in line.strip("|").split("|") if c.strip()]
            gamma, delta, window_size = components[:3]
            top_k, top_p = components[3:5]
            tpr_values = components[5:9]
            perplexity = components[9]
            if gamma in ["gamma", "-----"]:
                continue

            if gamma == delta == window_size == "--":
                top_k = int(top_k) if top_k != "--" else "--"
                baselines[top_k] = float(perplexity)
                continue

            gamma = float(gamma)
            delta = float(delta)
            window_size = int(window_size)
            top_k = int(top_k) if top_k != "--" else "--"
            all_records[top_k][(gamma, delta, window_size)] = {
                "tpr_values": [float(t[:-1]) for t in tpr_values],
                "perplexity": float(perplexity),
            }

    for top_k in top_ks:
        for window_size in window_sizes:
            plot_perplexity_against_tpr(baselines, all_records, window_size, top_k)


if __name__ == "__main__":
    main()
