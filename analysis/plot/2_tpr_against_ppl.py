from matplotlib.lines import Line2D
from setting import *

colors = color_palettes[COLOR_CYCLE]
arrowprops = dict(
    facecolor="black",
    shrink=0.2,
    width=0.1,
    headwidth=2,
    headlength=2,
)

no_watermark_ppl = 5.10
# (gamma, delta): [TPR, PPL]
vow_records = {
    # (0.375, 2.5): (99.8, 6.70),
    (0.5, 2.0): (96.6, 5.85),
    (0.5, 2.5): (99.8, 6.49),
    (0.5, 3.0): (99.8, 6.83),
    (0.625, 2.5): (98.2, 5.94),
}
baseline_records = {
    "LeftHash": (98.8, 6.51),
    "SelfHash": (98.2, 6.18),
}


def main():
    fig, ax = plt.subplots(figsize=(3.2, 1.8), layout="constrained")

    texts = []
    for (gamma, delta), (tpr, ppl) in vow_records.items():
        txt = f"$\\gamma={gamma}$\n$\\delta={delta}$"
        texts.append(txt)
        ax.scatter(
            ppl,
            tpr,
            color=colors[0],
            marker="o",
            s=30,  # 减小点大小适应双栏
            edgecolors="black",
            linewidth=0.5,
            label=f"Vow",
        )
        # texts.append(ax.text(ppl, tpr, txt, fontsize=6))
        # texts.append(ax.annotate(txt, (ppl, tpr), fontsize=6))
    # gamma, delta = 0.375, 2.5
    # tpr, ppl = vow_records[(gamma, delta)]
    # ax.text(ppl + 0.06, tpr - 0.55, f"$\\gamma={gamma}$\n$\\delta={delta}$", fontsize=7)
    gamma, delta = 0.5, 2.0
    tpr, ppl = vow_records[(gamma, delta)]
    ax.text(ppl - 0.06, tpr - 0.2, f"$\\gamma={gamma}$\n$\\delta={delta}$", fontsize=7)
    gamma, delta = 0.5, 2.5
    tpr, ppl = vow_records[(gamma, delta)]
    ax.text(ppl - 0.06, tpr - 0.42, f"$\\gamma={gamma}$\n$\\delta={delta}$", fontsize=7)
    gamma, delta = 0.5, 3.0
    tpr, ppl = vow_records[(gamma, delta)]
    ax.text(ppl + 0.05, tpr - 0.7, f"$\\gamma={gamma}$\n$\\delta={delta}$", fontsize=7)
    gamma, delta = 0.625, 2.5
    tpr, ppl = vow_records[(gamma, delta)]
    ax.text(ppl + 0.08, tpr - 0.7, f"$\\gamma={gamma}$\n$\\delta={delta}$", fontsize=7)

    for i, (baseline, (tpr, ppl)) in enumerate(baseline_records.items()):
        ax.scatter(
            ppl,
            tpr,
            color=colors[i + 1],
            marker="s",
            s=30,
            edgecolors="black",
            linewidth=0.5,
            label=f"{baseline}",
        )

    ax.axvline(no_watermark_ppl, color="r", linestyle="--", label="Baseline PPL")

    ax.set_xlabel("Perplexity (better $\\rightarrow$)")
    ax.set_ylabel("True positive rate")

    ax.invert_xaxis()

    ax.set_ylim(96.3, 100.0)

    shapes = ["o"] + ["s"] * len(baseline_records)
    labels = ["VOW", "LeftHash", "SelfHash"]
    label_colors = [colors[0], colors[1], colors[2]]
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=shapes[i],
            color="w",
            label=labels[i],
            markerfacecolor=label_colors[i],
            markersize=6,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        for i in range(len(labels))
    ]
    legend = ax.legend(
        handles=legend_elements,
        loc="upper right",
        # fontsize=8,
        bbox_to_anchor=(0.96, 0.99),
        # frameon=True,
        # framealpha=0.9,
    )
    ax.add_artist(legend)
    ax.grid(which="major", linestyle=":", alpha=0.7)

    # plt.tight_layout()

    plt.savefig("figures/evaluation/tpr_against_perplexity.pdf", dpi=300)


if __name__ == "__main__":
    main()
