import json
from pathlib import Path

import matplotlib.patches as mpatches
import numpy as np
from setting import *

# def adjacent_values(vals, q1, q3):
#     upper_adjacent_value = q3 + (q3 - q1) * 1.5
#     upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

#     lower_adjacent_value = q1 - (q3 - q1) * 1.5
#     lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
#     return lower_adjacent_value, upper_adjacent_value


record_file = Path("data/plot_data/robustness_eval_records_all.json")
records = json.loads(record_file.read_text())

scheme_map = {
    "lefthash": "LeftHash",
    "selfhash": "SelfHash",
    "vow": "VOW\n($h=1$)",
    "rdf": "RDF",
    "vow-w2": "VOW\n($h=2$)",
    "vow-w4": "VOW\n($h=4$)",
}
label_map = {
    "negative": "Non-watermarked",
    "paraphrased-gpt-5.1": "Paraphrase (GPT-5.1)",
}


scheme_names = ["vow", "vow-w2", "vow-w4", "lefthash", "selfhash"]
scheme_names.append("rdf")

# fig, ax = plt.subplots(figsize=(3.0, 2.0), layout="constrained")
# for i, scheme_name in enumerate(scheme_names):
#     negative_pvalues = records[scheme_name]["negative"]
#     attack_pvalues = records[scheme_name]["paraphrased-gpt-5.1"]

#     obj1 = ax.violinplot(
#         negative_pvalues,
#         [i],
#         showextrema=False,
#         showmedians=True,
#         side="low",
#         quantiles=[0.25, 0.75],
#     )
#     print(obj1["cmedians"])
#     print(obj1["cquantiles"])
#     obj2 = ax.violinplot(
#         attack_pvalues,
#         [i],
#         showextrema=False,
#         showmedians=True,
#         side="high",
#         quantiles=[0.25, 0.75],
#     )

# ax.set_xticks(range(len(scheme_names)), [scheme_map[s] for s in scheme_names])

# plt.savefig("violin_plot.pdf", dpi=300)


def apply_custom_style(parts, facecolor, median_ratio=0.7, quantile_ratio=0.7):
    for body in parts["bodies"]:
        body.set_facecolor(facecolor)
        body.set_edgecolor("black")
        body.set_linewidth(0.5)
        body.set_alpha(0.7)

    def shorten_segments(line_collection, ratio):
        if line_collection is None:
            return
        segs = line_collection.get_segments()
        new_segs = []
        for seg in segs:
            x_start, x_end = seg[0, 0], seg[1, 0]
            y = seg[0, 1]

            center = (x_start + x_end) / 2
            half_width = (x_end - x_start) / 2 * ratio

            new_segs.append([[center - half_width, y], [center + half_width, y]])

        line_collection.set_segments(new_segs)

    if "cmedians" in parts:
        parts["cmedians"].set_edgecolor("white")
        parts["cmedians"].set_linewidth(1.5)
        parts["cmedians"].set_alpha(1.0)
        shorten_segments(parts["cmedians"], median_ratio)

    if "cquantiles" in parts:
        parts["cquantiles"].set_edgecolor("black")
        parts["cquantiles"].set_linestyle("--")
        parts["cquantiles"].set_linewidth(0.8)
        parts["cquantiles"].set_alpha(0.6)
        shorten_segments(parts["cquantiles"], quantile_ratio)


fig, ax = plt.subplots(figsize=(3.6, 1.8), layout="constrained")

color_neg = "#6495ED"  # CornflowerBlue
color_att = "#FA8072"  # Salmon

for i, scheme_name in enumerate(scheme_names):
    negative_pvalues = records[scheme_name]["negative"]
    attack_pvalues = records[scheme_name]["paraphrased-gpt-5.1"]

    v1 = ax.violinplot(
        negative_pvalues,
        positions=[i],
        showextrema=False,
        showmedians=True,
        quantiles=[0.25, 0.75],
        side="low",
    )
    apply_custom_style(v1, color_neg)

    v2 = ax.violinplot(
        attack_pvalues,
        positions=[i],
        showextrema=False,
        showmedians=True,
        quantiles=[0.25, 0.75],
        side="high",
    )
    apply_custom_style(v2, color_att)

ax.set_xticks(range(len(scheme_names)))
ax.set_xticklabels([scheme_map[s] for s in scheme_names], fontsize=7)
ax.set_ylabel("Detection $p$-value")
ax.yaxis.grid(True, linestyle=":", alpha=0.3)

legend_patches = [
    mpatches.Patch(color=color_neg, label="Non-watermarked", alpha=0.7),
    mpatches.Patch(color=color_att, label="Paraphrased (GPT-5.1)", alpha=0.7),
]
fig.suptitle(" ")
ax.legend(
    handles=legend_patches,
    loc="upper center",
    ncol=2,
    fontsize=8,
    bbox_to_anchor=(0.5, 1.2),
    edgecolor="white",
    facecolor="white",
    framealpha=0.8,
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)


plt.savefig("violin_plot.pdf", dpi=300)
plt.show()
