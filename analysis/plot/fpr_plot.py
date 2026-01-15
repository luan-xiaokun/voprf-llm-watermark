from pathlib import Path

import numpy as np
from setting import *

file_names = [
    "fpr_p_value_dist_lefthash_100000.npy",
    "fpr_p_value_dist_selfhash_100000.npy",
    "fpr_p_value_dist_vow_100000.npy",
    "fpr_p_value_dist_rdf_1000.npy",
]
schemes = ["LeftHash", "SelfHash", "VOW", "RDF"]

folder_path = Path("output/detection_cost")

records = {
    scheme: np.load(folder_path / file_name)
    for scheme, file_name in zip(schemes, file_names)
}

fig, axes = plt.subplots(
    1, 4, figsize=(3.2, 1.2), sharex=True, sharey=True, layout="constrained"
)
axes[0].set_ylabel("Cumulative Density")
fig.supxlabel("Observed $p$-value", fontsize=9)
for i, (scheme, p_values) in enumerate(records.items()):
    sorted_p_values = np.sort(p_values)
    y_vals = np.arange(len(sorted_p_values)) / float(len(sorted_p_values))
    axes[i].plot(sorted_p_values, y_vals, label=scheme, linewidth=1.0)
    axes[i].plot([0, 1], [0, 1], "k--", label="Uniform", linewidth=0.8)
    axes[i].set_title(scheme, fontsize=9, fontweight="bold")
    # no upper and right spines
    axes[i].spines["top"].set_visible(False)
    axes[i].spines["right"].set_visible(False)

plt.savefig(folder_path / "fpr_p_value_distributions.pdf", dpi=300)
