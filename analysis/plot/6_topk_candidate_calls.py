from setting import *

import numpy as np
from pathlib import Path
from scipy import stats

# Load data
data_path = (
    Path(__file__).parent.parent.parent
    / "data"
    / "plot_data"
    / "adaptive_sampling_topk_sweep.npz"
)
data = np.load(data_path)

delta_values = data["delta_values"]
top_k_values = data["top_k_values"]
candidate_means = data["candidate_means"]
candidate_stds = data["candidate_stds"]
calls_means = data["calls_means"]
calls_stds = data["calls_stds"]

# Create figure: 2 rows, 2 columns with shared axes
fig, axes = plt.subplots(
    2, 2, figsize=(3.2, 2.4), layout="constrained", sharey="row", sharex=True
)
axes = axes.flatten()  # Flatten to 1D array for easier iteration

colors = color_palettes[COLOR_CYCLE]

print("Linear fit slopes (count per top-k unit):")
print("-" * 50)

for idx, (ax, delta) in enumerate(zip(axes, delta_values)):
    # Plot candidate size with lighter error bar
    ax.errorbar(
        top_k_values,
        candidate_means[idx],
        yerr=candidate_stds[idx],
        label="Candidate Set Size",
        color=colors[0],
        ecolor=mpl.colors.to_rgba(colors[0], 0.4),
        marker="o",
        markersize=1,
        linewidth=0.9,
        capsize=1.0,
        capthick=0.5,
    )
    # Plot num calls with lighter error bar
    ax.errorbar(
        top_k_values,
        calls_means[idx],
        yerr=calls_stds[idx],
        label="Number of Iterated Tokens",
        color=colors[1],
        ecolor=mpl.colors.to_rgba(colors[1], 0.4),
        marker="s",
        markersize=1,
        linewidth=0.9,
        capsize=1.0,
        capthick=0.5,
    )

    # Linear fit for candidate size
    slope_cand, intercept_cand, _, _, _ = stats.linregress(
        top_k_values, candidate_means[idx]
    )
    fit_line_cand = slope_cand * top_k_values + intercept_cand
    ax.plot(
        top_k_values,
        fit_line_cand,
        color=colors[0],
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    # Linear fit for num calls
    slope_call, intercept_call, _, _, _ = stats.linregress(
        top_k_values, calls_means[idx]
    )
    fit_line_call = slope_call * top_k_values + intercept_call
    ax.plot(
        top_k_values,
        fit_line_call,
        color=colors[1],
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )

    # Print slopes
    print(
        f"δ = {delta:.1f}: Candidate slope = {slope_cand:.2f}, Calls slope = {slope_call:.2f}"
    )

    ax.set_title(rf"$\delta = {delta:.0f}$", fontsize=9, pad=2)

    # Set ylabel for left column
    if idx % 2 == 0:
        ax.set_ylabel("Count")

    # Set xlabel for bottom row
    if idx >= 2:
        ax.set_xlabel("Top-$k$")

# Set x ticks on all axes
for ax in axes:
    ax.set_xticks([20, 50, 80])

# Add shared legend at top
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.12),
    # fontsize=7,
)

# Save figure
output_path = (
    Path(__file__).parent.parent.parent
    / "figures"
    / "appendix"
    / "topk_candidate_calls.pdf"
)
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {output_path}")

plt.show()
