import numpy as np
from scipy import stats

from setting import *


with open("data/plot_data/fpr_4.npy", "rb") as f:
    fpr_4 = np.load(f)

num = 1_000_010
sorted_fpr_4 = np.sort(fpr_4)[-num:]

fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")

theoretical_q4, empirical_q4 = stats.probplot(fpr_4, dist="uniform", fit=False)
mask_4 = theoretical_q4 > 0
ratio_4 = empirical_q4[mask_4] / theoretical_q4[mask_4]

ax.scatter(
    theoretical_q4[mask_4],
    ratio_4,
    s=1,
    alpha=0.5,
    color="#E69F00",
    rasterized=True,
    zorder=2,
)

ax.axhline(1, color="black", linestyle="--", lw=1, label="Ideal Ratio")

ax.set_xlabel("Theoretical FPR")
ax.set_ylabel("Empirical FPR / Theoretical FPR")

ax.set_xscale("log")
ax.set_ylim([0, 3])

ax.set_xticks([1e-6, 1e-4, 1e-2, 1])

ax.grid(which="major", linestyle=":", alpha=0.7)
ax.legend()
plt.savefig("figures/appendix/fpr.pdf", dpi=300)
