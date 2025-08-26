import numpy as np
from scipy import stats

from setting import *


with open("data/plot_data/chisquare_p_value_4.npy", "rb") as f:
    chisquare_p_value_4 = np.load(f)

print(len(chisquare_p_value_4))


fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")

xy_4 = stats.probplot(chisquare_p_value_4, dist="uniform", fit=False)

diff_4 = xy_4[1] - xy_4[0]

ax.scatter(
    xy_4[0],
    diff_4,
    s=1,
    alpha=0.5,
    color="#E69F00",
    rasterized=True,
    zorder=2,
)


ax.axhline(0, color="black", linestyle="--", lw=1, label="Perfect Uniformity")

ax.set_ylim([-0.03, 0.03])

ax.set_xlabel("Theoretical Quantiles")
ax.set_ylabel("Quantile Residuals")

ax.set_xlim(0, 1)
ax.legend()

ax.grid(True, linestyle=":", alpha=0.7)

plt.savefig("figures/appendix/pseudorandomness.pdf", dpi=300)

kstest_4 = stats.kstest(chisquare_p_value_4, "uniform")

print(kstest_4)
