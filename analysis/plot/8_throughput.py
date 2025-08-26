from setting import *

vow_records = {
    "--": {
        (1.0, 0.5): 1607.79,
        (1.5, 0.5): 1383.27,
        (2.0, 0.5): 1140.92,
        (2.5, 0.5): 865.26,
        (3.0, 0.5): 645.91,
        (3.5, 0.5): 470.05,
        (4.0, 0.5): 328.64,
        (2.5, 0.125): 601.56,
        (2.5, 0.25): 674.35,
        (2.5, 0.375): 766.23,
        (2.5, 0.5): 857.25,
        (2.5, 0.625): 935.05,
        (2.5, 0.75): 1075.25,
        (2.5, 0.875): 1308.85,
    },
    "top-10": {
        (1.0, 0.5): 976.03,
        (1.5, 0.5): 965.10,
        (2.0, 0.5): 954.28,
        (2.5, 0.5): 936.15,
        (3.0, 0.5): 932.95,
        (3.5, 0.5): 924.30,
        (4.0, 0.5): 909.56,
        (2.5, 0.125): 725.33,
        (2.5, 0.25): 806.27,
        (2.5, 0.375): 860.99,
        (2.5, 0.5): 918.36,
        (2.5, 0.625): 910.35,
        (2.5, 0.75): 930.69,
        (2.5, 0.875): 965.93,
    },
    "top-50": {
        (1.0, 0.5): 476.53,
        (1.5, 0.5): 470.80,
        (2.0, 0.5): 464.41,
        (2.5, 0.5): 444.01,
        (3.0, 0.5): 440.43,
        (3.5, 0.5): 433.82,
        (4.0, 0.5): 435.57,
        (2.5, 0.125): 289.18,
        (2.5, 0.25): 344.30,
        (2.5, 0.375): 412.24,
        (2.5, 0.5): 449.32,
        (2.5, 0.625): 455.72,
        (2.5, 0.75): 455.75,
        (2.5, 0.875): 457.85,
    },
}

kgw_records = {
    "--": {
        "lefthash": 1388.36,
        "selfhash": 9.49,
    },
    "top-10": {
        "lefthash": 1336.37,
        "selfhash": 9.81,
    },
    "top-50": {
        "lefthash": 1263.68,
        "selfhash": 9.93,
    },
}


def plot_tpr_against_params():
    fig, ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
    deltas = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    default_tps = [vow_records["--"][(d, 0.5)] for d in deltas]
    top10_tps = [vow_records["top-10"][(d, 0.5)] for d in deltas]
    top50_tps = [vow_records["top-50"][(d, 0.5)] for d in deltas]

    ax.plot(deltas, default_tps, label="Multinomial", marker="s", markersize=5)
    ax.plot(deltas, top10_tps, label="Top-10", marker="o", markersize=5)
    ax.plot(deltas, top50_tps, label="Top-50", marker="d", markersize=5)
    ax.set_xticks(deltas)
    ax.set_xlabel("Delta ($\\delta$)")
    ax.set_ylabel("Throughput (TPS)")
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    plt.savefig("figures/appendix/tps_params_delta.pdf", dpi=300)

    fig, ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
    gammas = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
    default_tps = [vow_records["--"][(2.5, g)] for g in gammas]
    top10_tps = [vow_records["top-10"][(2.5, g)] for g in gammas]
    top50_tps = [vow_records["top-50"][(2.5, g)] for g in gammas]

    ax.plot(gammas, default_tps, label="Multinomial", marker="s", markersize=5)
    ax.plot(gammas, top10_tps, label="Top-10", marker="o", markersize=5)
    ax.plot(gammas, top50_tps, label="Top-50", marker="d", markersize=5)
    ax.set_xticks(gammas)
    ax.set_xlabel("Gamma ($\\gamma$)")
    ax.set_ylabel("Throughput (TPS)")
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    plt.savefig("figures/appendix/tps_params_gamma.pdf", dpi=300)


def main():
    plot_tpr_against_params()


if __name__ == "__main__":
    main()
