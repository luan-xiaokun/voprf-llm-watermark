import json

from setting import *


import numpy as np


def load_data():
    costs_w4 = []
    with open("data/plot_data/detection_costs_w4.jsonl") as f:
        for line in f:
            costs_w4.append(json.loads(line))
    return costs_w4


def plot_communication_overhead(costs_w4: list):
    raw_text_bytes_w4 = np.array([r["raw_text_bytes"][0] for r in costs_w4])

    blinded_elements_bytes_w4 = np.array(
        [r["blinded_elements_bytes"][0] for r in costs_w4]
    )
    evaluation_elements_bytes_w4 = np.array(
        [r["evaluation_elements_bytes"][0] for r in costs_w4]
    )
    proof_bytes_w4 = np.array([r["proof_bytes"][0] for r in costs_w4])

    total_communication_bytes_w4 = np.array(
        [r["total_communication_bytes"][0] for r in costs_w4]
    )

    fig, ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
    ax.plot(
        raw_text_bytes_w4 / 1000,
        blinded_elements_bytes_w4 / 1000 / 1000,
        label="Blinded",
        marker="s",
        markersize=3,
    )
    ax.plot(
        raw_text_bytes_w4 / 1000,
        evaluation_elements_bytes_w4 / 1000 / 1000,
        label="Evaluation",
        marker="o",
        markersize=3,
    )
    ax.plot(
        raw_text_bytes_w4 / 1000,
        proof_bytes_w4 / 1000 / 1000,
        label="Proof",
        marker="d",
        markersize=3,
    )
    ax.plot(
        raw_text_bytes_w4 / 1000,
        total_communication_bytes_w4 / 1000 / 1000,
        label="Total",
        marker="^",
        markersize=3,
    )

    ax.set_xlabel("Text Size (KB)")
    ax.set_ylabel("Data Transferred (MB)")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    # plt.tight_layout()
    plt.savefig("figures/evaluation/communication_overhead.pdf", dpi=300)


def plot_computational_overhead(costs_w4: list):
    raw_text_bytes_w4 = np.array([r["raw_text_bytes"][0] for r in costs_w4])

    preparation_time_w4 = np.array([r["preparation_time"][0] for r in costs_w4])
    evaluation_time_w4 = np.array([r["evaluation_time"][0] for r in costs_w4])
    finalization_time_w4 = np.array([r["finalization_time"][0] for r in costs_w4])
    total_time_w4 = np.array([r["total_time"][0] for r in costs_w4])

    fig, ax = plt.subplots(figsize=(3.2, 2.1), layout="constrained")
    ax.plot(
        raw_text_bytes_w4 / 1000,
        preparation_time_w4,
        label="Preparation",
        marker="s",
        markersize=3,
    )

    ax.plot(
        raw_text_bytes_w4 / 1000,
        evaluation_time_w4,
        label="Evaluation",
        marker="o",
        markersize=3,
    )

    ax.plot(
        raw_text_bytes_w4 / 1000,
        finalization_time_w4,
        label="Finalization",
        marker="d",
        markersize=3,
    )
    ax.plot(
        raw_text_bytes_w4 / 1000,
        total_time_w4,
        label="Total",
        marker="^",
        markersize=3,
    )

    ax.set_xlabel("Text Size (KB)")
    ax.set_ylabel("Computational Overhead (s)")
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()
    # plt.tight_layout()
    plt.savefig("figures/evaluation/computational_overhead.pdf", dpi=300)


def main():
    costs_w4 = load_data()
    plot_communication_overhead(costs_w4)
    plot_computational_overhead(costs_w4)


if __name__ == "__main__":
    main()
