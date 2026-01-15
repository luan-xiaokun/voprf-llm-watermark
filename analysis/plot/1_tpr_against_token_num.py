import json

import numpy as np
from scipy import stats

from setting import *

vow_tpr_list_file = "data/plot_data/vow_tpr_list_w4_d2.5_g0.5.npy"
rdf_tpr_list_file = "data/plot_data/rdf_tpr_list_n100.npy"


def load_kgw_records(
    seeding_scheme: str, significance_level: float = 1e-5, window_size: int = 4
):
    file_name = f"kgw_{seeding_scheme}_Qwen2.5-3B_multinomial-top50_d2.0_g0.25.jsonl"
    z_score_at_T = []
    with open(f"data/{file_name}") as f:
        for line in f:
            record = json.loads(line)
            if "z_score_at_T" in record:
                z_score_at_T.append(record["z_score_at_T"])
    p_values_at_T = [[stats.norm.sf(z) for z in record] for record in z_score_at_T]
    tpr_list = [0.0] * window_size
    max_length = max(len(record) for record in z_score_at_T)
    for i in range(max_length):
        p_values_at_i = []
        for p_values in p_values_at_T:
            if i < len(p_values):
                p_values_at_i.append(p_values[i])
        detected_num = sum(p < significance_level for p in p_values_at_i)
        tpr = detected_num / len(p_values_at_i) if p_values_at_i else 0.0
        tpr_list.append(tpr)
    return np.array(tpr_list)


def load_upv_records():
    file_name = "generated_text_upv_Qwen2.5-3B_c4_multinomial_18bits_5layers.json"
    with open(f"data/plot_data/{file_name}") as f:
        records = json.load(f)
    tpr_list = records["tpr_per_step"]["1e-06"]
    return tpr_list, 10


def make_violin_plot():
    fig, ax = plt.subplots(figsize=(3.2, 1.8), layout="constrained")

    all_p_vals = []
    positions = [75, 150]
    for pos in positions:
        p_vals_array = np.load(f"data/plot_data/p_vals_at_{pos}_vow.npy")
        all_p_vals.append(p_vals_array)
    neg_p_vals_array = np.load("data/plot_data/p_vals_negative_vow.npy")
    all_p_vals.append(neg_p_vals_array)
    ax.boxplot(all_p_vals)

    # find the largest and the second largest elements' indices
    # in all_p_vals[0]
    largest_indices = np.argsort(all_p_vals[0])[-2:]
    print(largest_indices)

    ax.set_xticks(range(1, 4), ["75 Token", "150 Token", "Negative"])

    ax.set_ylabel("$p$-Values")
    # plt.tight_layout()
    plt.savefig("figures/p_vals_violin_plot_vow_g0.5.pdf", dpi=300, bbox_inches="tight")


def main():
    # make_violin_plot()

    vow_tpr_array = np.load(vow_tpr_list_file)
    sampled_vow_tpr_array = vow_tpr_array[::5]
    sampled_x_vow = np.arange(len(vow_tpr_array))[::5]

    rdf_tpr_array = np.load(rdf_tpr_list_file)
    rdf_tpr_array = np.array([0] + rdf_tpr_array.tolist())
    sampled_x_rdf = np.arange(len(rdf_tpr_array)) * 20
    sampled_x_rdf[-1] = 210

    lefthash_tpr_array = load_kgw_records("lefthash", window_size=1)
    sampled_lefthash_tpr_array = lefthash_tpr_array[::5]
    sampled_x_lefthash = np.arange(len(lefthash_tpr_array))[::5]

    selfhash_tpr_array = load_kgw_records("selfhash", window_size=4)
    sampled_selfhash_tpr_array = selfhash_tpr_array[::5]
    sampled_x_selfhash = np.arange(len(selfhash_tpr_array))[::5]

    upv_tpr_list, step_size = load_upv_records()
    sampled_upv_tpr_array = np.array([0] + upv_tpr_list)
    sampled_x_upv = np.arange(len(sampled_upv_tpr_array)) * step_size

    pdw_tpr_array = np.zeros(len(sampled_x_selfhash))
    sampled_x_pdw = sampled_x_selfhash
    print(
        len(sampled_selfhash_tpr_array),
        len(sampled_lefthash_tpr_array),
        len(sampled_selfhash_tpr_array),
        len(rdf_tpr_array),
    )

    fig, ax = plt.subplots(figsize=(3.2, 1.7), layout="constrained")
    ax.plot(sampled_x_vow, sampled_vow_tpr_array, label="VOW", linewidth=1.0)
    ax.plot(
        sampled_x_lefthash, sampled_lefthash_tpr_array, label="LeftHash", linewidth=1.0
    )
    ax.plot(
        sampled_x_selfhash, sampled_selfhash_tpr_array, label="SelfHash", linewidth=1.0
    )
    ax.plot(sampled_x_rdf, rdf_tpr_array, label="RDF", linestyle="--", linewidth=1.0)
    ax.plot(sampled_x_upv, sampled_upv_tpr_array, label="UPV", linewidth=1.0)
    # ax.plot(sampled_x_pdw, pdw_tpr_array, label="PDW", linestyle=":")
    ax.grid(which="major", linestyle=":", alpha=0.7)

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    # plt.tight_layout()
    plt.savefig("figures/evaluation/tpr_against_token_num_vow.pdf", dpi=300)


if __name__ == "__main__":
    main()
