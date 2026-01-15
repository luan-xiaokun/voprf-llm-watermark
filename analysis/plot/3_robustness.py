import json
from pathlib import Path

from setting import *


def load_json_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


plot_data_dir = Path("data/plot_data")
step_size = 10

par_lefthash_record = load_json_data(
    plot_data_dir
    / "paraphrased_lefthash_Qwen2.5-3B-Instruct_eli5_top-50_d2.0_g0.25.json"
)
syn_lefthash_record = load_json_data(
    plot_data_dir
    / "synonym_replaced_lefthash_Qwen2.5-3B-Instruct_eli5_top-50_d2.0_g0.25.json"
)
par_selfhash_record = load_json_data(
    plot_data_dir
    / "paraphrased_selfhash_Qwen2.5-3B-Instruct_eli5_top-50_d2.0_g0.25.json"
)
syn_selfhash_record = load_json_data(
    plot_data_dir
    / "synonym_replaced_selfhash_Qwen2.5-3B-Instruct_eli5_top-50_d2.0_g0.25.json"
)
par_vow_w4_record = load_json_data(
    plot_data_dir / "paraphrased_vow_Qwen2.5-3B-Instruct_eli5_top-50_w4_d2.5_g0.5.json"
)
syn_vow_w4_record = load_json_data(
    plot_data_dir
    / "synonym_replaced_vow_Qwen2.5-3B-Instruct_eli5_top-50_w4_d2.5_g0.5.json"
)
par_rdf_record = load_json_data(
    plot_data_dir / "paraphrased_rdf_Qwen2.5-3B-Instruct_eli5_default_l256_s42_cpu.json"
)
syn_rdf_record = load_json_data(
    plot_data_dir
    / "synonym_replaced_rdf_Qwen2.5-3B-Instruct_eli5_default_l256_s42_cpu.json"
)


def plot_synonym_replacement_results():
    syn_auc_against_token_num = {
        "VOW": [0.0] + syn_vow_w4_record["auc_per_step"][:30],
        "LeftHash": [0.0] + syn_lefthash_record["auc_per_step"][:30],
        "SelfHash": [0.0] + syn_selfhash_record["auc_per_step"][:30],
        "RDF": [0.0] + syn_rdf_record["auc_per_step"][:6],
    }
    syn_tpr_against_token_num_fpr_e_5 = {
        "VOW": [0.0] + syn_vow_w4_record["tpr_per_step"]["1e-05"][:30],
        "LeftHash": [0.0] + syn_lefthash_record["tpr_per_step"]["1e-05"][:30],
        "SelfHash": [0.0] + syn_selfhash_record["tpr_per_step"]["1e-05"][:30],
        # "RDF": syn_rdf_record["tpr_per_step"]["1e-05"][:60],
    }
    syn_tpr_against_token_num_fpr_e_2 = {
        "VOW": [0.0] + syn_vow_w4_record["tpr_per_step"]["0.01"][:30],
        "LeftHash": [0.0] + syn_lefthash_record["tpr_per_step"]["0.01"][:30],
        "SelfHash": [0.0] + syn_selfhash_record["tpr_per_step"]["0.01"][:30],
        "RDF": [0.0] + syn_rdf_record["tpr_per_step"]["0.01"][:6],
    }

    # plot auc scores
    fig, ax = plt.subplots(figsize=(3.2, 1.85), layout="constrained")

    for label, data in syn_auc_against_token_num.items():
        if label != "RDF":
            x = list(range(0, len(data) * step_size, step_size))
            ax.plot(x, data, label=label)
        else:
            x = list(range(0, len(data) * step_size * 5, step_size * 5))
            ax.plot(x, data, label=label, linestyle="--")

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("AUC Score")

    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()

    # plt.tight_layout()
    plt.savefig("figures/evaluation/synonym_replacement_auc.pdf", dpi=300)

    # plot tpr scores
    fig, axes = plt.subplots(
        2, 1, figsize=(3.2, 3.6), sharey=True, layout="constrained"
    )
    colors = {}
    linestyles = {"VOW": "-", "LeftHash": "-", "SelfHash": "-", "RDF": "--"}

    for label, data in syn_tpr_against_token_num_fpr_e_2.items():
        if label != "RDF":
            x = list(range(0, len(data) * step_size, step_size))
        else:
            x = list(range(0, len(data) * step_size * 5, step_size * 5))
        (line,) = axes[1].plot(x, data, label=label, linestyle=linestyles[label])
        colors[label] = line.get_color()

    for label, data in syn_tpr_against_token_num_fpr_e_5.items():
        x = list(range(0, len(data) * step_size, step_size))
        axes[0].plot(x, data, color=colors[label])

    axes[0].set_title("(a) FPR=$10^{-5}$", fontsize=10)
    axes[1].set_title("(b) FPR=$10^{-2}$", fontsize=10)
    axes[0].grid(which="major", linestyle=":", alpha=0.7)
    axes[1].grid(which="major", linestyle=":", alpha=0.7)

    fig.supxlabel("Number of Tokens", x=0.56, fontsize=10)
    fig.supylabel("True Positive Rate (TPR)", fontsize=10)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)
    unique_labels_map = dict(zip(labels, handles))
    fig.suptitle(" ")
    fig.legend(
        unique_labels_map.values(),
        unique_labels_map.keys(),
        loc="upper center",
        bbox_to_anchor=(0.52, 1.0),
        ncol=4,
        fontsize=7,
        # handlelength=1.5,
    )
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper left",
    #     bbox_to_anchor=(1, 0.85),
    #     frameon=False,
    #     title="Method",
    #     fontsize="small",
    # )

    # plt.tight_layout()
    plt.savefig("figures/evaluation/synonym_replacement_tpr.pdf", dpi=300)


def plot_paraphrase_results():
    par_auc_against_token_num = {
        "VOW": [0.0] + par_vow_w4_record["auc_per_step"][:20],
        "LeftHash": [0.0] + par_lefthash_record["auc_per_step"][:20],
        "SelfHash": [0.0] + par_selfhash_record["auc_per_step"][:20],
        "RDF": [0.0] + par_rdf_record["auc_per_step"][:4],
    }
    par_tpr_against_token_num_fpr_e_5 = {
        "VOW": [0.0] + par_vow_w4_record["tpr_per_step"]["1e-05"][:20],
        "LeftHash": [0.0] + par_lefthash_record["tpr_per_step"]["1e-05"][:20],
        "SelfHash": [0.0] + par_selfhash_record["tpr_per_step"]["1e-05"][:20],
        # "VOW ($h=7$)": par_vow_w7_record["tpr_per_step"]["1e-05"][:60],
    }
    par_tpr_against_token_num_fpr_e_2 = {
        "VOW": [0.0] + par_vow_w4_record["tpr_per_step"]["0.01"][:20],
        "LeftHash": [0.0] + par_lefthash_record["tpr_per_step"]["0.01"][:20],
        "SelfHash": [0.0] + par_selfhash_record["tpr_per_step"]["0.01"][:20],
        "RDF": [0.0] + par_rdf_record["tpr_per_step"]["0.01"][:4],
    }

    # plot auc scores
    fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")

    for label, data in par_auc_against_token_num.items():
        if label != "RDF":
            x = list(range(0, len(data) * step_size, step_size))
            ax.plot(x, data, label=label)
        else:
            x = list(range(0, len(data) * step_size * 5, step_size * 5))
            ax.plot(x, data, label=label, linestyle="--")

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("AUC Score")

    ax.grid(which="major", linestyle=":", alpha=0.7)
    ax.legend()

    # plt.tight_layout()
    plt.savefig("figures/evaluation/paraphrase_auc.pdf", dpi=300)

    # plot tpr scores
    fig, ax = plt.subplots(figsize=(3.2, 2.0), layout="constrained")

    for label, data in par_tpr_against_token_num_fpr_e_2.items():
        if label != "RDF":
            x = list(range(0, len(data) * step_size, step_size))
            ax.plot(x, data, label=label)
        else:
            x = list(range(0, len(data) * step_size * 5, step_size * 5))
            ax.plot(x, data, label=label, linestyle="--")

    ax.set_xlabel("Number of Tokens")
    ax.set_ylabel("True Positive Rate (TPR)")

    ax.grid(which="major", linestyle=":", alpha=0.7)

    # handles, labels = ax.get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper left",
    #     bbox_to_anchor=(1, 0.85),
    #     frameon=False,
    #     title="Method",
    #     fontsize="small",
    # )
    ax.legend()

    # plt.tight_layout()
    plt.savefig("figures/evaluation/paraphrase_tpr.pdf", dpi=300)


if __name__ == "__main__":
    plot_synonym_replacement_results()
    plot_paraphrase_results()
