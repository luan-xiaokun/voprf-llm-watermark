import json
from pathlib import Path

from setting import *


def load_json_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)
    return data


plot_data_dir = Path("data/plot_data/robustness")
attack_prefixes = [
    "synonym_replaced",
    "paraphrased-gpt-3.5-turbo",
    "paraphrased-gpt-5.1",
]
attack_names = [
    "Synonym Replacement",
    "Paraphrase (GPT-3.5 Turbo)",
    "Paraphrase (GPT-5.1)",
]
candidate_collections = {
    "VOW ($h=1$)": "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w1_d2.5_g0.5",
    "VOW ($h=2$)": "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w2_d2.5_g0.5",
    "VOW ($h=4$)": "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w4_d2.5_g0.5",
    "LeftHash": "lefthash_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_d2.0_g0.25",
    "SelfHash": "selfhash_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_d2.0_g0.25",
    "RDF": "rdf_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_default_l256_s42_cpu",
    "UPV": "upv_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_18bits_5layers",
}

all_records = {
    (prefix, name): load_json_data(
        plot_data_dir / f"{prefix}_{candidate_collections[name]}.json"
    )
    for prefix in attack_prefixes
    for name in candidate_collections.keys()
}
print(f"Loaded {len(all_records)} records")


def plot_tpr_against_token_num():
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.0), sharey=True, layout="constrained"
    )
    axes[0].set_ylabel("True Positive Rate")

    for ax, prefix, attack_name in zip(axes, attack_prefixes, attack_names):
        for name in candidate_collections.keys():
            print(prefix, name)

            record = all_records[(prefix, name)]
            step_size = record["step_size"]
            tpr_per_step = record["tpr_per_step"]["0.01"][: 600 // step_size]
            ax.plot(
                [0] + [step_size * (i + 1) for i in range(len(tpr_per_step))],
                [0] + tpr_per_step,
                label=name,
                linewidth=1.0,
            )

        ax.grid(which="major", linestyle=":", alpha=0.5)
        # ax.grid(which="major", linestyle=":", alpha=0.7)
        ax.set_title(attack_name, fontsize=9, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(" ")
    fig.supxlabel("Number of Tokens", fontsize=9)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=8,
    )

    plt.savefig("robustness_tpr.pdf")
    plt.show()


def plot_auc_against_token_num():
    fig, axes = plt.subplots(
        1, 3, figsize=(7.0, 2.0), sharey=True, layout="constrained"
    )
    axes[0].set_ylabel("AUC")
    axes[0].set_ylim(0.38, 1.02)

    for ax, prefix, attack_name in zip(axes, attack_prefixes, attack_names):
        for name in candidate_collections.keys():
            record = all_records[(prefix, name)]
            step_size = record["step_size"]
            auc_per_step = record["auc_per_step"][: 600 // step_size]
            ax.plot(
                [0] + [step_size * (i + 1) for i in range(len(auc_per_step))],
                [0.5] + auc_per_step,
                label=name,
                linewidth=1.0,
            )

        ax.grid(which="major", linestyle=":", alpha=0.5)
        # ax.grid(which="major", linestyle=":", alpha=0.7)
        ax.set_title(attack_name, fontsize=9, fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(" ")
    fig.supxlabel("Number of Tokens", fontsize=9)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=7,
        # bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=8,
    )

    plt.savefig("robustness_auc.pdf")
    plt.show()


# plot_tpr_against_token_num()
# plot_auc_against_token_num()


def plot_tpr_and_auc_against_token_num():
    fig, axes = plt.subplots(
        2, 3, figsize=(7.0, 3.2), sharey="row", sharex=True, layout="constrained"
    )
    axes[0][0].set_ylabel("AUC")
    axes[1][0].set_ylabel("True Positive Rate")

    axes[0][0].set_ylim(0.38, 1.02)

    for ax, prefix, attack_name in zip(axes[0], attack_prefixes, attack_names):
        for name in candidate_collections.keys():
            record = all_records[(prefix, name)]
            step_size = record["step_size"]
            auc_per_step = record["auc_per_step"][: 600 // step_size]
            ax.plot(
                [0] + [step_size * (i + 1) for i in range(len(auc_per_step))],
                [0.5] + auc_per_step,
                label=name,
                linewidth=1.0,
            )

        ax.grid(which="major", linestyle=":", alpha=0.5)
        ax.set_title(attack_name, fontsize=9, fontweight="bold")

    for ax, prefix, attack_name in zip(axes[1], attack_prefixes, attack_names):
        for name in candidate_collections.keys():
            print(prefix, name)

            record = all_records[(prefix, name)]
            step_size = record["step_size"]
            tpr_per_step = record["tpr_per_step"]["0.01"][: 600 // step_size]
            ax.plot(
                [0] + [step_size * (i + 1) for i in range(len(tpr_per_step))],
                [0] + tpr_per_step,
                label=name,
                linewidth=1.0,
            )

        ax.grid(which="major", linestyle=":", alpha=0.5)

    handles, labels = axes[0][0].get_legend_handles_labels()

    fig.suptitle(" ")
    fig.supxlabel("Number of Tokens", fontsize=9)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        fontsize=8,
    )

    plt.savefig("robustness.pdf")


plot_tpr_and_auc_against_token_num()
