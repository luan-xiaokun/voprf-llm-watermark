import itertools
import secrets
import struct
from typing import NamedTuple

import matplotlib as mpl
import numpy as np
import tqdm
from datasets import load_dataset
from matplotlib import pyplot as plt
from scipy import special
from transformers import AutoTokenizer
from voprf_py import VoprfServer

HASH_BITS = 512
MODEL_PATH = "Qwen/Qwen2.5-3B"
DATA_FILE_PATH = "data/c4_realnewslike_subset_673.jsonl"
STRIDE = 255

mpl.rcParams.update(
    {
        "font.family": "serif",
        "axes.labelsize": 7,
        "axes.linewidth": 0.5,
        "font.size": 7,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
        "mathtext.fontset": "stix",
    }
)


class DetectionResult(NamedTuple):
    green_token_num: int
    effective_token_num: int
    total_token_num: int
    green_ratio: float
    p_value: float
    p_values_per_token: list[float] | None


def _local_detect_hacked(
    token_ids_list: list[list[int]],
    window_size: int,
    gamma: float,
    voprf_server: VoprfServer,
) -> list[DetectionResult]:
    n_grams_list = [
        [
            tuple(token_ids[i - window_size : i + 1])
            for i in range(window_size, len(token_ids))
        ]
        for token_ids in token_ids_list
    ]
    deduplicated_n_grams_list = [
        list(dict.fromkeys(n_grams)) for n_grams in n_grams_list
    ]
    deduplicated_n_grams_lengths = [
        len(n_grams) for n_grams in deduplicated_n_grams_list
    ]
    indices = list(itertools.accumulate(deduplicated_n_grams_lengths, initial=0))

    flatten_msg_inputs = [
        struct.pack(f">{window_size + 1}I", *n_gram)
        for n_grams in n_grams_list
        for n_gram in n_grams
    ]
    flatten_msg_hashes = voprf_server.batch_evaluate(flatten_msg_inputs)
    msg_hashes_list = [
        flatten_msg_hashes[indices[i] : indices[i + 1]] for i in range(len(indices) - 1)
    ]

    threshold = gamma * (1 << HASH_BITS)
    threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")

    results = []
    for i, msg_hashes in enumerate(msg_hashes_list):
        dedup_n_grams = deduplicated_n_grams_list[i]
        gram_to_color_map = {
            gram: h < threshold_bytes for gram, h in zip(dedup_n_grams, msg_hashes)
        }
        effective_token_num = len(dedup_n_grams)
        green_token_num = sum(gram_to_color_map.values())
        p_value = special.betainc(
            green_token_num, effective_token_num - green_token_num + 1, gamma
        )
        if effective_token_num > 0:
            green_ratio = green_token_num / effective_token_num
        else:
            green_ratio = 0.0
        detect_result = DetectionResult(
            green_token_num=green_token_num,
            effective_token_num=effective_token_num,
            total_token_num=len(token_ids_list[i]),
            green_ratio=green_ratio,
            p_value=p_value,
            p_values_per_token=None,
        )
        results.append(detect_result)

    return results


def plot_fpr_curve(p_value_list: list[float], label: str, ax: plt.Axes):
    p_values = np.array(p_value_list)
    N = len(p_values)

    sorted_p_values = np.sort(p_values)

    empirical_fpr = (np.arange(1, N + 1)) / N

    ax.plot(sorted_p_values, empirical_fpr, label=label, linewidth=0.8)


def main():
    total_num = 1_000_000
    gamma = 0.25
    seed_num = 10

    seeds = [secrets.token_bytes(32) for _ in range(seed_num)]
    servers = [VoprfServer(seed) for seed in seeds]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    vocab_size = tokenizer.vocab_size

    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

    print(f"Using model: {MODEL_PATH}")
    print(f"Using dataset: {DATA_FILE_PATH}")
    print(f"Gamma: {gamma}")
    print(f"Vocab size: {vocab_size}")

    window_size_4_p_values = []
    # window_size_7_p_values = []
    pbar = tqdm.tqdm(total=total_num, desc="Processing samples")
    for sample in dataset:
        text = sample["text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            token_ids[i : i + STRIDE] for i in range(0, len(token_ids) - STRIDE, STRIDE)
        ]
        for server in servers:
            w4_results = _local_detect_hacked(chunks, 4, gamma, server)
            # w7_results = _local_detect_hacked(chunks, 7, gamma, server)
            window_size_4_p_values.extend([result.p_value for result in w4_results])
            # window_size_7_p_values.extend([result.p_value for result in w7_results])
            pbar.update(len(w4_results))
        if len(window_size_4_p_values) >= total_num:
            break
    pbar.close()

    array1 = np.array(window_size_4_p_values)
    # array2 = np.array(window_size_7_p_values)
    with open("data/validation_data/fpr_4.npy", "wb") as f:
        np.save(f, array1)
    # with open("data/validation_data/fpr_7.npy", "wb") as f:
    #     np.save(f, array2)

    # fig, ax = plt.subplots(figsize=(2.5, 2))
    # plot_fpr_curve(window_size_4_p_values, label=r"$h=4$", ax=ax)
    # plot_fpr_curve(window_size_7_p_values, label=r"$h=7$", ax=ax)

    # min_val = 1e-7
    # ax.plot([min_val, 1], [min_val, 1], "k--", linewidth=0.8)

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    # ax.set_xlabel("Theoretical FPR")
    # ax.set_ylabel("Empirical FPR")
    # ax.grid(True, which="major", ls="--", linewidth=0.5, alpha=0.8)
    # ax.legend()

    # plt.savefig("fpr_curve.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
