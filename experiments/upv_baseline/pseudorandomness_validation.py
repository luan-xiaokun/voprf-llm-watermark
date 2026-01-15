"""This script validates the pseudorandomness of the proposed IsGreen predicate.

The IsGreen predicate depends on two inputs, i.e., the key (or seed in this script)
and the preceding context token ids. It is also parameterized by the window size,
i.e., the context length, and the gamma value, which is the desired Bernoulli
distribution parameter.

This script samples random key and context token ids from the C4 dataset, and
checks the number of green tokens in the vocabulary determined by them. Based on
the number of green tokens (and the vocabulary size), it carries out a chi-squared
test to if the observed number of green tokens follows the expected Bernoulli
distribution, obtaining the chi-squared statistic and the p-value.

We iterate over the C4 dataset to collect a large number of such p-values, and
finally plot the quantiles of the observed p-values against the theoretical
quantiles of a uniform distribution. If the IsGreen predicate yields the desired
Bernoulli distribution, the p-values should be uniformly distributed, and the
quantile plot should be close to the diagonal line.

For the parameter settings of this script, we use gamma=0.25, window_size=4 or
window_size=7 (for 80-bit and 128-bit security, respectively), and the total
number of samples can be adjusted according to the need.
"""

import secrets
import struct

import matplotlib as mpl
import numpy as np
import tqdm
from datasets import load_dataset
from scipy import stats
from transformers import AutoTokenizer
from voprf_py import VoprfServer
from watermark_suite.schemes.upv import UPVDetector

MODEL_PATH = "Qwen/Qwen2.5-3B"
DATA_FILE_PATH = "data/c4_realnewslike_subset_673.jsonl"
HASH_BITS = 512
STRIDE = 200


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


def main():
    total_num = 1_000
    window_size = 4
    window_sizes = [4, 7]
    gamma = 0.5

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    vocab_size = tokenizer.vocab_size

    # dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

    threshold = gamma * (1 << HASH_BITS)
    threshold_bytes = int(threshold).to_bytes(HASH_BITS // 8, "big")

    print(f"Using model: {MODEL_PATH}")
    # print(f"Using dataset: {DATA_FILE_PATH}")
    print(f"Gamma: {gamma}")
    print(f"Vocab size: {vocab_size}")

    pbar = tqdm.tqdm(total=total_num, desc="Processing samples")
    finished = False
    num = 0
    for sample in dataset:
        text = sample["text"]
        if finished:
            break

        token_ids = tokenizer.encode(text, add_special_tokens=False)


        for i in range(window_size, len(token_ids), STRIDE):
            seed = secrets.token_bytes(32)
            voprf_server = VoprfServer(seed)
            context = token_ids[i - window_size : i]

            context_bytes = struct.pack(f">{window_size}I", *context)
            whole_vocab_msg_inputs = [
                context_bytes + struct.pack(">I", j) for j in range(vocab_size)
            ]
            msg_hashes = voprf_server.batch_evaluate(whole_vocab_msg_inputs)
            green_token_num = sum(h < threshold_bytes for h in msg_hashes)

            _, p_value = stats.chisquare(
                f_obs=[green_token_num, vocab_size - green_token_num],
                f_exp=[vocab_size * gamma, vocab_size * (1 - gamma)],
            )
            p_value_dict[window_size].append(p_value)
            num += 1
            pbar.update(1)

            finished = num >= total_num

            if finished:
                break
    pbar.close()

    for window_size in window_sizes:
        array = np.array(p_value_dict[window_size])
        with open(f"chisquare_p_value_{window_size}.npy", "wb") as f:
            np.save(f, array)

    # fig, ax = plt.subplots(figsize=(4, 4))

    # for window_size in window_sizes:
    #     stats.probplot(
    #         p_value_dict[window_size], dist=stats.uniform, plot=ax, fit=False
    #     )
    # lines = ax.get_lines()

    # data_line1 = lines[0]
    # data_line1.set_marker(".")
    # data_line1.set_markersize(3.0)
    # data_line1.set_markerfacecolor("tab:blue")
    # data_line1.set_markeredgecolor("tab:blue")
    # data_line1.set_alpha(0.8)
    # data_line1.set_label(r"$h=4$")

    # data_line2 = lines[1]
    # data_line2.set_marker(".")
    # data_line2.set_markersize(3.0)
    # data_line2.set_markerfacecolor("tab:orange")
    # data_line2.set_markeredgecolor("tab:orange")
    # data_line2.set_alpha(0.8)
    # data_line2.set_label(r"$h=7$")

    # ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)

    # ax.legend()

    # ax.set_xlabel("Theoretical Quantiles")
    # ax.set_ylabel("Sample Quantiles (p-values)")

    # ax.grid(True, linestyle="--", alpha=0.6)
    # plt.savefig("chisquare_p_value_quantile_plot.pdf")


if __name__ == "__main__":
    main()
