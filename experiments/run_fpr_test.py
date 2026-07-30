import argparse
import os
import random
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from voprf_py import BlindedElement, EvaluationElement, Proof, PublicKey, VoprfServer

from watermark_suite.core import detect_texts
from watermark_suite.schemes import KGWDetector, PDWDetector, RDFDetector, VOWDetector
from watermark_suite.schemes.upv import UPVDetector
from watermark_suite.schemes.vow.key import get_server_seed

# Defaults
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B"
DEFAULT_DATASET = "c4"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"


random.seed(42)


def get_public_key(voprf_server: VoprfServer) -> PublicKey:
    return voprf_server.get_public_key()


def get_server_interface(
    voprf_server: VoprfServer,
) -> Callable[[list[BlindedElement]], tuple[list[EvaluationElement], Proof]]:
    def server_interface(
        blinded_elements: list[BlindedElement],
    ) -> tuple[list[EvaluationElement], Proof]:
        return voprf_server.batch_blind_evaluate(blinded_elements)

    return server_interface


def load_data(
    min_tokens: int,
    tokenizer: AutoTokenizer,
    max_samples: int = 100_000,
    seed: int = 42,
) -> list[str]:
    ds = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)

    print(f"Dataset column names: {ds.column_names}")

    # Shuffle the streaming dataset with a buffer for random sampling
    ds = ds.shuffle(seed=seed, buffer_size=10_000)

    valid_texts = []
    text_column = "text"  # For c4

    # Iterate through shuffled streaming dataset and collect valid samples
    for item in ds:
        content = item.get(text_column, "")

        # Determine if content is usable string
        if isinstance(content, list):
            # Some datasets have answers as list
            if len(content) > 0 and isinstance(content[0], str):
                content = content[0]
            else:
                continue

        if not isinstance(content, str) or not content.strip():
            continue

        # Check token length
        tokens = tokenizer.encode(content, add_special_tokens=False)
        if len(tokens) >= min_tokens:
            valid_texts.append(content)

        if len(valid_texts) >= max_samples:
            break

    print(f"Loaded {len(valid_texts)} texts satisfying min_tokens={min_tokens}")
    return valid_texts


def parse_args():
    parser = argparse.ArgumentParser(description="Test False Positive Rate (FPR)")
    parser.add_argument(
        "method",
        type=str,
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw", "upv"],
        help="Watermarking method",
    )
    # parser.add_argument(
    #     "--dataset",
    #     type=str,
    #     default=DEFAULT_DATASET,
    #     choices=["c4", "eli5"],
    #     help="Dataset name",
    # )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="Model name or path "
    )
    parser.add_argument(
        "--min_tokens", type=int, default=200, help="Minimum token length required"
    )
    parser.add_argument(
        "--max_samples", type=int, default=500, help="Maximum number of samples to test"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/detection_cost",
        help="Directory to save plots/results",
    )
    # Method specific args
    parser.add_argument(
        "--delta", type=float, default=2.5
    )
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--length", type=int, default=256)  # RDF
    parser.add_argument("--seed", type=int, default=42)  # RDF
    parser.add_argument("--n_runs", type=int, default=100)  # RDF
    parser.add_argument(
        "--use_local", action="store_true", help="Use local detection (VOW)"
    )

    return parser.parse_args()


def plot_p_values(p_values, method, dataset, output_dir):
    plt.figure(figsize=(10, 5))

    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(
        p_values, bins=20, range=(0, 1), alpha=0.7, color="blue", edgecolor="black"
    )
    plt.title(f"P-value Distribution ({method} on {dataset})")
    plt.xlabel("P-value")
    plt.ylabel("Frequency")

    # CDF (Cumulative Distribution Function)
    plt.subplot(1, 2, 2)
    sorted_p = np.sort(p_values)
    yvals = np.arange(len(sorted_p)) / float(len(sorted_p))
    plt.plot(sorted_p, yvals, label="Observed")
    plt.plot([0, 1], [0, 1], "r--", label="Uniform")
    plt.title("CDF of P-values")
    plt.xlabel("P-value")
    plt.ylabel("Cumulative Probability")
    plt.legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"fpr_p_value_dist_{method}_{dataset}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    plt.close()


def main():
    args = parse_args()

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    # Load Data
    texts = load_data(args.min_tokens, tokenizer, args.max_samples)
    if not texts:
        print("No valid texts found!")
        return

    # Setup Detector
    detect_kwargs = {}
    if args.method == "vow":
        detector = VOWDetector(
            tokenizer,
            seed=get_server_seed(DEFAULT_SERVER_SEED_PATH),
            gamma=args.gamma,
            window_size=args.window_size,
        )
        if not args.use_local:
            detect_kwargs.update(
                dict(
                    server_public_key=get_public_key(detector.voprf_server),
                    server_interface=get_server_interface(detector.voprf_server),
                )
            )
    elif args.method in ["lefthash", "selfhash"]:
        detector = KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=args.gamma,
            seeding_scheme=args.method,
            device="cuda" if torch.cuda.is_available() else "cpu",
            tokenizer=tokenizer,
            normalizers=[],
            ignore_repeated_ngrams=True,
        )
    elif args.method == "rdf":
        detector = RDFDetector(
            tokenizer,
            length=args.length,
            seed=args.seed,
            n_runs=args.n_runs,
        )
    elif args.method == "pdw":
        detector = PDWDetector()
    elif args.method == "upv":
        detector = UPVDetector(
            tokenizer,
            "experiments/upv_baseline/model",
            window_size=4,
            bits_num=18,
            gamma=0.5,
        )
    else:
        raise ValueError(f"Unknown watermarking method: {args.method}")

    print(f"Running detection on {len(texts)} samples using {args.method}...")

    # Significance levels for FPR check
    significance_levels = [0.1, 0.05, 0.01, 0.005, 0.001, 1e-4]

    final_result = detect_texts(
        detector,
        texts,
        token_num=None,
        significance_levels=significance_levels,
        use_local=args.use_local,
        include_raw_results=True,
        **detect_kwargs,
    )

    tpr_dict = final_result["tpr_dict"]
    p_value_median = final_result["p_value_median"]

    # Gather p-values
    p_values = final_result["all_p_values"]

    print("-" * 50)
    print(f"FPR Results (False Positive Rate)")
    print(f"Method: {args.method}")
    print("-" * 50)
    for sl, fpr in sorted(tpr_dict.items(), reverse=True):
        print(f"Threshold {sl}: FPR = {fpr:.4f}")
    print("-" * 50)

    if args.method in ["lefthash", "selfhash", "vow", "rdf"]:
        print(f"Median P-Value: {p_value_median:.6e} (Expected ~0.5 for invalid)")

        output_dir = args.output_dir
        file_name = f"fpr_p_value_dist_{args.method}_{args.max_samples}.npy"
        np.save(os.path.join(output_dir, file_name), np.array(p_values))
        print(f"P-values saved to {os.path.join(output_dir, file_name)}")

        from scipy import stats

        ks_statistic, p_val = stats.kstest(p_values, "uniform")
        print(
            f"KS Test for Uniformity: statistic={ks_statistic:.4f}, p-value={p_val:.4e}"
        )
        if p_val > 0.05:
            print("=> Consistent with uniform distribution (p > 0.05)")
        else:
            print("=> Not consistent with uniform distribution (p <= 0.05)")
    else:
        print("Binary or non-probabilistic method. FPR is rate of detection.")
        # For PDW/UPV, p_value might be 0/1. If so, median p-value tells us something.
        print(f"Median P-Value/Score: {p_value_median}")


if __name__ == "__main__":
    main()
