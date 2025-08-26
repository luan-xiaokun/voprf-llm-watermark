import argparse
from pathlib import Path
from typing import Callable

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from voprf_py import BlindedElement, EvaluationElement, Proof, PublicKey, VoprfServer

from watermark_suite.core import detect_texts
from watermark_suite.core.detect import compute_auc_at_milestones
from watermark_suite.core.metrics import calculate_perplexity
from watermark_suite.schemes import KGWDetector, PDWDetector, RDFDetector, VOWDetector
from watermark_suite.schemes.vow.key import get_server_seed
from watermark_suite.utils import io_utils

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B"
DEFAULT_DATASET = "c4"
DEFAULT_INPUT_DIR = "output/generation"
DEFAULT_TARGET_COLUMN = "generated_text"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"
DEFAULT_EVAL_MODEL = "Qwen/Qwen2.5-7B"


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run watermark detection")
    parser.add_argument(
        "method",
        type=str,
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw"],
        help="Watermarking method",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="Model name or path "
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory to load texts",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt_text",
        help="Column name of dataset for prompts",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="generated_text",
        help="Column name for the generated text",
    )
    parser.add_argument(
        "--token_num",
        type=int,
        default=None,
        help="Maximum number of tokens to run detection per sample",
    )
    parser.add_argument(
        "--significance_levels",
        nargs="+",
        type=float,
        default=None,
        help="Significance levels for watermark detection",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=4,
        help="Window size for watermarking (VOW)",
    )
    parser.add_argument(
        "--use_local", action="store_true", help="Use local detection (VOW)"
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=2.5,
        help="Delta value for watermarking (VOW/KGW)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="Gamma value for watermarking (VOW/KGW)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=256,
        help="Length of watermark sequence (RDF)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for watermark generation (RDF)",
    )
    parser.add_argument(
        "--watermark_device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu", "cuda"],
        help="Device for storing watermark sequence (RDF)",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=100,
        help="Number of runs for watermark detection (RDF)",
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Whether to use greedy decoding"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--no_watermark", action="store_true", help="Disable watermarking"
    )
    parser.add_argument(
        "--step_size", type=int, default=None, help="Step size for TPR calculation"
    )
    parser.add_argument(
        "--include_auc",
        action="store_true",
        help="Whether to include AUC in the output",
    )
    parser.add_argument(
        "--ppl", action="store_true", help="Whether to calculate perplexity"
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default=DEFAULT_EVAL_MODEL,
        help="Model for calculating perplexity",
    )
    parser.add_argument(
        "--robustness_eval",
        action="store_true",
        help="Whether to perform robustness evaluation",
    )
    return parser.parse_args()


def load_negative_samples(args):
    negative_file_name = io_utils.get_file_name(
        "no-watermark", args.tokenizer, args.dataset, {}, not args.greedy, args.top_k
    )
    negative_sample_path = f"{args.input_dir}/{negative_file_name}"
    negative_samples = io_utils.read_jsonlines(negative_sample_path)
    texts = [
        sample.get(args.target_column, sample[DEFAULT_TARGET_COLUMN])
        for sample in negative_samples
    ]
    print(f"Loaded {len(texts)} negative samples from {negative_sample_path}")
    return texts


def main():
    args = parse_args()

    # prepare parameters and input file
    do_sample = not args.greedy
    if args.robustness_eval and args.method != "rdf":
        args.step_size = 10
    elif args.robustness_eval and args.method == "rdf":
        args.step_size = 50

    input_file_name = io_utils.get_file_name(
        args.method, args.tokenizer, args.dataset, vars(args), do_sample, args.top_k
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_file_path = input_dir / input_file_name

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

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
        if args.step_size:
            detect_kwargs.update(dict(return_z_at_T=True))
    elif args.method == "rdf":
        detector = RDFDetector(
            tokenizer,
            length=args.length,
            seed=args.seed,
            n_runs=args.n_runs,
        )
    elif args.method == "pdw":
        detector = PDWDetector()
    else:
        raise ValueError(f"Unknown watermarking method: {args.method}")

    if args.robustness_eval:
        detect_kwargs.update(dict(include_raw_results=True))

    print(f"Loading text from {input_file_path}")
    samples = list(io_utils.read_jsonlines(str(input_file_path)))
    texts = [sample[args.target_column] for sample in samples]
    print(f"Loaded {len(texts)} samples")

    if args.ppl:
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model, torch_dtype=torch.bfloat16, local_files_only=True
        )
        dataset = Dataset.from_list(samples)
        ppl = calculate_perplexity(
            model, tokenizer, dataset, args.prompt_column, args.target_column
        )
        print(f"Perplexity: {ppl:.2f}")

    negative_indices = None
    if args.include_auc:
        try:
            negative_texts = load_negative_samples(args)
            texts = negative_texts + texts
            negative_indices = list(range(len(negative_texts)))
        except FileNotFoundError:
            args.include_auc = False
            print("Warning: Negative samples not found. AUC will be ignored.")

    final_result = detect_texts(
        detector,
        texts,
        token_num=args.token_num,
        significance_levels=args.significance_levels,
        use_local=args.use_local,
        step_size=args.step_size,
        negative_indices=negative_indices,
        include_auc=args.include_auc,
        **detect_kwargs,
    )

    tpr_dict = final_result["tpr_dict"]
    p_value_median = final_result["p_value_median"]
    auc = final_result["auc"]
    overhead_per_sample = final_result["overhead_per_sample"]
    avg_tokens_per_sample = final_result["avg_tokens_per_sample"]

    print(f"Total Samples: {len(texts)}")
    print("-" * 50)
    print(f"True Positive Rate (TPR)")
    for sl, tpr in sorted(tpr_dict.items()):
        print(f"- {tpr:.4f} @ FPR {sl:.0e}")
    print("-" * 50)
    print(f"Median P-Value: {p_value_median:.6e}")
    if auc is not None:
        print("-" * 50)
        print(f"AUC: {auc:.6f}")
    if overhead_per_sample is not None:
        print("-" * 50)
        print(f"Overhead per Sample: {overhead_per_sample * 1000:.2f} ms")
    print("-" * 50)
    print(f"Average Tokens per Sample: {avg_tokens_per_sample:.2f}")

    tpr_per_step = final_result["tpr_per_step"]

    if args.robustness_eval:
        tpr_per_step = final_result["tpr_per_step"]
        raw_results = final_result["raw_results"]
        step_size = args.step_size
        auc_per_step = compute_auc_at_milestones(
            raw_results, negative_indices, step_size
        )
        prefix = args.target_column
        plot_data_file = prefix + "_" + input_file_name.replace(".jsonl", ".json")
        plot_data_dir = Path("data/plot_data")
        plot_data_dir.mkdir(parents=True, exist_ok=True)
        io_utils.write_json(
            plot_data_dir / plot_data_file,
            {
                "tpr_dict": tpr_dict,
                "p_value_median": p_value_median,
                "auc": auc,
                "tpr_per_step": tpr_per_step,
                "auc_per_step": auc_per_step,
            },
            indent=2,
        )


if __name__ == "__main__":
    main()
