import argparse
from pathlib import Path
from typing import Callable

import torch
from sentence_transformers import SentenceTransformer
from voprf_py import BlindedElement, EvaluationElement, Proof, PublicKey, VoprfServer

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


def main():
    args = parse_args()

    # prepare parameters and input file
    do_sample = not args.greedy
    if args.robustness_eval and args.method != "rdf":
        args.step_size = 10
    elif args.robustness_eval and args.method == "rdf":
        args.step_size = 100

    input_file_name = io_utils.get_file_name(
        args.method, args.tokenizer, args.dataset, vars(args), do_sample, args.top_k
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_file_path = input_dir / input_file_name

    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print(f"Loading text from {input_file_path}")
    samples = list(io_utils.read_jsonlines(str(input_file_path)))

    generated = [sample[DEFAULT_TARGET_COLUMN] for sample in samples]
    attacked = [sample[args.target_column] for sample in samples]

    generated_embeddings = model.encode(generated, convert_to_tensor=True)
    attacked_embeddings = model.encode(attacked, convert_to_tensor=True)

    sbert_scores = []
    for i in range(len(generated)):
        score = model.similarity(generated_embeddings[i], attacked_embeddings[i])
        sbert_scores.append(score)

    sbert_scores = torch.tensor(sbert_scores)

    print(f"Average SBERT score: {sbert_scores.mean().item():.6f}")
    print(f"Standard deviation: {sbert_scores.std().item():.6f}")


if __name__ == "__main__":
    main()

# paraphrasing
# rdf  0.784518
# pdw  0.494835
# selfhash 0.858196
# lefthash 0.838215
# vow 0.883821