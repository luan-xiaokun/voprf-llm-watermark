import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.evaluation.gsm8k import evaluate_gsm8k_benchmark
from watermark_suite.evaluation.humaneval import evaluate_human_eval_benchmark
from watermark_suite.schemes import (
    KGWAdapter,
    PDWAdapter,
    RDFAdapter,
    VOWAdapter,
    WatermarkAdapter,
)
from watermark_suite.schemes.vow.key import get_server_seed
from watermark_suite.utils import io_utils

DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_OUTPUT_DIR = "output/downstream_task"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"

METHOD_PARAMS_DICT = {
    "vow": ["window_size", "delta", "gamma"],
    "lefthash": ["delta", "gamma"],
    "selfhash": ["delta", "gamma"],
    "rdf": ["length", "seed", "watermark_device"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run downstream task evaluation")

    parser.add_argument(
        "task",
        type=str,
        choices=["gsm8k", "humaneval"],
        help="Downstream task to evaluate",
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["vow", "lefthash", "selfhash", "rdf", "pdw"],
        help="Watermarking method",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model name or path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=7,
        help="Window size for watermarking (VOW)",
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
        default=0.375,
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
        "--batch_size", type=int, default=64, help="Batch size for generation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.method is None:
        args.method = "no-watermark"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    watermark_param_name = io_utils.watermark_parameter_dict_to_string(
        args.method, vars(args)
    )
    output_file_name = f"{args.task}_{args.method}_{watermark_param_name}.json"
    output_file = str(output_dir / output_file_name)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, padding_side="left", local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.method == "no-watermark":
        adapter = WatermarkAdapter()
        adapter.model = model
        adapter.tokenizer = tokenizer
    elif args.method == "vow":
        adapter = VOWAdapter(
            model=model,
            tokenizer=tokenizer,
            window_size=args.window_size,
            delta=args.delta,
            gamma=args.gamma,
            seed=get_server_seed(DEFAULT_SERVER_SEED_PATH),
        )
    elif args.method in ["lefthash", "selfhash"]:
        adapter = KGWAdapter(
            model=model,
            tokenizer=tokenizer,
            delta=args.delta,
            gamma=args.gamma,
            seeding_scheme=args.method,
        )
    elif args.method == "rdf":
        adapter = RDFAdapter(
            model=model,
            tokenizer=tokenizer,
            watermark_sequence_length=args.length,
            seed=args.seed,
            watermark_sequence_device=args.watermark_device,
        )
    elif args.method == "pdw":
        adapter = PDWAdapter(
            model=model,
            tokenizer=tokenizer,
            timing=False,
        )
    else:
        raise ValueError(f"Unknown watermarking method: {args.method}")

    if args.task == "gsm8k":
        result = evaluate_gsm8k_benchmark(adapter, args.batch_size, 4, output_file)
    elif args.task == "humaneval":
        result = evaluate_human_eval_benchmark(adapter, args.batch_size, output_file)
    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == "__main__":
    main()
