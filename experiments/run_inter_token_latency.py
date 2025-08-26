import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.schemes import (
    KGWAdapter,
    PDWAdapter,
    RDFAdapter,
    VOWAdapter,
    WatermarkAdapter,
)
from watermark_suite.schemes.vow.key import get_server_seed

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"


def parse_args():
    parser = argparse.ArgumentParser(description="Run text generation with watermarks")
    parser.add_argument(
        "method",
        nargs="?",
        type=str,
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw"],
        help="Watermarking method",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model name or path"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=4,
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
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--greedy", action="store_true", help="Whether to use greedy decoding"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p (nucleus) sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # prepare parameters and output file
    do_sample = not args.greedy
    if args.method is None:
        args.method = "no-watermark"

    # load model and tokenizer
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
    suppress_tokens = [tokenizer.eos_token_id]

    # get watermark adapter
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
            timing=True,
        )
    else:
        raise ValueError(f"Unknown watermarking method: {args.method}")

    dataset = load_dataset(
        "json", data_files="data/c4_realnewslike_subset_673.jsonl", split="train"
    )

    batch_sizes = [1, 8, 32]
    if args.method == "pdw":
        batch_sizes = [1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter.to(device)
    adapter.eval()

    if args.method != "pdw":
        print(f"Warming up...")
        warmup_prompt = ["Just a test to warm up the GPU"]
        _ = adapter(
            prompts=warmup_prompt,
            max_new_tokens=8,
            pad_token_id=adapter.tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        print("Warm-up finished")

    adapter.timing = True

    for batch_size in batch_sizes:
        subset = dataset.select(range(batch_size))
        batch = next(subset.iter(batch_size=batch_size))
        print(f"Profiling batch size {batch_size}...")
        with torch.no_grad():
            _ = adapter(
                prompts=batch["prompt_text"],
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                suppress_tokens=suppress_tokens,
                pad_token_id=adapter.tokenizer.eos_token_id,
            )


if __name__ == "__main__":
    main()
