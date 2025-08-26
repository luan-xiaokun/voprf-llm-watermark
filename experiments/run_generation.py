import argparse
import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.core import generate_texts
from watermark_suite.schemes import (
    KGWAdapter,
    PDWAdapter,
    RDFAdapter,
    VOWAdapter,
    WatermarkAdapter,
)
from watermark_suite.schemes.vow.key import get_server_seed
from watermark_suite.utils import io_utils

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_DATASET = "c4"
DEFAULT_OUTPUT_DIR = "output/generation"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"


METHOD_PARAMS_DICT = {
    "vow": ["window_size", "delta", "gamma"],
    "lefthash": ["delta", "gamma"],
    "selfhash": ["delta", "gamma"],
    "rdf": ["length", "seed", "watermark_device"],
}
DATASET_PATH_DICT = {
    "c4": "data/c4_realnewslike_subset_673.jsonl",
    "eli5": "data/eli5",
}
ELI5_PROMPT_TEMPLATE = (
    "Explain the following question like I'm 5 years old. "
    "Use very simple language, short sentences, and analogies a child can understand."
    "\n\nQuestion: {question}"
)
INSTRUCT_STOP_STRINGS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction:",
    "\n\nQuestion",
    "\nQuestion",
    "Question:",
    "<|endoftext|>",
    "####",
]


def eli5_prompt_formatter(question):
    return ELI5_PROMPT_TEMPLATE.format(question=question)


def get_dataset(dataset):
    dataset_path = DATASET_PATH_DICT.get(dataset)
    if dataset == "c4":
        return load_dataset("json", data_files=dataset_path, split="train")
    if dataset == "eli5":
        return load_dataset(dataset_path, split="train")
    raise ValueError(f"Unknown dataset: {dataset}")


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
        "--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name"
    )
    parser.add_argument(
        "--num", type=int, default=500, help="Number of examples to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt_text",
        help="Column name of dataset for prompts",
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
        "--max_new_tokens",
        type=int,
        default=210,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for generation"
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
    parser.add_argument(
        "--suppress_eos",
        action="store_true",
        help="Suppress the end-of-sequence token in the generated text",
    )
    parser.add_argument(
        "--remove_columns",
        nargs="+",
        default=["text", "completion_text", "url", "timestamp", "answer"],
        type=str,
        help="Columns to remove from the dataset",
    )
    parser.add_argument(
        "--no_watermark", action="store_true", help="Disable watermarking"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, will overwrite existing output files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # prepare parameters and output file
    do_sample = not args.greedy
    if args.method is None or args.no_watermark:
        args.method = "no-watermark"

    output_file_name = io_utils.get_file_name(
        args.method, args.model, args.dataset, vars(args), do_sample, args.top_k
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / output_file_name
    if output_file_path.exists() and not args.overwrite:
        print(f"Output file {output_file_path} already exists. Skipping generation.")
        return
    metadata_file_path = output_dir / output_file_name.replace(".jsonl", ".meta.json")
    print(f"Output will be saved to {output_file_path}")

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
    suppress_tokens = [tokenizer.eos_token_id] if args.suppress_eos else None

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
            timing=False,
        )
    else:
        raise ValueError(f"Unknown watermarking method: {args.method}")

    dataset = get_dataset(args.dataset).select(range(args.num))
    print(f"Dataset contains {len(dataset)} samples")

    stop_strings = None
    if args.model.endswith("Instruct"):
        stop_strings = INSTRUCT_STOP_STRINGS

    prompt_formatter = None
    if args.dataset == "eli5":
        prompt_formatter = eli5_prompt_formatter

    # writing metadata
    metadata = vars(args)
    metadata.update(
        dict(
            modification=None,
            file_name=output_file_name,
            timestamp=datetime.datetime.now().isoformat(),
        )
    )
    io_utils.write_json(metadata_file_path, metadata, indent=2)

    for batch in generate_texts(
        adapter,
        dataset,
        args.batch_size,
        args.prompt_column,
        args.max_new_tokens,
        do_sample,
        args.top_p,
        args.top_k,
        args.temperature,
        suppress_tokens,
        stop_strings=stop_strings,
        prompt_formatter=prompt_formatter,
        no_watermark=args.no_watermark,
    ):
        for col in args.remove_columns:
            batch.pop(col, None)

        records = [dict(zip(batch.keys(), values)) for values in zip(*batch.values())]
        io_utils.write_jsonlines(output_file_path, records, mode="a")
    
    print(f"gamma = {args.gamma}, delta = {args.delta}, top-k = {args.top_k}")


if __name__ == "__main__":
    main()
