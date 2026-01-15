import argparse
from pathlib import Path

from watermark_suite.attacks.paraphrase import (
    paraphrase_texts_by_deepseek,
    paraphrase_texts_by_openai,
)
from watermark_suite.attacks.synonym_replacement import SynonymReplacer
from watermark_suite.utils import io_utils

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DATASET = "eli5"
DEFAULT_INPUT_DIR = "output/generation"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"


def parse_args():
    parser = argparse.ArgumentParser(description="Run watermark detection")
    parser.add_argument(
        "method",
        type=str,
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw", "upv"],
        help="Watermarking method",
    )
    parser.add_argument(
        "--synonym_replacement",
        action="store_true",
        help="Apply synonym replacement attack",
    )
    parser.add_argument(
        "--paraphrase",
        action="store_true",
        help="Apply paraphrase attack",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_TOKENIZER, help="Model name or path "
    )
    parser.add_argument(
        "--dataset", type=str, default=DEFAULT_DATASET, help="Dataset name"
    )
    parser.add_argument(
        "--replacement_rate",
        type=float,
        default=0.3,
        help="Rate of synonym replacement",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help="Directory to load texts",
    )
    parser.add_argument(
        "--target_column",
        type=str,
        default="generated_text",
        help="Column name for the generated text",
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
        "--greedy", action="store_true", help="Whether to use greedy decoding"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-3.5-turbo",
        help="OpenAI model to use (default: gpt-3.5-turbo)",
    )
    parser.add_argument(
        "--use_batch",
        action="store_true",
        help="Use OpenAI Batch API for paraphrasing to save costs",
    )
    parser.add_argument(
        "--batch_id",
        type=str,
        default=None,
        help="Batch ID to retrieve results for",
    )
    parser.add_argument("--no_watermark", action="store_true", help="No watermarking")
    return parser.parse_args()


def main():
    args = parse_args()

    # prepare parameters and input file
    do_sample = not args.greedy

    if args.no_watermark:
        args.method = "no-watermark"

    input_file_name = io_utils.get_file_name(
        args.method, args.model, args.dataset, vars(args), do_sample, args.top_k
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_file_path = input_dir / input_file_name

    print(f"Loading text from {input_file_path}")
    samples = list(io_utils.read_jsonlines(str(input_file_path)))
    texts = [sample[args.target_column] for sample in samples]

    updates = {}

    if args.synonym_replacement:
        replacer = SynonymReplacer()
        synonym_replaced_texts = replacer.replace_synonyms(texts, args.replacement_rate)
        updates["synonym_replaced"] = synonym_replaced_texts

    if args.paraphrase:
        # paraphrased_texts = paraphrase_texts_by_deepseek(texts)
        if args.openai_model == "deepseek":
            paraphrased_texts = paraphrase_texts_by_deepseek(texts)
        else:
            paraphrased_texts = paraphrase_texts_by_openai(
                texts,
                model_name=args.openai_model,
                use_batch=args.use_batch,
                batch_id=args.batch_id,
            )
        if paraphrased_texts is not None:
            updates[f"paraphrased-{args.openai_model}"] = paraphrased_texts

    io_utils.merge_and_write_jsonl(input_file_path, samples, updates)


if __name__ == "__main__":
    main()


# ❯ uv run experiments/run_attack.py vow --paraphrase --top_k 50 --window_size 1 --gamma 0.5 --delta 2.5 --model unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit --openai_model gpt-5.2 --use_batch
# Downloading required NLTK data... (punkt, averaged_perceptron_tagger)
# Download complete.
# Loading text from /home/lxk/projects/voprf/output/generation/vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w1_d2.5_g0.5.jsonl
# Generating batch input file batch_input_20260108-153538.jsonl...
# Submitting batch job...
# Batch job submitted! Batch ID: batch_695f5e4c7c7c8190acbd507c33fc79ba
# Please save this ID. Re-run with --batch_id batch_695f5e4c7c7c8190acbd507c33fc79ba to retrieve results later (up to 24h).
# Successfully merged results and updated /home/lxk/projects/voprf/output/generation/vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w1_d2.5_g0.5.jsonl
