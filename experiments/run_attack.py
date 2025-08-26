import argparse
from pathlib import Path

from watermark_suite.attacks.paraphrase import paraphrase_texts_by_deepseek
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
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw"],
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
    return parser.parse_args()


def main():
    args = parse_args()

    # prepare parameters and input file
    do_sample = not args.greedy

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
        paraphrased_texts = paraphrase_texts_by_deepseek(texts)
        updates["paraphrased"] = paraphrased_texts

    io_utils.merge_and_write_jsonl(input_file_path, samples, updates)


if __name__ == "__main__":
    main()
