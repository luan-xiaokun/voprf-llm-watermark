import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import openai
from tqdm import tqdm

# Add the project root to sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from watermark_suite.utils import io_utils

DEFAULT_TOKENIZER = "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit"
DEFAULT_DATASET = "eli5"
DEFAULT_INPUT_DIR = "output/generation"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
CACHE_DIR = ".cache/embeddings"


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate text similarity")
    parser.add_argument(
        "method",
        type=str,
        choices=["lefthash", "selfhash", "rdf", "vow", "pdw", "upv", "no-watermark"],
        help="Watermarking method",
    )

    # Arguments for file name construction (same as run_attack.py)
    parser.add_argument(
        "--model", type=str, default=DEFAULT_TOKENIZER, help="Model name or path "
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
        "--target_column_base",
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

    # Similarity specific arguments
    parser.add_argument(
        "--target_columns",
        nargs="+",
        default=["synonym_replaced", "paraphrased-gpt-3.5-turbo", "paraphrased-gpt-5.1"],
        help="Target columns to compare against generated_text",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model to use",
    )
    parser.add_argument(
        "--ignore_cache",
        action="store_true",
        help="Ignore cached embeddings and recompute",
    )

    return parser.parse_args()


class EmbeddingCache:
    def __init__(self, cache_dir: str, model_name: str, ignore_cache: bool = False):
        self.cache_dir = Path(cache_dir) / model_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ignore_cache = ignore_cache
        self.model_name = model_name
        self.client = openai.OpenAI()

    def _get_cache_path(self, text: str) -> Path:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{text_hash}.json"

    def get_embedding(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []

        cache_path = self._get_cache_path(text)

        if not self.ignore_cache and cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading cache for text hash {cache_path.stem}: {e}")

        # Compute embedding
        try:
            response = self.client.embeddings.create(input=text, model=self.model_name)
            embedding = response.data[0].embedding

            # Save to cache
            with open(cache_path, "w") as f:
                json.dump(embedding, f)

            return embedding
        except Exception as e:
            print(f"Error computing embedding: {e}")
            return []


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2:
        return 0.0
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def main():
    args = parse_args()

    # Load file
    do_sample = not args.greedy
    input_file_name = io_utils.get_file_name(
        args.method, args.model, args.dataset, vars(args), do_sample, args.top_k
    )
    input_dir = Path(args.input_dir).expanduser().resolve()
    input_file_path = input_dir / input_file_name

    if not input_file_path.exists():
        print(f"File not found: {input_file_path}")
        return

    print(f"Loading text from {input_file_path}")
    samples = list(io_utils.read_jsonlines(str(input_file_path)))

    # Initialize cache
    embedding_cache = EmbeddingCache(CACHE_DIR, args.embedding_model, args.ignore_cache)

    results = {col: [] for col in args.target_columns}

    print(f"Computing similarities for {len(samples)} samples...")
    for sample in tqdm(samples):
        gen_text = sample.get(args.target_column_base)
        if not gen_text:
            continue

        gen_emb = embedding_cache.get_embedding(gen_text)
        if not gen_emb:
            continue

        for col in args.target_columns:
            target_text = sample.get(col)
            if target_text:
                target_emb = embedding_cache.get_embedding(target_text)
                sim = cosine_similarity(gen_emb, target_emb)
                results[col].append(sim)

    print("\nSimilarity Results:")
    print("-" * 88)
    print(
        f"{'Target Column':<40} | {'Mean':<10} | {'Median':<10} | {'Std':<10} | {'Count':<10}"
    )
    print("-" * 88)

    for col, sims in results.items():
        if sims:
            mean_sim = np.mean(sims)
            median_sim = np.median(sims)
            std_sim = np.std(sims)
            count = len(sims)
            print(
                f"{col:<40} | {mean_sim:.4f}     | {median_sim:.4f}     | {std_sim:.4f}     | {count:<10}"
            )
        else:
            print(f"{col:<40} | N/A        | N/A        | N/A        | 0")
    print("-" * 88)


if __name__ == "__main__":
    main()
