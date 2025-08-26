import gc
import struct
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.schemes.vow import VOWAdapter, VOWDetector
from watermark_suite.utils import io_utils

PROMPT_TEXT_LENGTH = 120
BATCH_SIZE = 4096
SAMPLE_NUM = 5120
TOKEN_PER_SAMPLE = 200
WINDOW_SIZE = 4
DELTA = 2.5
GAMMA = 0.5
TOP_K = None
TOP_P = None
TEMPERATURE = 0.7
HASH_BITS = 512


def main():
    output_file = "data/learning_collection_vow_multinomial_w4_d2.5_g0.5.jsonl"

    dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
    dataset = dataset.take(SAMPLE_NUM)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-3B", torch_dtype=torch.bfloat16, local_files_only=True
    )
    with open("data/server_seed") as f:
        seed = bytes.fromhex(f.read().strip())

    adapter = VOWAdapter(model, tokenizer, WINDOW_SIZE, DELTA, GAMMA, seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter.to(device)
    adapter.eval()

    generation_total_time = 0.0
    generated_total_tokens = 0
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    for batch in (
        pbar := tqdm(
            dataset.iter(batch_size=BATCH_SIZE),
            desc="Processing batches",
            total=SAMPLE_NUM // BATCH_SIZE,
        )
    ):
        prompts = [p[:PROMPT_TEXT_LENGTH] for p in batch["text"]]

        with torch.no_grad():
            starter.record()
            texts = adapter(
                prompts=prompts,
                max_new_tokens=TOKEN_PER_SAMPLE,
                do_sample=True,
                top_p=TOP_P,
                top_k=TOP_K,
                temperature=TEMPERATURE,
                suppress_tokens=[tokenizer.eos_token_id],
                pad_token_id=tokenizer.eos_token_id,
            )
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000.0
            generation_total_time += curr_time

        generated_total_tokens += sum(
            len(tokenizer.encode(text, add_special_tokens=False)) for text in texts
        )

        avg_tps = generated_total_tokens / generation_total_time
        pbar.set_description(f"Avg tokens per second: {avg_tps:.2f} t/s")

        records = [{"prompt_text": p, "generated_text": t} for p, t in enumerate(texts)]
        io_utils.write_jsonlines(output_file, records, "a")


def build_green_cache():
    green_cache_df = "data/green_cache_w4_g0.5_indexed.parquet"
    output_file = "data/learning_collection_vow_multinomial_w4_d2.5_g0.5.jsonl"
    with open("data/server_seed") as f:
        seed = bytes.fromhex(f.read().strip())

    threshold = GAMMA * (1 << HASH_BITS)

    samples = list(io_utils.read_jsonlines(output_file))
    dataset = Dataset.from_list(samples)
    print(dataset)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    detector = VOWDetector(tokenizer, seed, GAMMA, WINDOW_SIZE)

    green_cache = defaultdict(set)
    for batch in tqdm(
        dataset.iter(batch_size=BATCH_SIZE), total=len(dataset) // BATCH_SIZE
    ):
        texts = batch["generated_text"]

        token_ids_list = [
            tokenizer.encode(text, add_special_tokens=False) for text in texts
        ]
        n_grams_list = [
            [
                tuple(token_ids[i - WINDOW_SIZE : i + 1])
                for i in range(WINDOW_SIZE, len(token_ids))
            ]
            for token_ids in token_ids_list
        ]
        flatten_msg_inputs = [
            struct.pack(f">{WINDOW_SIZE + 1}I", *n_gram)
            for n_grams in n_grams_list
            for n_gram in n_grams
        ]
        flatten_msg_hashes = detector.voprf_server.batch_evaluate(flatten_msg_inputs)
        hashes_int_array = np.array(
            [int.from_bytes(h, "big") for h in flatten_msg_hashes]
        )
        mask = hashes_int_array < threshold
        n_grams_array = np.array(
            [n_gram for n_grams in n_grams_list for n_gram in n_grams]
        )
        green_n_grams = n_grams_array[mask]

        for gram_row in green_n_grams:
            key = tuple(gram_row[:-1].tolist())
            value = int(gram_row[-1])
            green_cache[key].add(value)

    unique_context_num = len(green_cache)
    print(f"Unique context num: {unique_context_num}")
    total_token_num = sum(len(tokens) for tokens in green_cache.values())
    print(f"Total token num: {total_token_num}")

    green_cache = [(*grams, list(tokens)) for grams, tokens in green_cache.items()]

    df = pd.DataFrame(
        green_cache, columns=["gram1", "gram2", "gram3", "gram4", "tokens"]
    )
    del green_cache
    gc.collect()

    print("Creating and sorting index...")
    df_indexed = df.set_index(["gram1", "gram2", "gram3", "gram4"])
    df_indexed.sort_index(inplace=True)

    print(f"Saving green cache (DataFrame) to {green_cache_df}...")
    df_indexed.to_parquet(green_cache_df, engine="pyarrow", compression="snappy")


if __name__ == "__main__":
    build_green_cache()

# Unique context num: 45270139
# Total token num: 58039221
