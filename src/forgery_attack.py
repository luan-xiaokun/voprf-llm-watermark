import itertools
import struct
from pathlib import Path

from transformers import AutoTokenizer
from voprf_py import VoprfServer

import utils
from detection import local_detect_batch_text


def load_c4_newslike(window_size: int):
    data_dir = Path("output")
    all_jsonl_files = []
    for jsonl_file in data_dir.glob("*.jsonl"):
        if f"w{window_size}" in jsonl_file.name:
            all_jsonl_files.append(jsonl_file)

    watermarked_texts = []
    for json_file in all_jsonl_files:
        records = list(utils.read_jsonlines(str(json_file)))
        _, _, *samples = records
        texts = [sample["generated_text"] for sample in samples]
        watermarked_texts.extend(texts)

    return watermarked_texts


def load_eli5(window_size: int):
    jsonl_file = Path("output/robustness/eli5_watermarked_generation.jsonl")
    records = list(utils.read_jsonlines(str(jsonl_file)))
    texts = [
        sample["generation"]
        for sample in records
        if sample["window_size"] == window_size
    ]
    return texts


def main():
    window_size = 7
    batch_size = 16

    watermarked_texts = load_c4_newslike(window_size)
    watermarked_texts.extend(load_eli5(window_size))
    print(
        f"Loaded {len(watermarked_texts)} watermarked texts with window size {window_size}."
    )

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")

    # # Total token number in watermarked texts: 8047179 (window size 4)
    # # Total token number in watermarked texts: 7987582 (window size 7)
    # total_token_num = 0
    # for text in watermarked_texts:
    #     token_ids = tokenizer.encode(text, add_special_tokens=False)
    #     total_token_num += len(token_ids)

    # print(f"Total token number in watermarked texts: {total_token_num}")

    with open("data/server_seed", "r", encoding="utf-8") as f: 
        server_seed = bytes.fromhex(f.read().strip())
    voprf_server = VoprfServer(server_seed)

    for i in range(0, len(watermarked_texts), batch_size):
        batch = watermarked_texts[i : i + batch_size]
        token_ids_list = [
            tokenizer.encode(text, add_special_tokens=False) for text in batch
        ]

        n_grams_list = [
            [
                tuple(token_ids[i - window_size : i + 1])
                for i in range(window_size, len(token_ids))
            ]
            for token_ids in token_ids_list
        ]
        indices = list(
            itertools.accumulate((len(n_grams) for n_grams in n_grams_list), initial=0)
        )
        flatten_msg_inputs = [
            struct.pack(f">{window_size + 1}I", *n_gram)
            for n_grams in n_grams_list
            for n_gram in n_grams
        ]
        flatten_msg_hashes = voprf_server.batch_evaluate(flatten_msg_inputs)
        msg_hashes_list = [
            flatten_msg_hashes[indices[i] : indices[i + 1]]
            for i in range(len(indices) - 1)
        ]


if __name__ == "__main__":
    main()
