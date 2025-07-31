"""This script prepares a subset of the C4 dataset (realnewslike configuration)
where each example is split into a prompt and a completion. We use a fixed prompt
length and filter out examples with a completion length below a specified minimum.
The script supports both non-streaming and streaming methods for creating the subset,
depending on the available RAM. The final subset is saved in JSONL format, which can
be easily loaded later for training or evaluation purposes.
"""

import json
import os
import random
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DATASET_NAME = "allenai/c4"
CONFIG_NAME = "realnewslike"
SPLIT_NAME = "train"
NUM_SAMPLES = 1000
RANDOM_SEED = 46
DATA_DIR = "data"
OUTPUT_FILE = "c4_realnewslike_subset_{size}.jsonl"
TOTAL_SAMPLE_NUM = 13_799_838
TOKENIZER_PATH = "models/Qwen/Qwen2.5-3B"
PROMPT_LENGTH = 60
MIN_COMPLETION_LENGTH = 200


def create_subset_non_streaming() -> Dataset:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    dataset = load_dataset(DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME, num_proc=12)

    total_samples = len(dataset)
    random_indices = np.random.choice(total_samples, size=NUM_SAMPLES, replace=False)
    subset = dataset.select(random_indices)

    def add_indices(example, idx):
        example["index"] = idx
        example["original_index"] = random_indices[idx]
        return example

    subset_with_indices = subset.map(
        add_indices,
        with_indices=True,
        batched=False,
        num_proc=12,
    )

    return subset_with_indices


def create_subset_reservoir_sampling() -> Dataset:
    random.seed(RANDOM_SEED)
    streaming_dataset = load_dataset(
        DATASET_NAME, CONFIG_NAME, split=SPLIT_NAME, streaming=True
    )

    reservoir = []
    for i, sample in tqdm(
        enumerate(streaming_dataset), total=TOTAL_SAMPLE_NUM, desc="Sampling"
    ):
        sample_with_index = {**sample, "original_index": i}
        if i < NUM_SAMPLES:
            reservoir.append(sample_with_index)
        else:
            j = random.randint(0, i)
            if j < NUM_SAMPLES:
                reservoir[j] = sample_with_index

    reservoir.sort(key=lambda x: x["original_index"])
    for i, sample in enumerate(reservoir):
        sample["index"] = i

    subset = Dataset.from_list(reservoir)

    return subset


def create_subset() -> Dataset:
    print(f"Creating a subset of {DATASET_NAME} ({CONFIG_NAME}) dataset...")
    print(f"Subset size: {NUM_SAMPLES}")
    print(f"Random seed: {RANDOM_SEED}")

    available_ram = (
        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0**3)
    )
    if available_ram > 16:
        return create_subset_non_streaming()
    return create_subset_reservoir_sampling()


def truncate_text_for_prompt_and_completion(
    example: dict,
    tokenizer: PreTrainedTokenizerBase,
    prompt_length: int,
) -> dict:
    assert "text" in example, "Expects 'text' key in the example."
    assert prompt_length > 0, "Prompt length must be greater than 0."

    input_ids = tokenizer.encode(example["text"])
    prompt_input_ids = input_ids[:prompt_length]
    completion_input_ids = input_ids[prompt_length:]

    prompt_text = tokenizer.decode(prompt_input_ids, skip_special_tokens=True)
    completion_text = tokenizer.decode(completion_input_ids, skip_special_tokens=True)
    example.update(
        {
            "prompt_text": prompt_text,
            "completion_text": completion_text,
            "completion_length": len(completion_input_ids),
        }
    )

    return example


def main():
    subset = create_subset()
    print(f"Using tokenizer from {TOKENIZER_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    print("Truncating text for prompt and completion")
    truncated_subset = subset.map(
        lambda example: truncate_text_for_prompt_and_completion(
            example, tokenizer, PROMPT_LENGTH
        ),
        num_proc=12,
    )
    print(f"Filtering examples with minimum completion length {MIN_COMPLETION_LENGTH}")
    filtered_subset = truncated_subset.filter(
        lambda example: example.get("completion_length", 0) >= MIN_COMPLETION_LENGTH,
        num_proc=12,
    )

    size = len(filtered_subset)
    print(f"Filtered subset size: {size}")

    output_file_path = Path(DATA_DIR) / OUTPUT_FILE.format(size=size)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving subset to {output_file_path}...")
    with open(output_file_path, "w", encoding="utf-8") as f:
        for sample in filtered_subset:
            del sample["completion_length"]
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print("Subset saved successfully.")
    print("\nTo load the subset, use the following code:")
    print("from datasets import load_dataset")
    print(f"dataset = load_dataset('json', data_files='{output_file_path}')")
    print("print(dataset['train'][0])")


if __name__ == "__main__":
    main()
