import argparse

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from watermark_suite.schemes.upv.watermarking import UPVAdapter
from watermark_suite.schemes.upv.detection import train_private_detector_model

DATASET_PATH_DICT = {
    "c4": "../../data/c4_realnewslike_subset_673.jsonl",
    "eli5": "../../data/eli5",
}


def get_dataset(dataset):
    dataset_path = DATASET_PATH_DICT.get(dataset)
    if dataset == "c4":
        return load_dataset("json", data_files=dataset_path, split="train")
    if dataset == "eli5":
        return load_dataset(dataset_path, split="train")
    raise ValueError(f"Unknown dataset: {dataset}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="c4")

    parser.add_argument("--train_num_samples", type=int, default=10_000)
    parser.add_argument("--output_dir", type=str, default="train_detector_data")

    parser.add_argument("--use_sampling", type=bool, default=True)
    parser.add_argument("--sampling_temp", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=200)

    parser.add_argument("--model_dir", type=str, default="model")
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--bit_number", type=int, default=18)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--n_beams", type=int, default=0)

    parser.add_argument("--z_value_threshold", type=float, default=4.0)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    adapter = UPVAdapter(
        model,
        tokenizer,
        args.model_dir,
        args.window_size,
        args.delta,
        args.gamma,
        args.bit_number,
        args.layers,
        args.n_beams,
    )

    # dataset = get_dataset(args.dataset)
    # adapter.generate_and_save_train_data(args.train_num_samples, args.output_dir)
    # adapter.generate_and_save_test_data(
    #     dataset, args.output_dir, args.sampling_temp, args.max_new_tokens
    # )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_private_detector_model(
        tokenizer,
        args.bit_number,
        args.output_dir,
        args.model_dir,
        args.model_dir,
        args.layers,
        args.z_value_threshold,
    )


if __name__ == "__main__":
    main()
