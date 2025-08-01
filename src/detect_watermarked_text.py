"""This script detects watermarks in generated texts without using FHE.
It reads the watermarked text from a specified directory, derives the watermarking key,
and checks each text for the presence of a watermark using the specified parameters.
Pass '--no_watermark' to check the unwatermarked completions.
"""

import argparse
import math

import torch
import tqdm
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

import utils
from detection import local_detect_batch_text, client_detect_batch_text


def get_args():
    parser = argparse.ArgumentParser(description="Detect watermark in texts.")
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Path to the pre-trained tokenizer directory",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./output",
        help="Directory to load the generated watermarked text",
    )
    parser.add_argument(
        "--significance_level",
        type=float,
        default=0.01,
        help="Significance level for watermark detection",
    )
    parser.add_argument(
        "--window_size", type=int, default=7, help="Window size for watermarking"
    )
    parser.add_argument(
        "--delta", type=float, default=2.0, help="Delta value for watermarking"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.25, help="Gamma value for watermarking"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Whether to use sampling"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--top_k", type=int, default=None, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--eval_model_path",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Path to the pre-trained model directory for evaluation",
    )
    parser.add_argument(
        "--ppl",
        action="store_true",
        help="Whether to compute perplexity for the generated text",
    )
    parser.add_argument(
        "--no_watermark",
        action="store_true",
        help="Whether to check the unwatermarked text",
    )
    return parser.parse_args()


def get_input_file_name(args) -> str:
    model_name = args.tokenizer_path.split("/")[-1]

    do_sample = args.do_sample
    num_beams = args.num_beams
    sampling_method = ""
    if num_beams == 1 and not do_sample:
        sampling_method = "greedy"
        args.temperature = None
        args.top_p = None
        args.top_k = None
    elif num_beams > 1:
        sampling_method = f"beam{num_beams}"
    elif do_sample:
        sampling_method = "multinomial"
        if args.top_k is not None and args.top_k > 0:
            sampling_method += f"-top{args.top_k}"
    else:
        raise ValueError(
            "Invalid sampling method: must be either greedy, beam, or multinomial"
        )

    if args.no_watermark:
        return f"{model_name}_{sampling_method}_no_watermark.jsonl"

    window_size = args.window_size
    delta = args.delta
    gamma = args.gamma

    return f"{model_name}_{sampling_method}_w{window_size}_d{delta}_g{gamma}.jsonl"


def calculate_conditional_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    prompt_column: str,
    target_column: str,
    batch_size: int = 8,
    max_length: int = 2048,
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    def preprocess(examples):
        prompts = examples[prompt_column]
        targets = examples[target_column]
        full_texts = [p + t for p, t in zip(prompts, targets)]

        full_tokenized = tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        prompt_tokenized = tokenizer(prompts, add_special_tokens=False)

        labels = []
        for i in range(len(full_tokenized["input_ids"])):
            prompt_len = len(prompt_tokenized["input_ids"][i])

            label = list(full_tokenized["input_ids"][i])

            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)
        full_tokenized["labels"] = torch.tensor(labels)

        return full_tokenized

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=DefaultDataCollator(),
        shuffle=False,
    )

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm.tqdm(dataloader, desc="Calculating conditional perplexity"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            num_target_tokens = (batch["labels"] != -100).sum().item()
            total_loss += outputs.loss.item() * num_target_tokens
            total_tokens += num_target_tokens

    if total_tokens == 0:
        return float("inf")

    average_loss = total_loss / total_tokens
    perplexity = math.exp(average_loss)

    return perplexity


def main(args):
    input_dir = args.input_dir
    input_file_name = get_input_file_name(args)
    input_file_path = f"{input_dir}/{input_file_name}"
    print(f"Loading text from {input_file_path}")

    generation_records = list(utils.read_jsonlines(input_file_path))
    gen_args, server_seed_dict, *generation_records = generation_records
    server_seed = bytes.fromhex(server_seed_dict["server_seed"])
    for key in ["window_size", "delta", "gamma", "do_sample", "num_beams", "top_k"]:
        assert getattr(args, key) == gen_args.get(key), (
            f"Argument {key} mismatch: expected {gen_args.get(key)}, "
            f"got {getattr(args, key)}"
        )

    window_size = args.window_size
    significance_level = args.significance_level

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    target_column = "generated_text"
    detection_results = local_detect_batch_text(
        texts=[record[target_column] for record in generation_records],
        tokenizer=tokenizer,
        window_size=window_size,
        gamma=args.gamma,
        seed=server_seed,
    )

    example_num = len(detection_results)
    detected_num = sum(r.p_value < significance_level for r in detection_results)
    avg_green_token_num = (
        sum(r.green_token_num for r in detection_results) / example_num
    )
    avg_total_token_num = (
        sum(r.total_token_num for r in detection_results) / example_num
    )
    avg_z_score = sum(r.z_score for r in detection_results) / example_num
    check_object = "unwatermarked" if args.no_watermark else "watermarked"
    print(f"Watermark detection results for {input_file_name}:")
    print(f"Checking {check_object} completions")
    print(f"Detected: {detected_num} / {example_num}")
    print(f"Average green token number: {avg_green_token_num:.2f}")
    print(f"Average total token number: {avg_total_token_num:.2f}")
    print(f"Average z-score: {avg_z_score:.2f}")

    for record, result in zip(generation_records, detection_results):
        if result.p_value < significance_level:
            print(
                f"Example {record['index']} ({record['original_index']}): "
                f"Not detected (p-value: {result.p_value:.4f}, z-score: {result.z_score:.2f})"
            )
            print(
                f"- Green token number: {result.green_token_num} / {result.total_token_num}"
            )

    if args.ppl:
        eval_model = AutoModelForCausalLM.from_pretrained(
            args.eval_model_path, torch_dtype=torch.bfloat16
        )
        eval_tokenizer = AutoTokenizer.from_pretrained(args.eval_model_path)
        dataset = Dataset.from_list(generation_records)
        ppl = calculate_conditional_perplexity(
            model=eval_model,
            tokenizer=eval_tokenizer,
            dataset=dataset,
            prompt_column="prompt_text",
            target_column=target_column,
        )
        print(f"Average perplexity: {ppl:.2f}")


if __name__ == "__main__":
    args = get_args()
    main(args)
