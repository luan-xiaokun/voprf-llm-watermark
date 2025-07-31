import argparse
import secrets
import time
from pathlib import Path

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import utils
from watermark import WatermarkAdapter

DATA_FILE_PATH = "data/c4_realnewslike_subset_673.jsonl"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate watermarked texts.")
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=500,
        help="Number of examples to generate watermarked text for",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-3B",
        help="Path to the pre-trained model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Directory to save the generated watermarked text",
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
        "--max_tokens",
        type=int,
        default=210,
        help="Maximum number of tokens to generate in response to a prompt",
    )
    parser.add_argument(
        "--server_seed",
        type=str,
        default=None,
        help="File path to the server seed for watermarking",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for generation"
    )
    parser.add_argument(
        "--do_sample", action="store_true", help="Whether to use sampling"
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams for beam search"
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
        "--overwrite",
        action="store_true",
        help="If set, will overwrite existing output files",
    )
    return parser.parse_args()


def get_output_file_name(args) -> str:
    model_name = args.model_path.split("/")[-1]
    window_size = args.window_size
    delta = args.delta
    gamma = args.gamma
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

    return f"{model_name}_{sampling_method}_w{window_size}_d{delta}_g{gamma}.jsonl"


def main(args):
    # prepare output file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_name = get_output_file_name(args)
    output_file_path = output_dir / output_file_name
    if output_file_path.exists():
        if not args.overwrite:
            print(
                f"Output file {output_file_path} already exists. Skipping generation."
            )
            return
        overwrite = input(
            f"Output file {output_file_path} already exists. Overwrite? (y/n): "
        )
        if overwrite.strip().lower() != "y":
            print("Exiting without overwriting the file.")
            return

    # print and save generation arguments
    print(f"Output will be saved to {output_file_path}")
    print("Generation arguments:")
    for key, value in vars(args).items():
        print(f"- {key}: {value}")
    utils.write_jsonlines(output_file_path, vars(args))

    # load model, tokenizer, and dataset
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_size="left", use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files=DATA_FILE_PATH, split="train")
    dataset = dataset.select(range(args.num))  # limit to the first `num` examples

    # set up watermark adapter and save the server seed
    if not args.server_seed:
        print("Generating a new server seed for watermarking")
        server_seed = secrets.token_bytes(32)
    else:
        print(f"Loading server seed from {args.server_seed}")
        with open(args.server_seed, "r", encoding="utf-8") as f:
            server_seed = bytes.fromhex(f.read().strip())
    adapter = WatermarkAdapter(
        model=model,
        tokenizer=tokenizer,
        window_size=args.window_size,
        delta=args.delta,
        gamma=args.gamma,
        seed=server_seed,
    )
    utils.write_jsonlines(
        output_file_path, {"server_seed": server_seed.hex()}, mode="a"
    )

    # iterate over the dataset and generate watermarked text
    generation_total_time = 0.0
    generated_total_tokens = 0
    for batch in (pbar := tqdm.tqdm(dataset.batch(args.batch_size))):
        prompts = batch["prompt_text"]
        with torch.no_grad():
            generation_start_time = time.time()
            generated_texts = adapter(
                prompts=prompts,
                max_new_tokens=args.max_tokens,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
                suppress_tokens=[tokenizer.eos_token_id] if args.suppress_eos else None,
                pad_token_id=tokenizer.eos_token_id,
            )
            generation_total_time += time.time() - generation_start_time

        generation_results = [
            {
                "index": idx,
                "original_index": orig_idx,
                "prompt_text": prompt,
                "completion_text": completion,
                "generated_text": gen_text,
            }
            for idx, orig_idx, prompt, completion, gen_text in zip(
                batch["index"],
                batch["original_index"],
                batch["prompt_text"],
                batch["completion_text"],
                generated_texts,
            )
        ]
        utils.write_jsonlines(output_file_path, generation_results, mode="a")

        generated_total_tokens += sum(
            len(tokenizer.encode(text, add_special_tokens=False))
            for text in generated_texts
        )
        avg_tps = generated_total_tokens / generation_total_time
        pbar.set_description(f"Avg tokens per example: {avg_tps:.2f} t/s")

    final_avg_tps = generated_total_tokens / generation_total_time
    print(f"Generated {len(dataset)} completions in total")
    print(f"Total generation time: {generation_total_time:.2f} seconds")
    print(f"Average tokens per second: {final_avg_tps:.2f} t/s")


if __name__ == "__main__":
    args = get_args()
    main(args)
