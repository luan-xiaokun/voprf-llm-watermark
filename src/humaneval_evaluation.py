import argparse
import json
import secrets
import sys
from pathlib import Path

import torch
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from voprf_py import VoprfServer

from top_k_adaptive_sampling import TopkAdaptiveLogitsProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from kgw_watermark import WatermarkLogitsProcessor

# --- CONFIGURATION FOR THE FINAL REPRODUCTION ---
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "output/human_eval"
NUM_SAMPLES_PER_TASK = 1
BATCH_SIZE = 64


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task performance.")
    parser.add_argument(
        "-w", "--window_size", type=int, default=7, help="Window size for watermarking"
    )
    parser.add_argument(
        "-d", "--delta", default=2.0, type=float, help="Delta value for watermarking"
    )
    parser.add_argument(
        "-g", "--gamma", default=0.25, type=float, help="Gamma value for watermarking"
    )
    parser.add_argument(
        "--no_watermark",
        action="store_true",
        help="If set, disables watermarking and uses the original model",
    )
    parser.add_argument(
        "--kgw_scheme",
        default="",
        type=str,
        choices=["selfhash", "lefthash"],
        help="KGW seeding scheme to use",
    )
    return parser.parse_args()


# --- Official Cleanup Logic (remains the same and correct) ---
def _clean_python_code_for_sft(code: str) -> str:
    code = code.replace("\r", "")
    if "```python" in code:
        code_start_idx = code.find("```python")
        if code_start_idx != -1:
            code = code[code_start_idx:].replace("```python", "").strip()
            end_idx = code.find("```") if "```" in code else len(code)
            code = code[:end_idx].strip()
    return code


def _truncate_code_at_stopwords(code: str, stop_words: list) -> str:
    min_stop_idx = len(code)
    for stop_word in stop_words:
        stop_index = code.find(stop_word)
        if 0 <= stop_index < min_stop_idx:
            min_stop_idx = stop_index
    return code[:min_stop_idx]


def cleanup_code_official(code: str) -> str:
    code = _clean_python_code_for_sft(code)
    stop_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
    code = _truncate_code_at_stopwords(code, stop_words)
    return code


@torch.inference_mode()
def generate_samples(
    model, tokenizer, prompts, generation_config, batch_size, logits_processor=None
):
    results = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generate (pass@1)"):
        batch_prompts = prompts[i : i + batch_size]

        enc = tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            max_length=3072,
            return_tensors="pt",
        ).to(model.device)

        gen_out = model.generate(
            **enc,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

        # For plain text prompts, we can decode everything and then remove the prompt
        full_texts = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
        clean_texts = [
            text[len(prompt) :] for prompt, text in zip(batch_prompts, full_texts)
        ]
        results.extend(clean_texts)

    return results


def main():
    args = get_args()
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    problems = read_problems()

    prompts = []
    task_ids = []
    for task_id, prob in problems.items():
        # --- THE FINAL, CRITICAL CHANGE: Use the exact plain-text Alpaca-style prompt ---
        original_prompt = prob["prompt"].strip()
        formatted_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Write a program to perform the given task.

Input:
{original_prompt}

### Response:
"""

        prompts.append(formatted_prompt)
        task_ids.append(task_id)

    generation_config = GenerationConfig(
        max_new_tokens=1024,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )

    logits_processor = None
    if not args.no_watermark and not args.kgw_scheme:
        seed = secrets.token_bytes(32)
        voprf_server = VoprfServer(seed)
        logits_processor = LogitsProcessorList(
            [
                TopkAdaptiveLogitsProcessor(
                    window_size=args.window_size,
                    delta=args.delta,
                    gamma=args.gamma,
                    top_k=1,
                    voprf_server=voprf_server,
                )
            ]
        )

        generation_config = GenerationConfig(
            max_new_tokens=1024,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
        )
    elif not args.no_watermark and args.kgw_scheme:
        logits_processor = LogitsProcessorList(
            [
                WatermarkLogitsProcessor(
                    vocab=list(tokenizer.get_vocab().values()),
                    delta=args.delta,
                    gamma=args.gamma,
                    seeding_scheme=args.kgw_scheme,
                )
            ]
        )

    print(
        f"Generating 1 sample for each of {len(prompts)} problems using the final official prompt format..."
    )
    raw_codes = generate_samples(
        model, tokenizer, prompts, generation_config, BATCH_SIZE, logits_processor
    )

    print("Cleaning up generated code with official logic...")
    cleaned_codes = [
        cleanup_code_official(code) for code in tqdm(raw_codes, desc="Cleaning")
    ]

    samples = []
    for task_id, code in zip(task_ids, cleaned_codes):
        samples.append({"task_id": task_id, "completion": code})

    samples_file_name = "samples.jsonl"
    if not args.no_watermark and not args.kgw_scheme:
        samples_file_name = (
            f"samples_w{args.window_size}_d{args.delta}_g{args.gamma}.jsonl"
        )
    elif not args.no_watermark and args.kgw_scheme:
        samples_file_name = (
            f"kgw_{args.kgw_scheme}_samples_d{args.delta}_g{args.gamma}.jsonl"
        )
    samples_file = output_dir / samples_file_name
    write_jsonl(str(samples_file), samples)

    print(f"\nRunning evaluation for k=[1]...")
    results = evaluate_functional_correctness(str(samples_file), k=[1])

    evaluation_file_name = "evaluation_results.json"
    if not args.no_watermark and not args.kgw_scheme:
        evaluation_file_name = (
            f"evaluation_results_w{args.window_size}_d{args.delta}_g{args.gamma}.json"
        )
    elif not args.no_watermark and args.kgw_scheme:
        evaluation_file_name = (
            f"kgw_{args.kgw_scheme}_evaluation_results_d{args.delta}_g{args.gamma}.json"
        )
    results_file = output_dir / evaluation_file_name
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation finished. Final Results:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
