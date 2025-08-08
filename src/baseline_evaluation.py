import argparse
import sys
from pathlib import Path
import time

import torch
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from scipy import stats

sys.path.append(str(Path(__file__).resolve().parent.parent))

import utils
from kgw_watermark import WatermarkDetector, WatermarkLogitsProcessor

DATA_FILE_PATH = "data/c4_realnewslike_subset_673.jsonl"


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate watermarked text using KGW baseline method"
    )
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
        default="./output/kgw_baseline",
        help="Directory to save the generated watermarked text",
    )
    parser.add_argument(
        "--significance_level",
        type=float,
        default=1e-6,
        help="Significance level for watermark detection",
    )
    parser.add_argument(
        "--delta", type=float, default=2.0, help="Delta value for watermarking"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.25, help="Gamma value for watermarking"
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="selfhash",
        choices=["selfhash", "lefthash", "minhash", "skipgram"],
        help="Seeding scheme for watermarking",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=210,
        help="Maximum number of tokens to generate in response to a prompt",
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

    delta = args.delta
    gamma = args.gamma
    hash_scheme = args.seeding_scheme

    return f"kgw_{hash_scheme}_{model_name}_{sampling_method}_d{delta}_g{gamma}.jsonl"


def generate(args):
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

    print(f"Output will be saved to {output_file_path}")
    print("Generation arguments:")
    for key, value in vars(args).items():
        print(f"- {key}: {value}")
    utils.write_jsonlines(output_file_path, vars(args))

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
    dataset = dataset.select(range(args.num))

    logits_processor = WatermarkLogitsProcessor(
        vocab=list(tokenizer.get_vocab().values()),
        delta=args.delta,
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
    )

    z_threshold = stats.norm.ppf(1 - args.significance_level)
    watermark_detector = WatermarkDetector(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=args.gamma,
        seeding_scheme=args.seeding_scheme,
        device=model.device,
        tokenizer=tokenizer,
        z_threshold=z_threshold,
        normalizers=[],
        ignore_repeated_ngrams=True,
    )

    all_generated_texts = []
    generation_total_time = 0.0
    generated_total_tokens = 0
    for batch in (pbar := tqdm.tqdm(dataset.batch(args.batch_size))):
        prompts = batch["prompt_text"]
        with torch.no_grad():
            generation_start_time = time.perf_counter()

            batch_encoding = tokenizer(prompts, return_tensors="pt", padding=True)
            input_ids: torch.LongTensor = batch_encoding["input_ids"]
            attention_mask: torch.LongTensor = batch_encoding["attention_mask"]
            inputs = {
                "input_ids": input_ids.to(model.device),
                "attention_mask": attention_mask.to(model.device),
            }
            generation_config = GenerationConfig(
                max_new_tokens=args.max_tokens,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                suppress_tokens=[tokenizer.eos_token_id] if args.suppress_eos else None,
                pad_token_id=tokenizer.pad_token_id,
            )

            output_ids = model.generate(
                **inputs,
                generation_config=generation_config,
                tokenizer=tokenizer,
                logits_processor=LogitsProcessorList([logits_processor]),
            )
            # for decoder-only models, we need to slice the output_ids
            generated_ids = output_ids[:, input_ids.shape[1] :]
            generated_texts = [
                tokenizer.decode(ids, skip_special_tokens=True)
                for ids in generated_ids.tolist()
            ]

            generation_total_time += time.perf_counter() - generation_start_time

        detection_results = [
            watermark_detector.detect(text) for text in generated_texts
        ]
        # {'num_tokens_scored': 206, 'num_green_tokens': 114, 'green_fraction': 0.5533980582524272
        # 'z_score': 10.05647483386412, 'p_value': np.float64(4.301179336252213e-24), 'z_score_at_T': tensor
        # 'prediction': np.True_, 'confidence'

        generation_results = [
            {
                "index": idx,
                "original_index": orig_idx,
                "prompt_text": prompt,
                "completion_text": completion,
                "generated_text": gen_text,
                "total_token_num": det_res["num_tokens_scored"],
                "green_token_num": det_res["num_green_tokens"],
                "green_fraction": det_res["green_fraction"],
                "z_score": det_res["z_score"],
                "p_value": float(det_res["p_value"]),
                "prediction": bool(det_res["prediction"]),
                "confidence": float(det_res.get("confidence", 0.0)),
                "z_score_at_T": det_res.get(
                    "z_score_at_T", torch.tensor([0.0])
                ).tolist(),
            }
            for idx, orig_idx, prompt, completion, gen_text, det_res in zip(
                batch["index"],
                batch["original_index"],
                batch["prompt_text"],
                batch["completion_text"],
                generated_texts,
                detection_results,
            )
        ]
        utils.write_jsonlines(output_file_path, generation_results, mode="a")
        all_generated_texts.extend(generated_texts)

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
    generate(args)
