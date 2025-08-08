import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from voprf_py import VoprfServer

import utils
from top_k_adaptive_sampling import TopkAdaptiveLogitsProcessor
from detection import local_detect_batch_text

MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "output/robustness"
DATA_PATH = "data/eli5"
BATCH_SIZE = 128
MAX_NEW_TOKENS = 1024
STOP_STRINGS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction:",
    "\n\nQuestion",
    "\nQuestion",
    "Question:",
    "<|endoftext|>",
    "####",
]
TOKEN_NUMS = [200, 400, 600, 800, 1000]
WINDOW_SIZES = [4, 7]


prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
You are a friendly and patient teacher. Your task is to answer the following question. Explain it like I'm five years old.
The explanation should be simple, detailed, and around {word_num} words long.

### Input:
{question}

### Response:
"""


def format_prompt(question: str, token_num: int) -> str:
    word_num = int(token_num * 3 / 4)  # Approximate word count based on token count
    return prompt_template.format(word_num=word_num, question=question)


def get_args():
    parser = argparse.ArgumentParser(
        description="Generate texts for robustness evaluation."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=MODEL_PATH,
        help="Path to the pre-trained model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory to save the evaluation results",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help="Path to the dataset for evaluation",
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE, help="Batch size for generation"
    )
    parser.add_argument(
        "-n", "--num", type=int, default=1000, help="Number of examples to generate"
    )
    return parser.parse_args()


def main():
    args = get_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "eli5_watermarked_generation.jsonl"

    print(f"Writing outputs to {output_file}")

    dataset = load_dataset(args.data_path, split="train")
    dataset = dataset.select(range(args.num))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    with open("data/server_seed") as seed_file:
        seed = bytes.fromhex(seed_file.read().strip())
    voprf_server = VoprfServer(seed)

    for token_num in TOKEN_NUMS:
        for window_size in [4, 7]:
            answers = []
            generation_config = GenerationConfig(
                do_sample=True,
                max_new_tokens=token_num + 25,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stop_strings=STOP_STRINGS,
                top_p=0.9,
                temperature=0.7,
            )
            logits_processor = LogitsProcessorList(
                [
                    TopkAdaptiveLogitsProcessor(
                        window_size=window_size,
                        delta=2.0,
                        gamma=0.25,
                        top_k=50,
                        voprf_server=voprf_server,
                    )
                ]
            )
            p_bar = tqdm(
                range(0, len(dataset), args.batch_size),
                desc=f"Token num {token_num}, window size {window_size}",
            )
            for i in p_bar:
                batch = dataset[i : i + args.batch_size]
                prompts = [format_prompt(q, token_num) for q in batch["question"]]

                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                    model.device
                )
                outputs = model.generate(
                    **inputs,
                    tokenizer=tokenizer,
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                )

                input_token_len = inputs.input_ids.shape[1]
                generated_tokens = outputs[:, input_token_len:]

                decoded_outputs = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )

                results = []
                for j, answer in enumerate(decoded_outputs):
                    answers.append(answer)
                    results.append(
                        {
                            "question": batch["question"][j],
                            "answer": batch["answer"][j],
                            "generation": answer,
                            "expected_token_num": token_num,
                            "window_size": window_size,
                        }
                    )

                utils.write_jsonlines(output_file, results, mode="a")
                p_bar.update(1)

            detection_results = local_detect_batch_text(
                answers,
                tokenizer,
                window_size=window_size,
                gamma=0.25,
                seed=seed,
            )
            detected_num = sum(r.p_value < 1e-5 for r in detection_results)
            tpr = detected_num / len(detection_results)
            print(f"TPR @1e-5 TPR: {tpr:.4f}")


if __name__ == "__main__":
    main()
