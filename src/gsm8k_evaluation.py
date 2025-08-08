import argparse
import json
import re
import secrets
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList
from voprf_py import VoprfServer

from top_k_adaptive_sampling import TopkAdaptiveLogitsProcessor

sys.path.append(str(Path(__file__).resolve().parent.parent))
from kgw_watermark import WatermarkLogitsProcessor

# --- Configuration ---
MODEL_PATH = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "output/gsm8k"
NUM_SHOTS = 4
BATCH_SIZE = 32
MAX_NEW_TOKENS = 1024
STOP_STRINGS = [
    "USER:",
    "ASSISTANT:",
    "### Instruction:",
    "Response:",
    "\n\nProblem",
    "\nProblem",
    "Problem:",
    "<|endoftext|>",
    "####",
]

# =============================================================================
# 1. Prompt Construction: Replicating the official prompt logic
#    (Based on prompt_utils.py and the 4-shot examples for gsm8k)
# =============================================================================

# Hardcoded 4-shot examples for GSM8K from the official repository
FEW_SHOT_EXAMPLES = [
    (
        "Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?",
        "Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4",
    ),
    (
        "Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?",
        "Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points.\nThe answer is 201",
    ),
    (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
        "Natalia sold 48/2 = 24 clips in May. Natalia sold 48+24 = 72 clips altogether in April and May. The answer is 72",
    ),
    (
        "Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?",
        "When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items.\nThe answer is 140",
    ),
]


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate task performance.")
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=7,
        choices=[4, 7],
        help="Window size for watermarking",
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
        choices=["lefthash", "selfhash"],
        help="KGW scheme to use for watermarking",
    )
    return parser.parse_args()


def build_gsm8k_prompt(query: str, num_shots: int) -> str:
    """Builds a few-shot CoT prompt based on the official 'short' format."""
    # System instruction
    header = "You are supposed to provide a solution to a given problem.\n\n"

    # Few-shot examples
    demo_prompts = []
    for q, a in FEW_SHOT_EXAMPLES[:num_shots]:
        demo_prompts.append(f"Problem:\n{q}\nSolution:\n{a}")

    # New query
    test_prompt = f"Problem:\n{query}\nSolution:\n"

    return header + "\n\n".join(demo_prompts) + "\n\n" + test_prompt


# =============================================================================
# 2. Answer Extraction: Replicating the official cleaning logic
#    (Based on utils.py: answer_clean, delete_extra_zero)
# =============================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_gsm8k_answer(pred_str: str) -> str:
    """
    Extracts the numerical answer from the prediction string,
    replicating the logic from the official scripts.
    """
    # 1. Use trigger phrases to locate the answer section.
    trigger_phrases = ["The answer is:", "The answer is", "the answer is", "####"]
    # Use a regex to split by any of the trigger phrases, ignoring case
    parts = re.split(
        "|".join(re.escape(p) for p in trigger_phrases), pred_str, flags=re.IGNORECASE
    )
    if len(parts) > 1:
        pred_str = parts[-1]  # Take the part after the last trigger

    # 2. Find all number-like substrings (integers, decimals, fractions)
    # This regex is from the official `answer_clean` function for gsm8k
    number_regex = r"-?\d+/?\.?\d*"
    preds = re.findall(number_regex, pred_str.replace(",", ""))

    if not preds:
        return ""  # Return empty if no number found

    # 3. Choose the last number found
    answer_str = preds[-1]

    # 4. Normalize the number string (replicating `delete_extra_zero`)
    try:
        # Convert to float to handle decimals
        num = float(answer_str)
        # If it's an integer, return as integer string (e.g., 5.0 -> 5)
        if num.is_integer():
            return str(int(num))
        else:
            # For floats, just return their string representation
            return str(num)
    except ValueError:
        # If it's a fraction or something else that float() can't handle,
        # return the raw extracted string.
        return answer_str


# =============================================================================
# 3. Main Evaluation Script
# =============================================================================


@torch.inference_mode()
def main():
    args = get_args()

    # --- Setup ---
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stop tokens are handled by setting eos_token_id in GenerationConfig
    # This is a simplification from the official vLLM script but effective.
    # The official repo uses a long list of text-based stop tokens.
    # We will rely on our robust answer extraction instead.

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=MAX_NEW_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=STOP_STRINGS,
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
            do_sample=True,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=STOP_STRINGS,
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

    # --- Load Dataset ---
    dataset = load_dataset("gsm8k", "main", split="test")

    # --- Evaluation Loop ---
    predictions = []
    correct_count = 0
    total_count = 0

    pbar = tqdm(
        range(0, len(dataset), BATCH_SIZE), desc=f"Evaluating GSM8K ({MODEL_PATH})"
    )
    for i in pbar:
        batch = dataset[i : i + BATCH_SIZE]

        # Prepare prompts for the batch
        batch_prompts = [build_gsm8k_prompt(q, NUM_SHOTS) for q in batch["question"]]

        # Tokenize and generate
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            model.device
        )
        outputs = model.generate(
            **inputs,
            tokenizer=tokenizer,
            generation_config=generation_config,
            logits_processor=logits_processor,
        )

        # Decode and extract answers
        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, full_output in enumerate(decoded_outputs):
            # Remove the prompt part to get the generation
            generation = full_output[len(batch_prompts[j]) :]

            # Extract numerical answer from model's generation
            pred_answer_str = extract_gsm8k_answer(generation)

            # Extract ground truth answer
            gt_answer_str = extract_gsm8k_answer(batch["answer"][j])

            # Compare and update counts
            is_correct = pred_answer_str == gt_answer_str
            if is_correct:
                correct_count += 1
            total_count += 1

            # Store detailed results
            predictions.append(
                {
                    "id": i + j,
                    "question": batch["question"][j],
                    "prompt": batch_prompts[j],
                    "generation": generation,
                    "prediction": pred_answer_str,
                    "ground_truth": gt_answer_str,
                    "is_correct": is_correct,
                }
            )

            # Update progress bar
            accuracy = (correct_count / total_count) * 100
            pbar.set_postfix({"Accuracy": f"{accuracy:.2f}%"})

    # --- Save Results ---
    final_accuracy = (correct_count / total_count) * 100
    print(f"\nFinal GSM8K Accuracy ({NUM_SHOTS}-shot): {final_accuracy:.2f}%")

    # Save detailed predictions to a file
    results_file_name = "gsm8k_predictions.jsonl"
    if not args.no_watermark and not args.kgw_scheme:
        results_file_name = (
            f"gsm8k_predictions_w{args.window_size}_d{args.delta}_g{args.gamma}.jsonl"
        )
    elif not args.no_watermark and args.kgw_scheme:
        results_file_name = (
            f"kgw_{args.kgw_scheme}_gsm8k_predictions_d{args.delta}_g{args.gamma}.jsonl"
        )
    results_file = output_dir / results_file_name
    with open(results_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Detailed results saved to {results_file}")


if __name__ == "__main__":
    main()
