import json
import re

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..schemes.adapter import WatermarkAdapter

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
def evaluate_gsm8k_benchmark(
    adapter: WatermarkAdapter,
    batch_size: int,
    num_shots: int = 4,
    samples_file: str | None = None,
) -> float:
    dataset = load_dataset("gsm8k", "main", split="test")

    predictions = []
    correct_count = 0
    total_count = 0

    pbar = tqdm(range(0, len(dataset), batch_size), desc=f"Evaluating GSM8K")
    for i in pbar:
        batch = dataset[i : i + batch_size]
        batch_prompts = [build_gsm8k_prompt(q, num_shots) for q in batch["question"]]

        generated_texts = adapter(
            prompts=batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=adapter.tokenizer.eos_token_id,
            stop_strings=STOP_STRINGS,
        )

        for j, gen in enumerate(generated_texts):
            pred_answer = extract_gsm8k_answer(gen)
            true_answer = extract_gsm8k_answer(batch["answer"][j])

            is_correct = pred_answer == true_answer
            correct_count += is_correct
            total_count += 1

            predictions.append(
                {
                    "id": i + j,
                    "question": batch["question"][j],
                    "prompt": batch_prompts[j],
                    "generation": gen,
                    "prediction": pred_answer,
                    "ground_truth": true_answer,
                    "is_correct": is_correct,
                }
            )

            accuracy = (correct_count / total_count) * 100
            pbar.set_postfix({"Accuracy": f"{accuracy:.2f}%"})

    if samples_file:
        with open(samples_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
        print(f"Detailed results saved to {samples_file}")

    final_accuracy = (correct_count / total_count) * 100
    print(f"Final GSM8K Accuracy ({num_shots}-shot): {final_accuracy:.2f}%")

    return final_accuracy
