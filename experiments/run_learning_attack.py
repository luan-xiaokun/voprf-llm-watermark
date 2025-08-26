import re

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.attacks.learning import GreenCache, LearningAdapter
from watermark_suite.schemes import VOWDetector
from watermark_suite.utils import io_utils


def find_most_significant_repetition(text: str) -> tuple[str, int] | None:

    pattern = r"(.+?)\1+"

    best_match = None
    max_len = 0

    for match in re.finditer(pattern, text):

        current_len = len(match.group(0))

        if current_len > max_len:
            max_len = current_len
            best_match = match

    if best_match:
        cycle = best_match.group(1)
        full_repetition = best_match.group(0)

        count = len(full_repetition) // len(cycle)

        return cycle, count

    return None


def analyze_records(records: list[str]) -> list[dict]:
    results = []
    for i, record in enumerate(records):
        repetition_info = find_most_significant_repetition(record)

        if repetition_info:
            cycle, count = repetition_info
            results.append(
                {
                    "record_index": i,
                    "cycle": cycle,
                    "count": count,
                }
            )

    return results


def main():
    green_cache = GreenCache("data/green_cache_w4_g0.5_indexed.parquet")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B")
    model.eval()
    model.to("cuda")

    learning_adapter = LearningAdapter(
        model=model,
        tokenizer=tokenizer,
        window_size=4,
        delta=4.0,
        green_cache=green_cache,
    )

    with open("data/server_seed") as f:
        seed = bytes.fromhex(f.read().strip())
    detector = VOWDetector(tokenizer, seed, gamma=0.5, window_size=4)

    dataset = load_dataset("data/eli5", split="train")
    dataset = dataset.take(500)

    batch_size = 32
    attack_result_filepath = "data/learning_attack_eli5_full.jsonl"
    total = 0

    generated_texts = []
    for batch in tqdm(
        dataset.iter(batch_size=batch_size),
        total=(len(dataset) + batch_size - 1) // batch_size,
    ):
        prompts = batch["question"]

        texts = learning_adapter(
            prompts,
            max_new_tokens=300,
            # suppress_tokens=[tokenizer.eos_token_id],
            do_sample=True,
            top_k=50,
            top_p=0.9,
        )

        detection_results = detector.local_batch_detect(texts)
        indices = [i for i, r in enumerate(detection_results) if r.p_value > 0.04]
        indices = list(range(len(texts)))
        total += len(indices)
        print(
            f"Obtained {len(indices)} negative samples out of {len(prompts)} (total {total})"
        )
        batch_to_save = [{"prompt": prompts[i], "generated": texts[i]} for i in indices]
        if batch_to_save:
            io_utils.write_jsonlines(attack_result_filepath, batch_to_save, "a")

    records = list(io_utils.read_jsonlines("data/learning_attack_eli5_full.jsonl"))
    texts = [r["generated"] for r in records]

    detection_results = detector.local_batch_detect(texts)

    significance_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for sl in significance_levels:
        attack_success_rate = sum(
            r.p_value < sl for r in detection_results if r.p_value < sl
        ) / len(detection_results)
        print(
            f"Attack success rate at significance level {sl:.1e}: {attack_success_rate:.4f}"
        )

    p_values = np.array([r.p_value for r in detection_results])
    print(f"Median p-value: {np.median(p_values):.4e}")

    found_repetitions = analyze_records(texts)

    texts_with_enough_reps = []
    texts_without_reps = []
    for rep, text in zip(found_repetitions, texts):
        if len(rep["cycle"]) > 50:
            texts_with_enough_reps.append(text)
        else:
            texts_without_reps.append(text)

    print(
        f"Found {len(texts_with_enough_reps)} texts with enough repetitions and "
        f"{len(texts_without_reps)} texts without enough repetitions."
    )

    print("No repetitions group:\n" + "=" * 50)
    no_rep_results = detector.local_batch_detect(texts_without_reps)
    p_values = np.array([r.p_value for r in no_rep_results])
    print(f"Median p-value: {np.median(p_values):.4e}")
    print(
        f"Average effective token number: {np.mean([r.effective_token_num for r in no_rep_results]):.2f}"
    )
    for sl in significance_levels:
        attack_success_rate = sum(
            r.p_value < sl for r in no_rep_results if r.p_value < sl
        ) / len(no_rep_results)
        print(
            f"Attack success rate at significance level {sl:.1e}: {attack_success_rate:.4f}"
        )

    print("\nRepetitions group:\n" + "=" * 50)
    rep_results = detector.local_batch_detect(texts_with_enough_reps)
    p_values = np.array([r.p_value for r in rep_results])
    print(f"Median p-value: {np.median(p_values):.4e}")
    print(
        f"Average effective token number: {np.mean([r.effective_token_num for r in rep_results]):.2f}"
    )
    for sl in significance_levels:
        attack_success_rate = sum(
            r.p_value < sl for r in rep_results if r.p_value < sl
        ) / len(rep_results)
        print(
            f"Attack success rate at significance level {sl:.1e}: {attack_success_rate:.4f}"
        )
    print(min(p_values))

    # model = AutoModelForCausalLM.from_pretrained(
    #     "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16
    # )
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    # from datasets import Dataset

    # from watermark_suite.core.metrics import calculate_perplexity

    # dataset = Dataset.from_list(records)

    # ppl = calculate_perplexity(model, tokenizer, dataset, "prompt", "generated")
    # print(f"PPL: {ppl:.2f}")


if __name__ == "__main__":
    main()
