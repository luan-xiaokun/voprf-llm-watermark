import tempfile

import torch
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness
from tqdm import tqdm

from ..schemes.adapter import WatermarkAdapter

MAX_NEW_TOKENS = 1024
PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.

### Instruction:
Write a program to perform the given task.

Input:
{prompt}

### Response:
"""


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
def evaluate_human_eval_benchmark(
    adapter: WatermarkAdapter,
    batch_size: int,
    samples_file: str | None = None,
) -> dict:
    problems = read_problems()

    prompts = []
    task_ids = []
    samples = []

    for task_id, prob in problems.items():
        original_prompt = prob["prompt"].strip()
        formatted_prompt = PROMPT_TEMPLATE.format(prompt=original_prompt)

        prompts.append(formatted_prompt)
        task_ids.append(task_id)

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generate (pass@1)"):
        batch_prompts = prompts[i : i + batch_size]

        generated_texts = adapter(
            prompts=batch_prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=adapter.tokenizer.eos_token_id,
        )

        samples.extend(generated_texts)

    cleaned_codes = [cleanup_code_official(code) for code in samples]
    cleaned_samples = []
    for task_id, code in zip(task_ids, cleaned_codes):
        cleaned_samples.append({"task_id": task_id, "completion": code})

    print(f"Running evaluation for k=[1]...")
    if samples_file:
        write_jsonl(samples_file, cleaned_samples)
        results = evaluate_functional_correctness(samples_file, k=[1])
    else:
        # use temp file for evaluation
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            write_jsonl(temp_file.name, cleaned_samples)
            results = evaluate_functional_correctness(temp_file.name, k=[1])

    print(f"Evaluation results: {results}")

    return results
