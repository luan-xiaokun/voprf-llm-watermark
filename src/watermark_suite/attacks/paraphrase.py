import os
import json
import time
import threading
from queue import Queue
from openai import OpenAI
from tqdm import tqdm

 
WRITE_BUFFER_SIZE = 50  # Buffer size, write to file every 50 processed results

# DeepSeek API configuration
DEEPSEEK_API_KEY = ""
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"

# API call retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5


# Added explicit instruction at the end to require only plain text output
SYSTEM_PROMPT = """As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length.

Your output must be only the rewritten text itself, without any introductory phrases like "Generated:", concluding remarks, or any other conversational content."""


def set_deepseek_api_key(api_key: str | None = None) -> None:
    global DEEPSEEK_API_KEY
    if api_key is not None:
        DEEPSEEK_API_KEY = api_key
    elif "DEEPSEEK_API_KEY" in os.environ:
        DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]


def load_processed_indices(filename: str) -> set:
    """Read existing output file and return a set containing all processed line indices for resume capability."""
    processed_indices = set()
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "index" in data:
                        processed_indices.add(data["index"])
                except json.JSONDecodeError:
                    continue  # Skip corrupted lines
    except FileNotFoundError:
        pass  # If file doesn't exist, it means first run, return empty set
    return processed_indices


def paraphrase_text(client: OpenAI, text_to_paraphrase: str):
    """Call DeepSeek API to paraphrase a single text segment (essentially the same as previous version)."""
    if not text_to_paraphrase or not text_to_paraphrase.strip():
        return ""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text_to_paraphrase},
                ],
                temperature=0.7,
                max_tokens=2048,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            # In multi-threading, use tqdm.write for safe printing to avoid breaking progress bar
            tqdm.write(
                f"[Error] API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                tqdm.write(
                    f"[Critical Error] Still failed after {MAX_RETRIES} retries."
                )
                return None


def worker(
    task_queue: Queue,
    results_queue: Queue,
    client: OpenAI,
):
    """Worker thread function: get tasks from task queue, process them, and put results into result queue."""
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value, indicating no more tasks
            break

        index, data = task
        paraphrased_generation = paraphrase_text(client, data)

        results_queue.put((index, paraphrased_generation))
        task_queue.task_done()


def paraphrase_texts_by_deepseek(
    samples: list[str],
    api_key: str | None = None,
    num_workers: int = 20,
) -> list[str | None]:
    set_deepseek_api_key(api_key)

    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY variable not set.")

    num_samples = len(samples)
    print(f"Number of tasks to process: {num_samples}")

    task_queue = Queue()
    results_queue = Queue()
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_BASE)

    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=worker,
            args=(task_queue, results_queue, client),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for i, sample in enumerate(samples):
        task_queue.put((i, sample))

    final_results: list[str | None] = [None] * num_samples

    print(f"Processing {num_samples} samples...")
    for _ in tqdm(range(num_samples), desc="Paraphrasing Progress"):
        index, result = results_queue.get()
        final_results[index] = result
        results_queue.task_done()

    for _ in range(num_workers):
        task_queue.put(None)

    for t in threads:
        t.join()

    print("\nAll paraphrase tasks completed!")
    return final_results
