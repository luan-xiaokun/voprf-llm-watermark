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

# OpenAI API configuration
OPENAI_API_KEY = ""

# API call retry configuration
MAX_RETRIES = 3
RETRY_DELAY = 5


# Added explicit instruction at the end to require only plain text output
USER_PROMPT = "As an expert copy-editor, please rewrite the following text in your own voice while ensuring that the final output contains the same information as the original text and has roughly the same length. Please paraphrase all sentences and do not omit any crucial details. Additionally, please take care to provide any relevant information about public figures, organizations, or other entities mentioned in the text to avoid any potential misunderstandings or biases."
# USER_PROMPT = "Perform the following two-step task on the text provided below:\nExtract: First, analyze the text and list all the key facts and logical relations in a bulleted list.\nReconstruct: Using only that bulleted list (ignore the original wording), write a new paragraph that conveys the same meaning and has the same length as the original.\nCrucial: Use a completely different vocabulary distribution. Use synonyms for nouns and verbs where contextually appropriate. Do not mimic the original sentence rhythm."


def set_deepseek_api_key(api_key: str | None = None) -> None:
    global DEEPSEEK_API_KEY
    if api_key is not None:
        DEEPSEEK_API_KEY = api_key
    elif "DEEPSEEK_API_KEY" in os.environ:
        DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]


def set_openai_api_key(api_key: str | None = None) -> None:
    global OPENAI_API_KEY
    if api_key is not None:
        OPENAI_API_KEY = api_key
    elif "OPENAI_API_KEY" in os.environ:
        OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


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


def paraphrase_text(client: OpenAI, text_to_paraphrase: str, model_name: str):
    """Call API to paraphrase a single text segment."""
    if not text_to_paraphrase or not text_to_paraphrase.strip():
        return ""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": USER_PROMPT + "\n" + text_to_paraphrase,
                    },
                ],
                temperature=0.7,
                # max_tokens=2048,
                max_completion_tokens=2048, # for gpt-5.x
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


def create_batch_file(samples: list[str], model_name: str, filename: str):
    """Create a JSONL file for OpenAI Batch API."""
    with open(filename, "w", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if not sample or not sample.strip():
                continue
            request = {
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "user",
                            "content": USER_PROMPT + "\n" + sample,
                        }
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2048,
                },
            }
            f.write(json.dumps(request) + "\n")


def submit_batch_job(client: OpenAI, input_file_path: str) -> str:
    """Upload file and submit a batch job."""
    batch_input_file = client.files.create(
        file=open(input_file_path, "rb"),
        purpose="batch",
    )

    batch_input_file_id = batch_input_file.id

    batch_job = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "paraphrase attack"},
    )
    return batch_job.id


def retrieve_batch_results(client: OpenAI, batch_id: str) -> dict[int, str]:
    """Check batch status and retrieve results if completed."""
    batch_job = client.batches.retrieve(batch_id)

    if batch_job.status != "completed":
        print(f"Batch job {batch_id} status: {batch_job.status}")
        if batch_job.errors:
            print(f"Errors: {batch_job.errors}")
        return None

    output_file_id = batch_job.output_file_id
    if not output_file_id:
        print("Batch job completed but no output file ID found.")
        return None

    file_response = client.files.content(output_file_id)
    content = file_response.text

    results = {}
    for line in content.split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            custom_id = data["custom_id"]
            # custom_id format is "req-{i}"
            index = int(custom_id.split("-")[1])

            response_body = data["response"]["body"]
            # OpenAI batch response structure
            if "choices" in response_body:
                paraphrased_text = response_body["choices"][0]["message"][
                    "content"
                ].strip()
                results[index] = paraphrased_text
            else:
                print(f"Warning: No choices in response for {custom_id}")

        except Exception as e:
            print(f"Error parsing line: {e}")

    return results


def worker(
    task_queue: Queue,
    results_queue: Queue,
    client: OpenAI,
    model_name: str,
):
    """Worker thread function: get tasks from task queue, process them, and put results into result queue."""
    while True:
        task = task_queue.get()
        if task is None:  # Sentinel value, indicating no more tasks
            break

        index, data = task
        paraphrased_generation = paraphrase_text(client, data, model_name)

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
            args=(task_queue, results_queue, client, MODEL_NAME),
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


def paraphrase_texts_by_openai(
    samples: list[str],
    model_name: str = "gpt-3.5-turbo",
    api_key: str | None = None,
    num_workers: int = 20,
    use_batch: bool = False,
    batch_id: str | None = None,
) -> list[str | None]:
    set_openai_api_key(api_key)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY variable not set.")

    # No base_url for OpenAI, using default
    client = OpenAI(api_key=OPENAI_API_KEY)

    if batch_id:
        # Retrieve mode
        print(f"Retrieving results for batch {batch_id}...")
        results_map = retrieve_batch_results(client, batch_id)
        if results_map is None:
            print(f"Batch {batch_id} is not ready or failed. Please try again later.")
            return None

        # Reconstruct list
        # Note: If samples length is not known (e.g. if we just pass empty list), we can't reconstruct fully but
        # usually calling code provides original samples.
        num_samples = len(samples)
        final_results: list[str | None] = [None] * num_samples
        for idx in range(num_samples):
            if idx in results_map:
                final_results[idx] = results_map[idx]
            else:
                # If original was empty, we skipped it in create_batch_file, so result should be empty
                if not samples[idx] or not samples[idx].strip():
                    final_results[idx] = ""

        print("Batch results retrieved successfully.")
        return final_results

    if use_batch:
        # Submit mode
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        input_file_name = f"batch_input_{timestamp}.jsonl"

        print(f"Generating batch input file {input_file_name}...")
        create_batch_file(samples, model_name, input_file_name)

        print("Submitting batch job...")
        try:
            batch_id = submit_batch_job(client, input_file_name)
            print(f"Batch job submitted! Batch ID: {batch_id}")
            print(
                f"Please save this ID. Re-run with --batch_id {batch_id} to retrieve results later (up to 24h)."
            )
        except Exception as e:
            print(f"Failed to submit batch job: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(input_file_name):
                os.remove(input_file_name)

        return None

    num_samples = len(samples)
    print(f"Number of tasks to process: {num_samples}")

    task_queue = Queue()
    results_queue = Queue()

    threads = []
    for _ in range(num_workers):
        thread = threading.Thread(
            target=worker,
            args=(task_queue, results_queue, client, model_name),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for i, sample in enumerate(samples):
        task_queue.put((i, sample))

    final_results: list[str | None] = [None] * num_samples

    print(f"Processing {num_samples} samples with {model_name}...")
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
