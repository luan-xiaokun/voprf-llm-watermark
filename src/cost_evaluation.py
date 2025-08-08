"""
This script evaluates the cost of detecting watermarks using the VOPRF protocol.

We first construct a set of long texts, and then use the substrings of them to
evaluate the cost of detection at different text lengths.

Long texts are obtained by concatenating samples from the C4 dataset.
Each long text has 409,000 characters. These long texts are obtained using the
following script.

```
LONG_TEXT_LENGTH = 409600
LONG_TEXT_NUM = 200

dataset = load_dataset("allenai/c4", "realnewslike", split="train", streaming=True)
long_texts = []
candidate = ""
for sample in dataset:
    text = sample["text"]
    if len(candidate) < LONG_TEXT_LENGTH:
        candidate += " " + text
    else:
        candidate = candidate[:LONG_TEXT_LENGTH]
        long_texts.append({"text": candidate})
        candidate = text
        if len(long_texts) >= LONG_TEXT_NUM:
            break
print("Long texts count:", len(long_texts))

utils.write_jsonlines("data/long_texts.jsonl", long_texts)
```

We report the average and standard deviation of the following metrics:
- raw_text_bytes: the size of the raw text in bytes
- blinded_elements_bytes: the size of the blinded elements in bytes
- evaluation_elements_bytes: the size of the evaluation elements in bytes
- proof_bytes: the size of the proof in bytes
- total_communication_bytes: the total size of communication in bytes
- preparation_time: the time taken to prepare the detection
- evaluation_time: the time taken to evaluate the detection
- finalization_time: the time taken to finalize the detection
- total_time: the total time taken for the detection
- expansion_factor: the ratio of total_communication_bytes to raw_text_bytes + 9
  (the +9 accounts for the boolean detection result and the 64-bit float p-value)
"""

import random
import secrets

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from voprf_py import BlindedElement, EvaluationElement, Proof, VoprfServer

import utils
from detection import DetectionCost, client_detect_batch_text

MODEL_PATH = "Qwen/Qwen2.5-3B"
DATA_FILE_PATH = "data/long_texts.jsonl"
OUTPUT_RECORD_PATH = "output/cost/detection_costs_w7.jsonl"
LONG_TEXT_LENGTH = 409600
LENGTH_BINS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
LONG_TEXT_NUM = 200


def get_server_interface(voprf_server: VoprfServer):
    def server_interface(
        blinded_elements: list[BlindedElement],
    ) -> tuple[list[EvaluationElement], Proof]:
        return voprf_server.batch_blind_evaluate(blinded_elements)

    return server_interface


def get_batch_detection_costs_stats(
    detection_costs: list[DetectionCost], field: str
) -> tuple[float, float]:
    if not detection_costs:
        return 0.0, 0.0
    array = np.array([getattr(cost, field) for cost in detection_costs])
    return float(np.mean(array)), float(np.std(array))


def main():
    window_size = 7
    gamma = 0.25
    seed = secrets.token_bytes(32)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    dataset = load_dataset(
        "json", data_files=DATA_FILE_PATH, split="train", streaming=True
    )
    voprf_server = VoprfServer(seed)

    server_public_key = voprf_server.get_public_key()
    server_interface = get_server_interface(voprf_server)

    final_detection_costs = {}

    for size in LENGTH_BINS:
        print(f"Evaluating size: {size}")
        start_index = random.randint(0, LONG_TEXT_LENGTH - size)
        texts = [sample["text"][start_index : start_index + size] for sample in dataset]

        _, detection_costs = client_detect_batch_text(
            texts,
            tokenizer,
            window_size,
            gamma,
            server_public_key,
            server_interface,
        )

        raw_text_bytes_stats = get_batch_detection_costs_stats(
            detection_costs, "raw_string_bytes"
        )
        blinded_elements_bytes_stats = get_batch_detection_costs_stats(
            detection_costs, "blinded_elements_bytes"
        )
        evaluation_elements_bytes_stats = get_batch_detection_costs_stats(
            detection_costs, "evaluation_elements_bytes"
        )
        proof_bytes_stats = get_batch_detection_costs_stats(
            detection_costs, "proof_bytes"
        )
        total_communication_bytes_stats = get_batch_detection_costs_stats(
            detection_costs, "total_communication_bytes"
        )
        preparation_time_stats = get_batch_detection_costs_stats(
            detection_costs, "preparation_time"
        )
        evaluation_time_stats = get_batch_detection_costs_stats(
            detection_costs, "evaluation_time"
        )
        finalization_time_stats = get_batch_detection_costs_stats(
            detection_costs, "finalization_time"
        )
        total_time_stats = get_batch_detection_costs_stats(
            detection_costs, "total_time"
        )
        expansion_factor_array = []
        for cost in detection_costs:
            # here, the `+ 9` represents the boolean detection result
            # and the 64-bit float p-value
            expansion_factor = cost.total_communication_bytes / (
                cost.raw_string_bytes + 9
            )
            expansion_factor_array.append(expansion_factor)
        expansion_factor_array = np.array(expansion_factor_array)
        expansion_factor_stats = (
            float(np.mean(expansion_factor_array)),
            float(np.std(expansion_factor_array)),
        )

        final_detection_costs[size] = {
            "raw_text_bytes": raw_text_bytes_stats,
            "blinded_elements_bytes": blinded_elements_bytes_stats,
            "evaluation_elements_bytes": evaluation_elements_bytes_stats,
            "proof_bytes": proof_bytes_stats,
            "total_communication_bytes": total_communication_bytes_stats,
            "preparation_time": preparation_time_stats,
            "evaluation_time": evaluation_time_stats,
            "finalization_time": finalization_time_stats,
            "total_time": total_time_stats,
            "expansion_factor": expansion_factor_stats,
        }

        utils.write_jsonlines(OUTPUT_RECORD_PATH, final_detection_costs[size], mode="a")


if __name__ == "__main__":
    main()
