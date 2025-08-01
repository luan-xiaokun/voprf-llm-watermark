import re
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm

import utils
from detection import local_detect_batch_text
# from detect_watermarked_text import calculate_conditional_perplexity

SAMPLING_METHOD_PATTERN = r"greedy|beam\d+|multinomial(?:-top(\d+))?"
WATERMARK_PARAM_PATTERN = r"w(\d+)_d(\d+\.?\d*)_g(0\.\d+)"

TOKENIZER_PATH = "Qwen/Qwen2.5-7B"
COMPUTE_PPL = False
OUTPUT_DIR = "output"
SIGNIFICANCE_LEVEL = 0.01
FINAL_RECORDS_FILE = "logs/results/final_results.jsonl"


def parse_output_file_name(file_name: str):
    file_name = file_name.rstrip(".jsonl")
    file_name_pattern = rf"^(\S+?)_({SAMPLING_METHOD_PATTERN})_(no_watermark|{WATERMARK_PARAM_PATTERN})$"
    match = re.match(file_name_pattern, file_name)
    if not match:
        raise ValueError(f"Invalid file name format: {file_name}.")
    groups = match.groups()
    model_name, sampling_method = groups[:2]
    top_k, param = groups[2:4]
    window_size, delta, gamma = groups[4:]
    return {
        "model_name": model_name,
        "sampling_method": sampling_method,
        "top_k": int(top_k) if top_k else None,
        "window_size": int(window_size) if window_size else None,
        "delta": float(delta) if delta else None,
        "gamma": float(gamma) if gamma else None,
        "no_watermark": param == "no_watermark",
    }


def main():
    output_dir = Path(OUTPUT_DIR)
    output_files = list(output_dir.glob("*.jsonl"))

    final_records = []
    for file in tqdm(output_files):
        parse_res = parse_output_file_name(file.name)

        if parse_res["no_watermark"]:
            print(f"Skipping file {file.name} as it does not contain watermarked text.")
            continue

        generation_records = utils.read_jsonlines(file)
        gen_args, server_seed_dict, *generation_records = generation_records
        server_seed = bytes.fromhex(server_seed_dict["server_seed"])

        window_size = parse_res["window_size"]
        gamma = parse_res["gamma"]

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

        detection_results = local_detect_batch_text(
            texts=[record["generated_text"] for record in generation_records],
            tokenizer=tokenizer,
            window_size=window_size,
            gamma=gamma,
            seed=server_seed,
        )

        example_num = len(detection_results)
        detected_num = sum(r.p_value < SIGNIFICANCE_LEVEL for r in detection_results)
        avg_green_token_num = (
            sum(r.green_token_num for r in detection_results) / example_num
        )
        avg_total_token_num = (
            sum(r.total_token_num for r in detection_results) / example_num
        )
        z_score_array = np.array([r.z_score for r in detection_results])
        avg_z_score = np.mean(z_score_array)
        std_z_score = np.std(z_score_array)

        parse_res.update(
            {
                "example_num": example_num,
                "detected_num": detected_num,
                "avg_green_token_num": avg_green_token_num,
                "avg_total_token_num": avg_total_token_num,
                "avg_z_score": avg_z_score,
                "std_z_score": std_z_score,
            }
        )
        final_records.append(parse_res)

    utils.write_jsonlines(FINAL_RECORDS_FILE, final_records)


if __name__ == "__main__":
    main()
