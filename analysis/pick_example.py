from datasets import Dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

from watermark_suite.core.metrics import calculate_perplexity
from watermark_suite.utils import io_utils
from watermark_suite.schemes import VOWDetector, KGWDetector, RDFDetector, PDWDetector
from watermark_suite.schemes.upv.detection import UPVDetector


output_dir = "output/generation"
vow_file_path = "vow_Qwen2.5-3B_c4_multinomial_w4_d2.5_g0.5.jsonl"
lefthash_file_path = "lefthash_Qwen2.5-3B_c4_multinomial_d2.0_g0.25.jsonl"
selfhash_file_path = "selfhash_Qwen2.5-3B_c4_multinomial_d2.0_g0.25.jsonl"
rdf_file_path = "rdf_Qwen2.5-3B_c4_default_l256_s42_cpu.jsonl"
pdw_file_path = "pdw_Qwen2.5-3B_c4_multinomial.jsonl"
upv_file_path = "upv_Qwen2.5-3B_c4_multinomial_18bits_5layers.jsonl"
no_watermark_file_path = "no-watermark_Qwen2.5-3B_c4_multinomial.jsonl"


def load_datasets():
    record_dir = Path(output_dir)
    vow_records = list(io_utils.read_jsonlines(record_dir / vow_file_path))
    lefthash_records = list(io_utils.read_jsonlines(record_dir / lefthash_file_path))
    selfhash_records = list(io_utils.read_jsonlines(record_dir / selfhash_file_path))
    rdf_records = list(io_utils.read_jsonlines(record_dir / rdf_file_path))
    pdw_records = list(io_utils.read_jsonlines(record_dir / pdw_file_path))
    upv_records = list(io_utils.read_jsonlines(record_dir / upv_file_path))
    no_watermark_records = list(
        io_utils.read_jsonlines(record_dir / no_watermark_file_path)
    )

    num = min(
        len(vow_records),
        len(lefthash_records),
        len(selfhash_records),
        len(rdf_records),
        len(pdw_records),
        len(upv_records),
    )
    vow_dataset = Dataset.from_list(vow_records).select(range(num))
    lefthash_dataset = Dataset.from_list(lefthash_records).select(range(num))
    selfhash_dataset = Dataset.from_list(selfhash_records).select(range(num))
    rdf_dataset = Dataset.from_list(rdf_records).select(range(num))
    pdw_dataset = Dataset.from_list(pdw_records).select(range(num))
    upv_dataset = Dataset.from_list(upv_records).select(range(num))
    no_watermark_dataset = Dataset.from_list(no_watermark_records).select(range(num))
    datasets = {
        # "VOW": vow_dataset,
        # "LeftHash": lefthash_dataset,
        # "SelfHash": selfhash_dataset,
        # "PDW": pdw_dataset,
        # "RDF": rdf_dataset,
        "UPV": upv_dataset,
        "No-watermark": no_watermark_dataset,
    }
    return datasets


def escape_latex_chars(text):
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
        "\n": "\\newline ",
    }
    for char, escaped_char in replacements.items():
        text = text.replace(char, escaped_char)
    return text


def generate_latex_colored_text(tokenizer, text: str, green_token_mask: list[bool]):
    green_color = "green!20"
    red_color = "red!20"

    latex_string = ""
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    tokens_str = tokenizer.convert_ids_to_tokens(token_ids)

    output_chunks = []
    current_chunk_tokens = []
    current_color = "red"

    for i, token_str in enumerate(tokens_str):
        token_color = "green" if green_token_mask[i] else "red"

        if token_color == current_color:
            current_chunk_tokens.append(token_str)
        else:
            decoded_chunk = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(current_chunk_tokens)
            )
            escaped_chunk = escape_latex_chars(decoded_chunk)
            latex_command = f"\\hl{current_color}{{{escaped_chunk}}}"
            output_chunks.append(latex_command)

            current_chunk_tokens = [token_str]
            current_color = token_color

    if current_chunk_tokens:
        decoded_chunk = tokenizer.decode(
            tokenizer.convert_tokens_to_ids(current_chunk_tokens)
        )
        escaped_chunk = escape_latex_chars(decoded_chunk)
        latex_command = f"\\hl{current_color}{{{escaped_chunk}}}"
        output_chunks.append(latex_command)

    return "".join(output_chunks)


def show_examples(datasets, indices: list[int]):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B", torch_dtype=torch.bfloat16, local_files_only=True
    )

    with open("data/server_seed") as f:
        seed = bytes.fromhex(f.read().strip())
    detectors = {
        "VOW": VOWDetector(tokenizer, seed, gamma=0.5, window_size=4),
        "LeftHash": KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,
            seeding_scheme="lefthash",
            device="cuda" if torch.cuda.is_available() else "cpu",
            tokenizer=tokenizer,
            normalizers=[],
            ignore_repeated_ngrams=True,
        ),
        "SelfHash": KGWDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=0.25,
            seeding_scheme="selfhash",
            device="cuda" if torch.cuda.is_available() else "cpu",
            tokenizer=tokenizer,
            normalizers=[],
            ignore_repeated_ngrams=True,
        ),
        "PDW": PDWDetector(),
        "RDF": RDFDetector(tokenizer),
        "UPV": UPVDetector(
            tokenizer,
            "experiments/upv_baseline/model",
            window_size=4,
            bits_num=18,
            gamma=0.5,
        ),
    }
    for idx in indices:
        for method, dataset in datasets.items():
            sample = dataset[idx]
            subset = dataset.select(range(idx, idx + 1))
            ppl = calculate_perplexity(
                model, tokenizer, subset, "prompt_text", "generated_text"
            )
            prompt = sample["prompt_text"]
            generated_text = sample["generated_text"]

            token_ids = tokenizer.encode(generated_text, add_special_tokens=False)
            token_ids = token_ids[:75]
            generated_text = tokenizer.decode(token_ids)

            print(f"{method} sample {idx + 1} (PPL: {ppl:.2f}):")
            print(f"Prompt: {prompt}")
            print("=" * 120)
            if method == "VOW":
                result = detectors[method].local_detect(
                    generated_text, return_green_token_mask=True
                )
                print(generated_text)
                print("-" * 120)
                text = generate_latex_colored_text(
                    tokenizer, generated_text, result.green_token_mask
                )
            elif method in ["SelfHash", "LeftHash"]:
                window_size = 1 if method == "LeftHash" else 4
                green_token_mask = [False] * window_size + result.green_token_mask
                result = detectors[method].detect(
                    generated_text, return_green_token_mask=True
                )
                print(generated_text)
                print("-" * 120)
                text = generate_latex_colored_text(
                    tokenizer, generated_text, green_token_mask
                )
            elif method == "UPV":
                result = detectors[method].detect(
                    generated_text, return_green_token_mask=True
                )
                print(generated_text)
                print("-" * 120)
                text = generate_latex_colored_text(
                    tokenizer,
                    generated_text,
                    result.green_token_mask,
                )

            elif method == "No-watermark":
                text = generated_text
            else:
                result = detectors[method].detect(generated_text)
                text = generated_text
            print(text)
            print("-" * 120)
            print(result)
            print("-" * 120)


def main():
    datasets = load_datasets()

    show_examples(datasets, [0, 45, 60, 85])


if __name__ == "__main__":
    main()
