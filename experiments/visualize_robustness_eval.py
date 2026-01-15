from pathlib import Path
from transformers import AutoTokenizer
from watermark_suite.core import detect_texts
from watermark_suite.schemes import VOWDetector, KGWDetector, RDFDetector
from watermark_suite.schemes.upv.detection import UPVDetector
from watermark_suite.schemes.vow.key import get_server_seed
from watermark_suite.utils import io_utils

DEFAULT_SERVER_SEED_PATH = "data/server_seed"


def load_samples():
    folder_path = Path("output/generation")
    file_names = [
        "upv_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_18bits_5layers.jsonl",
        "lefthash_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_d2.0_g0.25.jsonl",
        "selfhash_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_d2.0_g0.25.jsonl",
        "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w1_d2.5_g0.5.jsonl",
        "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w2_d2.5_g0.5.jsonl",
        "vow_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50_w4_d2.5_g0.5.jsonl",
        "rdf_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_default_l256_s42_cpu.jsonl",
    ]
    all_samples = {}
    for file_name in file_names:
        samples = list(io_utils.read_jsonlines(folder_path / file_name))
        name = file_name.split("_")[0]
        if "vow" in name and "w1" in file_name:
            name = "vow"
        elif "vow" in name and "w4" in file_name:
            name = "vow-w4"
        elif "vow" in name and "w2" in file_name:
            name = "vow-w2"
        all_samples[name] = samples

    return all_samples


def main():
    all_samples = load_samples()
    tokenizer = AutoTokenizer.from_pretrained(
        "unsloth/Llama-3.1-8B-Instruct-unsloth-bnb-4bit", local_files_only=True
    )
    negative_samples = list(
        io_utils.read_jsonlines(
            Path("output/generation")
            / "no-watermark_Llama-3.1-8B-Instruct-unsloth-bnb-4bit_eli5_top-50.jsonl"
        )
    )

    all_records = {}

    for scheme, samples in all_samples.items():
        print(f"Processing scheme: {scheme}", len(samples))
        if "vow" in scheme:
            window_size = 1
            if "w2" in scheme:
                window_size = 2
            elif "w4" in scheme:
                window_size = 4
            print(f"  Using window size: {window_size}")
            detector = VOWDetector(
                tokenizer,
                seed=get_server_seed(DEFAULT_SERVER_SEED_PATH),
                gamma=0.5,
                window_size=window_size,
            )
        elif scheme in ["lefthash", "selfhash"]:
            detector = KGWDetector(
                vocab=list(tokenizer.get_vocab().values()),
                gamma=0.25,
                seeding_scheme=scheme,
                device="cuda",
                tokenizer=tokenizer,
                normalizers=[],
                ignore_repeated_ngrams=True,
            )
        elif scheme == "rdf":
            detector = RDFDetector(tokenizer, length=256, seed=42, n_runs=100)
        elif scheme == "upv":
            detector = detector = UPVDetector(
                tokenizer,
                "experiments/upv_baseline/model",
                window_size=4,
                bits_num=18,
                gamma=0.5,
            )
        else:
            print(f"Unknown scheme: {scheme}, skipping...")
            continue

        all_records[scheme] = {}

        column_name = "paraphrased-gpt-5.1"
        column_name = "synonym_replaced"
        print(f"  Detecting column: {column_name}")
        detection_results = detect_texts(
            detector,
            [s[column_name] for s in samples if s[column_name]],
            token_num=500,
        )
        all_records[scheme][column_name] = detection_results["all_p_values"]
        # Negative samples
        print(f"  Detecting negative samples")
        negative_detection_results = detect_texts(
            detector,
            [s["generated_text"] for s in negative_samples if s["generated_text"]],
        )
        all_records[scheme]["negative"] = negative_detection_results["all_p_values"]

        output_path = Path("data/plot_data/robustness_eval_records_upv.json")
        io_utils.write_json(output_path, all_records)
        print(f"Detection results saved to {output_path}")


if __name__ == "__main__":
    main()
