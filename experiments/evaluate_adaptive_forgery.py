import argparse
import datetime
import math
import statistics
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_suite.core.metrics import calculate_perplexities
from watermark_suite.utils import io_utils

DEFAULT_EVAL_MODEL = "Qwen/Qwen2.5-7B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate conditional perplexity and lexical quality metrics for "
            "an adaptive-forgery JSONL file."
        )
    )
    parser.add_argument("input_file")
    parser.add_argument("--eval_model", default=DEFAULT_EVAL_MODEL)
    parser.add_argument(
        "--baseline_file",
        default=None,
        help=(
            "Optional ordinary-generation JSONL with prompt_text and "
            "generated_text columns for a direct perplexity comparison."
        ),
    )
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--summary_file", default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "float16", "bfloat16"],
        default="auto",
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow evaluator downloads instead of requiring local files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing evaluated output, never the raw input file.",
    )
    return parser.parse_args()


def resolve_dtype(name: str, device: str) -> torch.dtype | str:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if device == "cuda":
        return torch.bfloat16
    return "auto"


def default_output_paths(
    input_path: Path,
    output_file: str | None,
    summary_file: str | None,
) -> tuple[Path, Path]:
    output_path = (
        Path(output_file).expanduser().resolve()
        if output_file
        else input_path.with_name(f"{input_path.stem}.evaluated.jsonl")
    )
    summary_path = (
        Path(summary_file).expanduser().resolve()
        if summary_file
        else input_path.with_name(f"{input_path.stem}.evaluation.json")
    )
    return output_path, summary_path


def describe(values: list[float | int]) -> dict:
    finite_values = [
        value
        for value in values
        if not isinstance(value, float) or math.isfinite(value)
    ]
    if not finite_values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    ordered = sorted(finite_values)
    p95_index = max(
        0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    )
    return {
        "mean": statistics.mean(finite_values),
        "median": statistics.median(finite_values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_index],
    }


def distinct_n(token_ids: list[int], n: int) -> float:
    gram_num = len(token_ids) - n + 1
    if gram_num <= 0:
        return 0.0
    grams = {
        tuple(token_ids[index : index + n])
        for index in range(gram_num)
    }
    return len(grams) / gram_num


def correlation(left: list[float], right: list[float]) -> float | None:
    pairs = [
        (x, y)
        for x, y in zip(left, right)
        if math.isfinite(x) and math.isfinite(y)
    ]
    if len(pairs) < 2:
        return None
    x_values, y_values = zip(*pairs)
    if len(set(x_values)) < 2 or len(set(y_values)) < 2:
        return None
    return statistics.correlation(x_values, y_values)


def validate_records(records: list[dict], path: Path) -> None:
    if not records:
        raise ValueError(f"{path} contains no records")
    missing = [
        index
        for index, record in enumerate(records)
        if "prompt_text" not in record or "generated_text" not in record
    ]
    if missing:
        raise ValueError(
            f"{path} has records without prompt_text/generated_text at "
            f"indices {missing[:10]}"
        )


def evaluate_records(model, tokenizer, records: list[dict], args):
    dataset = Dataset.from_list(
        [
            {
                "prompt_text": record["prompt_text"],
                "generated_text": record["generated_text"],
            }
            for record in records
        ]
    )
    perplexity = calculate_perplexities(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        prompt_column="prompt_text",
        target_column="generated_text",
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
    )

    quality_records = []
    for index, record in enumerate(records):
        text = record["generated_text"]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_num = len(token_ids)
        quality_records.append(
            {
                "conditional_perplexity": (
                    perplexity.sample_perplexities[index]
                ),
                "mean_negative_log_likelihood": (
                    perplexity.sample_negative_log_likelihoods[index]
                ),
                "perplexity_target_token_num": (
                    perplexity.sample_target_token_nums[index]
                ),
                "evaluation_token_num": token_num,
                "character_num": len(text),
                "characters_per_token": (
                    len(text) / token_num if token_num else 0.0
                ),
                "distinct_1": distinct_n(token_ids, 1),
                "distinct_2": distinct_n(token_ids, 2),
                "distinct_4": distinct_n(token_ids, 4),
            }
        )
    return perplexity, quality_records


def build_summary(
    args: argparse.Namespace,
    records: list[dict],
    quality_records: list[dict],
    perplexity,
    baseline_summary: dict | None,
    elapsed_seconds: float,
) -> dict:
    sample_perplexities = [
        record["conditional_perplexity"] for record in quality_records
    ]
    summary = {
        "sample_num": len(records),
        "evaluation_model": args.eval_model,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "conditional_perplexity": perplexity.perplexity,
        "mean_negative_log_likelihood": (
            perplexity.mean_negative_log_likelihood
        ),
        "perplexity_target_token_num": perplexity.total_target_token_num,
        "sample_conditional_perplexity": describe(sample_perplexities),
        "evaluation_token_length": describe(
            [record["evaluation_token_num"] for record in quality_records]
        ),
        "characters_per_token": describe(
            [record["characters_per_token"] for record in quality_records]
        ),
        "distinct_1": describe(
            [record["distinct_1"] for record in quality_records]
        ),
        "distinct_2": describe(
            [record["distinct_2"] for record in quality_records]
        ),
        "distinct_4": describe(
            [record["distinct_4"] for record in quality_records]
        ),
        "elapsed_seconds": elapsed_seconds,
    }

    if all("sample_metrics" in record for record in records):
        query_rates = [
            record["sample_metrics"]["queries_per_scored_token"]
            for record in records
        ]
        green_ratios = [
            record["sample_metrics"]["selected_green_ratio"]
            for record in records
        ]
        fallback_ratios = [
            record["sample_metrics"]["fallback_ratio"]
            for record in records
        ]
        p_values = [
            float(record["detection"]["p_value"]) for record in records
        ]
        local_model_perplexities = [
            (
                float(record["adaptive_forgery"]["local_model_perplexity"])
                if record["adaptive_forgery"].get(
                    "local_model_perplexity"
                )
                is not None
                else float("nan")
            )
            for record in records
        ]
        log_probability_gaps = [
            (
                float(record["adaptive_forgery"]["mean_log_probability_gap"])
                if record["adaptive_forgery"].get(
                    "mean_log_probability_gap"
                )
                is not None
                else float("nan")
            )
            for record in records
        ]
        summary["quality_cost_correlations"] = {
            "perplexity_vs_queries_per_scored_token": correlation(
                sample_perplexities, query_rates
            ),
            "perplexity_vs_selected_green_ratio": correlation(
                sample_perplexities, green_ratios
            ),
            "perplexity_vs_fallback_ratio": correlation(
                sample_perplexities, fallback_ratios
            ),
            "perplexity_vs_log10_p_value": correlation(
                sample_perplexities,
                [math.log10(max(value, 1e-300)) for value in p_values],
            ),
            "evaluator_perplexity_vs_local_model_perplexity": correlation(
                sample_perplexities, local_model_perplexities
            ),
            "perplexity_vs_mean_log_probability_gap": correlation(
                sample_perplexities, log_probability_gaps
            ),
        }

    if baseline_summary is not None:
        baseline_perplexity = baseline_summary["conditional_perplexity"]
        summary["baseline"] = baseline_summary
        summary["perplexity_ratio_to_baseline"] = (
            perplexity.perplexity / baseline_perplexity
            if baseline_perplexity
            else None
        )
        summary["perplexity_increase_over_baseline"] = (
            perplexity.perplexity - baseline_perplexity
        )

    return summary


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_file).expanduser().resolve()
    output_path, summary_path = default_output_paths(
        input_path, args.output_file, args.summary_file
    )
    if output_path == input_path:
        raise ValueError("output_file must not overwrite the raw input JSONL")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    records = list(io_utils.read_jsonlines(input_path))
    validate_records(records, input_path)
    baseline_records = None
    if args.baseline_file:
        baseline_path = Path(args.baseline_file).expanduser().resolve()
        baseline_records = list(io_utils.read_jsonlines(baseline_path))
        validate_records(baseline_records, baseline_path)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.eval_model,
            padding_side="right",
            local_files_only=not args.allow_download,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.eval_model,
            torch_dtype=resolve_dtype(args.dtype, device),
            local_files_only=not args.allow_download,
        )
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            f"Unable to load evaluator {args.eval_model!r}. Provide a complete "
            "local checkpoint or pass --allow_download."
        ) from error

    start = time.perf_counter()
    perplexity, quality_records = evaluate_records(
        model, tokenizer, records, args
    )
    evaluated_records = []
    for record, quality in zip(records, quality_records):
        evaluated = dict(record)
        evaluated["quality"] = quality
        evaluated_records.append(evaluated)

    baseline_summary = None
    if baseline_records is not None:
        baseline_perplexity, baseline_quality = evaluate_records(
            model, tokenizer, baseline_records, args
        )
        baseline_summary = {
            "file": str(Path(args.baseline_file).expanduser().resolve()),
            "sample_num": len(baseline_records),
            "conditional_perplexity": baseline_perplexity.perplexity,
            "mean_negative_log_likelihood": (
                baseline_perplexity.mean_negative_log_likelihood
            ),
            "perplexity_target_token_num": (
                baseline_perplexity.total_target_token_num
            ),
            "sample_conditional_perplexity": describe(
                [
                    record["conditional_perplexity"]
                    for record in baseline_quality
                ]
            ),
        }

    elapsed_seconds = time.perf_counter() - start
    summary = build_summary(
        args=args,
        records=records,
        quality_records=quality_records,
        perplexity=perplexity,
        baseline_summary=baseline_summary,
        elapsed_seconds=elapsed_seconds,
    )
    summary["input_file"] = str(input_path)
    summary["output_file"] = str(output_path)
    summary["timestamp"] = datetime.datetime.now().isoformat()

    io_utils.write_jsonlines(output_path, evaluated_records, mode="w")
    io_utils.write_json(summary_path, summary, indent=2)
    print(f"Evaluated records: {output_path}")
    print(f"Evaluation summary: {summary_path}")


if __name__ == "__main__":
    main()
