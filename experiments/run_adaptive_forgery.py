import argparse
import datetime
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from voprf_py import VoprfServer

from watermark_suite.attacks import (
    AdaptiveWatermarkForger,
    VOPRFColorOracle,
    theoretical_green_probability,
    theoretical_queries_per_scored_token,
)
from watermark_suite.schemes import VOWDetector
from watermark_suite.schemes.vow.key import get_server_seed
from watermark_suite.utils import io_utils

DEFAULT_MODEL = "Qwen/Qwen2.5-3B"
DEFAULT_DATASET = "c4"
DEFAULT_OUTPUT_DIR = "output/forgery"
DEFAULT_SERVER_SEED_PATH = "data/server_seed"
DATASET_PATHS = {
    "c4": "data/c4_realnewslike_subset_1000.jsonl",
    "eli5": "data/eli5",
}
ELI5_SYSTEM_MESSAGE = (
    "Explain the following question in about 500 words like I'm 5 years old. "
    "Use very simple language, short sentences, and analogies a child can understand."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Forge VOW-marked text using a local language model and only the "
            "public color-query interface."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_PATHS), default=DEFAULT_DATASET
    )
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument(
        "--max_candidates",
        "-k",
        type=int,
        default=8,
        help="Maximum number of candidates queried at each scored position.",
    )
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument(
        "--significance_levels",
        nargs="+",
        type=float,
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument("--server_seed_path", default=DEFAULT_SERVER_SEED_PATH)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random_seed", type=int, default=42)
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
        help="Allow model/tokenizer downloads instead of requiring local files.",
    )
    parser.add_argument(
        "--allow_special_tokens",
        action="store_true",
        help="Allow special-token candidates; by default they are suppressed.",
    )
    parser.add_argument(
        "--trace_level",
        choices=["none", "compact", "full"],
        default="compact",
        help=(
            "Process detail saved per sample: none, compact per-position "
            "decisions, or full contexts and candidate lists."
        ),
    )
    parser.add_argument(
        "--include_steps",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output file with the same configuration.",
    )
    args = parser.parse_args()
    if args.include_steps:
        args.trace_level = "full"
    return args


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


def load_samples(dataset_name: str, num: int):
    if num <= 0:
        raise ValueError("num must be positive")

    dataset_path = DATASET_PATHS[dataset_name]
    if dataset_name == "c4":
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = load_dataset(dataset_path, split="train")
    return dataset.select(range(min(num, len(dataset))))


def format_prompt(sample: dict, dataset_name: str, tokenizer) -> tuple[str, str]:
    if dataset_name == "c4":
        source_prompt = sample["prompt_text"]
        return source_prompt, source_prompt

    source_prompt = sample["question"]
    if getattr(tokenizer, "chat_template", None) is not None:
        model_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": ELI5_SYSTEM_MESSAGE},
                {"role": "user", "content": source_prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        model_prompt = (
            f"{ELI5_SYSTEM_MESSAGE}\n\nQuestion: {source_prompt}\nAnswer:"
        )
    return source_prompt, model_prompt


def make_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    model_name = args.model.rstrip("/").split("/")[-1].replace("_", "-")
    gamma = str(args.gamma).replace(".", "p")
    stem = (
        f"adaptive-forgery_{model_name}_{args.dataset}"
        f"_w{args.window_size}_g{gamma}_k{args.max_candidates}"
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return (
        output_dir / f"{stem}.jsonl",
        output_dir / f"{stem}.meta.json",
        output_dir / f"{stem}.summary.json",
    )


def describe(values: list[float | int]) -> dict:
    if not values:
        return {
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
        }
    ordered = sorted(values)
    p95_index = max(
        0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    )
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[p95_index],
    }


def build_sample_metrics(result, detection) -> dict:
    effective_token_num = detection.effective_token_num
    green_ranks = [
        step.selected_rank
        for step in result.steps
        if step.selected_green is True
    ]
    return {
        "generated_token_num": result.generated_token_num,
        "detector_token_num": detection.total_token_num,
        "scored_position_num": result.scored_token_num,
        "detector_effective_token_num": effective_token_num,
        "oracle_query_count": result.oracle_query_count,
        "color_check_count": result.color_check_count,
        "voprf_round_count": (
            result.oracle_protocol_stats.server_round_count
            if result.oracle_protocol_stats is not None
            else result.oracle_query_count
        ),
        "queries_per_generated_token": result.queries_per_generated_token,
        "queries_per_scored_token": result.queries_per_scored_token,
        "queries_per_honest_audit_query": (
            result.oracle_query_count / effective_token_num
            if effective_token_num
            else 0.0
        ),
        "color_checks_per_scored_token": (
            result.color_checks_per_scored_token
        ),
        "cache_hit_ratio": (
            result.cache_hit_count / result.color_check_count
            if result.color_check_count
            else 0.0
        ),
        "fallback_ratio": result.fallback_ratio,
        "selected_green_ratio": result.selected_green_ratio,
        "detector_green_ratio": detection.green_ratio,
        "effective_token_ratio": (
            effective_token_num / detection.total_token_num
            if detection.total_token_num
            else 0.0
        ),
        "duplicate_selected_pair_ratio": (
            result.duplicate_selected_pair_num / result.scored_token_num
            if result.scored_token_num
            else 0.0
        ),
        "green_candidate_rank_histogram": {
            str(rank): count for rank, count in sorted(Counter(green_ranks).items())
        },
        "mean_green_candidate_rank": result.mean_green_candidate_rank,
        "local_model_perplexity": result.local_model_perplexity,
        "mean_selected_negative_log_likelihood": (
            result.mean_selected_negative_log_likelihood
        ),
        "mean_log_probability_gap": result.mean_log_probability_gap,
        "forgery_seconds": result.elapsed_seconds,
        "tokens_per_second": result.tokens_per_second,
        "oracle_time_fraction": (
            result.oracle_time_seconds / result.elapsed_seconds
            if result.elapsed_seconds
            else 0.0
        ),
        "tokenization_preserved": result.tokenization_preserved,
    }


def build_summary(
    args: argparse.Namespace,
    records: list[dict],
    elapsed_seconds: float,
) -> dict:
    forgery_records = [record["adaptive_forgery"] for record in records]
    detection_records = [record["detection"] for record in records]

    total_oracle_queries = sum(
        record["oracle_query_count"] for record in forgery_records
    )
    total_color_checks = sum(
        record["color_check_count"] for record in forgery_records
    )
    total_cache_hits = sum(
        record["cache_hit_count"] for record in forgery_records
    )
    total_scored_tokens = sum(
        record["scored_token_num"] for record in forgery_records
    )
    total_selected_green = sum(
        record["selected_green_token_num"] for record in forgery_records
    )
    total_fallbacks = sum(
        record["fallback_count"] for record in forgery_records
    )
    total_effective_tokens = sum(
        record["effective_token_num"] for record in detection_records
    )
    total_detected_green = sum(
        record["green_token_num"] for record in detection_records
    )
    p_values = [record["p_value"] for record in detection_records]
    sample_metrics = [record["sample_metrics"] for record in records]
    total_generated_tokens = sum(
        record["generated_token_num"] for record in forgery_records
    )
    total_detector_tokens = sum(
        record["total_token_num"] for record in detection_records
    )
    total_duplicate_pairs = sum(
        record["duplicate_selected_pair_num"] for record in forgery_records
    )
    total_oracle_time = sum(
        record["oracle_time_seconds"] for record in forgery_records
    )
    total_forgery_time = sum(
        record["elapsed_seconds"] for record in forgery_records
    )
    protocol_stats = [
        record["oracle_protocol_stats"]
        for record in forgery_records
        if record["oracle_protocol_stats"] is not None
    ]
    total_communication_bytes = sum(
        record["total_communication_bytes"] for record in protocol_stats
    )
    green_rank_histogram = Counter()
    for record in sample_metrics:
        green_rank_histogram.update(
            {
                int(rank): count
                for rank, count in record[
                    "green_candidate_rank_histogram"
                ].items()
            }
        )
    theoretical_green = theoretical_green_probability(
        args.gamma, args.max_candidates
    )
    theoretical_queries = theoretical_queries_per_scored_token(
        args.gamma, args.max_candidates
    )
    observed_green_ratio = (
        total_selected_green / total_scored_tokens
        if total_scored_tokens
        else 0.0
    )
    observed_queries_per_scored = (
        total_oracle_queries / total_scored_tokens
        if total_scored_tokens
        else 0.0
    )

    return {
        "sample_num": len(records),
        "gamma": args.gamma,
        "window_size": args.window_size,
        "max_candidates": args.max_candidates,
        "theoretical_green_probability": theoretical_green,
        "theoretical_queries_per_scored_token": theoretical_queries,
        "observed_minus_theoretical_green_probability": (
            observed_green_ratio - theoretical_green
        ),
        "observed_to_theoretical_query_ratio": (
            observed_queries_per_scored / theoretical_queries
            if theoretical_queries
            else None
        ),
        "oracle_query_count": total_oracle_queries,
        "color_check_count": total_color_checks,
        "cache_hit_count": total_cache_hits,
        # Candidate queries are deliberately sequential, so every uncached
        # logical query is also one public VOPRF request/response round.
        "voprf_round_count": total_oracle_queries,
        "generated_token_num": total_generated_tokens,
        "detector_token_num": total_detector_tokens,
        "scored_token_num": total_scored_tokens,
        "detector_effective_token_num": total_effective_tokens,
        "duplicate_selected_pair_num": total_duplicate_pairs,
        "observed_queries_per_generated_token": (
            total_oracle_queries / total_generated_tokens
            if total_generated_tokens
            else 0.0
        ),
        "observed_queries_per_scored_token": observed_queries_per_scored,
        "query_overhead_vs_honest_audit": (
            total_oracle_queries / total_effective_tokens
            if total_effective_tokens
            else 0.0
        ),
        "selected_green_ratio": observed_green_ratio,
        "detector_green_ratio": (
            total_detected_green / total_effective_tokens
            if total_effective_tokens
            else 0.0
        ),
        "fallback_ratio": (
            total_fallbacks / total_scored_tokens
            if total_scored_tokens
            else 0.0
        ),
        "cache_hit_ratio": (
            total_cache_hits / total_color_checks
            if total_color_checks
            else 0.0
        ),
        "duplicate_selected_pair_ratio": (
            total_duplicate_pairs / total_scored_tokens
            if total_scored_tokens
            else 0.0
        ),
        "green_candidate_rank_histogram": {
            str(rank): count
            for rank, count in sorted(green_rank_histogram.items())
        },
        "generated_token_length": describe(
            [record["generated_token_num"] for record in sample_metrics]
        ),
        "oracle_queries_per_sample": describe(
            [record["oracle_query_count"] for record in sample_metrics]
        ),
        "queries_per_scored_token_per_sample": describe(
            [record["queries_per_scored_token"] for record in sample_metrics]
        ),
        "queries_per_honest_audit_query_per_sample": describe(
            [
                record["queries_per_honest_audit_query"]
                for record in sample_metrics
            ]
        ),
        "p_value": describe(p_values),
        "local_model_perplexity": describe(
            [
                record["local_model_perplexity"]
                for record in sample_metrics
                if record["local_model_perplexity"] is not None
            ]
        ),
        "mean_log_probability_gap": describe(
            [
                record["mean_log_probability_gap"]
                for record in sample_metrics
                if record["mean_log_probability_gap"] is not None
            ]
        ),
        "total_oracle_time_seconds": total_oracle_time,
        "oracle_time_fraction": (
            total_oracle_time / total_forgery_time
            if total_forgery_time
            else 0.0
        ),
        "total_communication_bytes": total_communication_bytes,
        "communication_bytes_per_oracle_query": (
            total_communication_bytes / total_oracle_queries
            if total_oracle_queries
            else 0.0
        ),
        "tokenization_preserved_ratio": (
            sum(record["tokenization_preserved"] for record in forgery_records)
            / len(forgery_records)
            if forgery_records
            else 0.0
        ),
        "median_p_value": statistics.median(p_values) if p_values else None,
        "forgery_success_rate": {
            f"{level:.0e}": (
                sum(p_value < level for p_value in p_values) / len(p_values)
                if p_values
                else 0.0
            )
            for level in args.significance_levels
        },
        "total_elapsed_seconds": elapsed_seconds,
        "mean_forgery_seconds": (
            statistics.mean(
                record["elapsed_seconds"] for record in forgery_records
            )
            if forgery_records
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    if args.window_size <= 0:
        raise ValueError("window_size must be positive")
    if args.max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    if not 0 < args.gamma < 1:
        raise ValueError("gamma must be in the open interval (0, 1)")

    set_seed(args.random_seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            padding_side="left",
            local_files_only=not args.allow_download,
        )
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            f"Unable to load tokenizer {args.model!r}. Provide a complete local "
            "checkpoint or pass --allow_download."
        ) from error
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=resolve_dtype(args.dtype, device),
            local_files_only=not args.allow_download,
        )
    except (OSError, AttributeError) as error:
        raise RuntimeError(
            f"Unable to load model {args.model!r}. Provide a complete local "
            "checkpoint or pass --allow_download."
        ) from error
    model.to(device)
    model.eval()

    # The secret seed is held by the local experimental service and evaluator.
    # The forger below receives only the public key and blinded service callback.
    server_seed = get_server_seed(args.server_seed_path)
    voprf_server = VoprfServer(server_seed)

    def server_interface(blinded_elements):
        return voprf_server.batch_blind_evaluate(blinded_elements)

    color_oracle = VOPRFColorOracle(
        server_public_key=voprf_server.get_public_key(),
        server_interface=server_interface,
        gamma=args.gamma,
    )
    forger = AdaptiveWatermarkForger(
        model=model,
        tokenizer=tokenizer,
        oracle=color_oracle,
        window_size=args.window_size,
        max_candidates=args.max_candidates,
    )
    evaluator = VOWDetector(
        tokenizer=tokenizer,
        seed=server_seed,
        gamma=args.gamma,
        window_size=args.window_size,
    )

    samples = load_samples(args.dataset, args.num)
    output_path, metadata_path, summary_path = make_output_paths(args)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; pass --overwrite to replace it"
        )

    metadata = vars(args).copy()
    metadata.update(
        {
            "file_name": output_path.name,
            "timestamp": datetime.datetime.now().isoformat(),
            "forgery_threat_model": (
                "local language model plus public VOPRF color oracle; "
                "no watermark-key access"
            ),
        }
    )
    io_utils.write_json(metadata_path, metadata, indent=2)

    suppress_token_ids = (
        None if args.allow_special_tokens else tokenizer.all_special_ids
    )
    records = []
    start = datetime.datetime.now()
    write_mode = "w"

    for sample in tqdm(samples, desc="Forging"):
        source_prompt, model_prompt = format_prompt(
            sample, args.dataset, tokenizer
        )
        result = forger.forge(
            prompt=model_prompt,
            max_new_tokens=args.max_new_tokens,
            suppress_token_ids=suppress_token_ids,
            stop_on_eos=True,
        )
        detection = evaluator.local_detect(result.text)

        record = {
            "source_prompt": source_prompt,
            "prompt_text": model_prompt,
            "generated_text": result.text,
            "adaptive_forgery": result.to_dict(
                trace_level=args.trace_level
            ),
            "detection": {
                "p_value": detection.p_value,
                "green_token_num": detection.green_token_num,
                "effective_token_num": detection.effective_token_num,
                "green_ratio": detection.green_ratio,
                "total_token_num": detection.total_token_num,
            },
            "sample_metrics": build_sample_metrics(result, detection),
        }
        records.append(record)
        io_utils.write_jsonlines(output_path, record, mode=write_mode)
        write_mode = "a"

    elapsed_seconds = (datetime.datetime.now() - start).total_seconds()
    summary = build_summary(args, records, elapsed_seconds)
    io_utils.write_json(summary_path, summary, indent=2)

    print(f"Results: {output_path}")
    print(f"Summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
