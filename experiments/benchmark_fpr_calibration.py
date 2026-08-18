#!/usr/bin/env python3
"""Probe a host and benchmark the FPR-calibration detectors on C4 windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from watermark_suite.experiments.scheme_registry import WATERMARK_SCHEMES


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPOSITORY / "data/c4_realnewslike_subset_1000.jsonl"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-3B"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _cpu_affinity_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _physical_core_count() -> int | None:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    cores: set[tuple[str, str]] = set()
    physical_id: str | None = None
    core_id: str | None = None
    for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
        if not line.strip():
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            physical_id = None
            core_id = None
            continue
        key, separator, value = line.partition(":")
        if not separator:
            continue
        if key.strip() == "physical id":
            physical_id = value.strip()
        elif key.strip() == "core id":
            core_id = value.strip()
    return len(cores) or None


def _cgroup_cpu_limit() -> float | None:
    path = Path("/sys/fs/cgroup/cpu.max")
    if not path.is_file():
        return None
    pieces = path.read_text(encoding="utf-8").strip().split()
    if len(pieces) != 2 or pieces[0] == "max":
        return None
    quota, period = (int(piece) for piece in pieces)
    return quota / period if period > 0 else None


def _memory_total_gib() -> float | None:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            kibibytes = int(line.split()[1])
            return kibibytes / 1024**2
    return None


def _gpu_environment() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except (RuntimeError, TypeError):
            free_bytes = 0
            total_bytes = properties.total_memory
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
                "total_memory_gib": total_bytes / 1024**3,
                "free_memory_gib": free_bytes / 1024**3,
            }
        )
    return devices


def inspect_environment() -> dict[str, Any]:
    affinity = _cpu_affinity_count()
    physical_cores = _physical_core_count()
    quota = _cgroup_cpu_limit()
    quota_threads = None if quota is None else max(1, math.floor(quota))
    usable_threads = min(
        affinity,
        physical_cores or affinity,
        quota_threads or affinity,
    )
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu": {
            "logical_visible": os.cpu_count() or 1,
            "affinity_threads": affinity,
            "physical_cores": physical_cores,
            "cgroup_cpu_limit": quota,
            "recommended_thread_budget": usable_threads,
            "torch_intraop_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "memory_total_gib": _memory_total_gib(),
        "gpus": _gpu_environment(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
    }


def recommend_parallelism(environment: dict[str, Any]) -> dict[str, Any]:
    cpu_budget = int(environment["cpu"]["recommended_thread_budget"])
    gpus = environment["gpus"]
    if gpus:
        selected_gpu = max(gpus, key=lambda item: item["free_memory_gib"])
        kgw_device = f"cuda:{selected_gpu['index']}"
    else:
        kgw_device = "cpu"

    rdf_intraop = min(4, cpu_budget)
    rdf_workers = max(1, min(8, cpu_budget // rdf_intraop))
    vow_workers = min(4, cpu_budget)
    kgw_workers = min(4, cpu_budget)
    return {
        "detect_vow_null": {
            "device": "cpu",
            "detector_batch_size": 256,
            "concurrent_batches": vow_workers,
            "max_in_flight_batches": max(2, vow_workers * 2),
            "intraop_threads": 1,
        },
        "detect_lefthash_null": {
            "device": kgw_device,
            "detector_batch_size": 8,
            "concurrent_batches": kgw_workers,
            "max_in_flight_batches": max(2, kgw_workers * 2),
            "intraop_threads": 1,
        },
        "detect_selfhash_null": {
            "device": kgw_device,
            "detector_batch_size": 8,
            "concurrent_batches": kgw_workers,
            "max_in_flight_batches": max(2, kgw_workers * 2),
            "intraop_threads": 1,
        },
        "detect_rdf_null": {
            "device": "cpu",
            "detector_batch_size": 1,
            "concurrent_batches": rdf_workers,
            "max_in_flight_batches": max(2, rdf_workers * 2),
            "intraop_threads": rdf_intraop,
        },
    }


def _load_windows(
    path: Path,
    tokenizer: Any,
    *,
    sample_num: int,
    token_num: int,
) -> tuple[list[list[int]], float]:
    texts = []
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            value = json.loads(line)
            text = value.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
    started = time.perf_counter()
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )["input_ids"]
    windows = [
        [int(value) for value in token_ids[:token_num]]
        for token_ids in encoded
        if len(token_ids) >= token_num
    ][:sample_num]
    elapsed = time.perf_counter() - started
    if len(windows) != sample_num:
        raise RuntimeError(
            f"{path} contains only {len(windows)} eligible {token_num}-token "
            f"windows; requested {sample_num}"
        )
    return windows, elapsed


def _p_value(result: Any) -> float:
    value = result.get("p_value") if isinstance(result, dict) else result.p_value
    return float(value)


def _synchronize(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))


def _progress(name: str, total: int):
    lock = threading.Lock()
    completed = 0
    interval = max(1, total // 10)

    def advance(amount: int) -> None:
        nonlocal completed
        with lock:
            previous = completed
            completed += amount
            if completed == total or completed // interval > previous // interval:
                print(f"[{name}] {completed}/{total}", flush=True)

    return advance


def _standard_config(method: str) -> dict[str, Any]:
    if method == "vow":
        return {
            "method": "vow",
            "enabled": True,
            "window_size": 4,
            "gamma": 0.5,
            "delta": 2.5,
            "server_seed_path": "data/server_seed",
            "naive_baseline": False,
        }
    return {
        "method": method,
        "enabled": True,
        "gamma": 0.25,
        "delta": 2.0,
    }


def benchmark_standard_detector(
    method: str,
    tokenizer: Any,
    windows: list[list[int]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    config = WATERMARK_SCHEMES.resolve(_standard_config(method), REPOSITORY)
    local = threading.local()
    device = str(settings["device"])
    batch_size = int(settings["detector_batch_size"])
    workers = int(settings["concurrent_batches"])
    batches = [
        windows[start : start + batch_size]
        for start in range(0, len(windows), batch_size)
    ]
    advance = _progress(method, len(windows))

    def get_detector():
        detector = getattr(local, "detector", None)
        if detector is None:
            detector, _ = WATERMARK_SCHEMES.detector(
                config,
                tokenizer,
                device=device,
            )
            if method in {"lefthash", "selfhash"}:
                detector.cache_ngram_scores = False
            local.detector = detector
        return detector

    def score(batch: list[list[int]]) -> list[Any]:
        detector = get_detector()
        if method == "vow":
            results = detector.local_batch_detect_tokens(batch)
        else:
            results = [
                detector.detect(
                    tokenized_text=torch.as_tensor(
                        token_ids,
                        dtype=torch.long,
                        device=detector.device,
                    ),
                    return_z_at_T=False,
                )
                for token_ids in batch
            ]
        advance(len(results))
        return results

    _synchronize(device)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        nested = list(executor.map(score, batches))
    _synchronize(device)
    elapsed = time.perf_counter() - started
    results = [result for batch in nested for result in batch]
    return _benchmark_summary(
        method,
        results,
        elapsed,
        settings=settings,
        setup_seconds=0.0,
    )


def _rdf_seed(index: int) -> int:
    digest = hashlib.sha256(f"rdf-benchmark\0{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def benchmark_rdf(
    tokenizer: Any,
    windows: list[list[int]],
    settings: dict[str, Any],
    *,
    n_runs: int,
) -> dict[str, Any]:
    config = WATERMARK_SCHEMES.resolve(
        {
            "method": "rdf",
            "enabled": True,
            "length": 256,
            "seed": 42,
            "watermark_device": "cpu",
            "n_runs": n_runs,
        },
        REPOSITORY,
    )
    detector, _ = WATERMARK_SCHEMES.detector(
        config,
        tokenizer,
        device="cpu",
    )
    started = time.perf_counter()
    detector.prepare_token_detection()
    setup_seconds = time.perf_counter() - started
    advance = _progress("rdf", len(windows))
    intraop_threads = int(settings["intraop_threads"])

    def score(item: tuple[int, list[int]]) -> Any:
        index, token_ids = item
        result = detector.detect_token_ids(
            token_ids,
            sample_seed=_rdf_seed(index),
            intraop_threads=intraop_threads,
        )
        advance(1)
        return result

    started = time.perf_counter()
    with ThreadPoolExecutor(
        max_workers=int(settings["concurrent_batches"])
    ) as executor:
        results = list(executor.map(score, enumerate(windows)))
    elapsed = time.perf_counter() - started
    summary = _benchmark_summary(
        "rdf",
        results,
        elapsed,
        settings=settings,
        setup_seconds=setup_seconds,
    )
    summary["n_runs"] = n_runs
    summary["keyed_matrix_mib"] = detector._keyed_matrix().nbytes / 1024**2
    return summary


def _benchmark_summary(
    method: str,
    results: list[Any],
    elapsed: float,
    *,
    settings: dict[str, Any],
    setup_seconds: float,
) -> dict[str, Any]:
    p_values = [_p_value(result) for result in results]
    plan_sample_num = 10_000 if method == "rdf" else 100_000
    return {
        "scheme": method,
        "sample_num": len(results),
        "elapsed_seconds": elapsed,
        "setup_seconds": setup_seconds,
        "samples_per_second": len(results) / elapsed,
        "milliseconds_per_sample_wall": elapsed * 1000 / len(results),
        "projected_plan_sample_num": plan_sample_num,
        "projected_plan_seconds": elapsed * plan_sample_num / len(results),
        "p_value": {
            "min": min(p_values),
            "median": statistics.median(p_values),
            "max": max(p_values),
        },
        "settings": dict(settings),
    }


def _override_recommendations(
    recommendations: dict[str, Any],
    arguments: argparse.Namespace,
) -> None:
    if arguments.kgw_device is not None:
        for name in ("detect_lefthash_null", "detect_selfhash_null"):
            recommendations[name]["device"] = arguments.kgw_device
    overrides = {
        "detect_vow_null": arguments.vow_workers,
        "detect_lefthash_null": arguments.kgw_workers,
        "detect_selfhash_null": arguments.kgw_workers,
        "detect_rdf_null": arguments.rdf_workers,
    }
    for name, workers in overrides.items():
        if workers is not None:
            recommendations[name]["concurrent_batches"] = workers
            recommendations[name]["max_in_flight_batches"] = workers * 2
    if arguments.rdf_intraop_threads is not None:
        recommendations["detect_rdf_null"]["intraop_threads"] = (
            arguments.rdf_intraop_threads
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--sample-num", type=_positive_int, default=1000)
    parser.add_argument("--rdf-sample-num", type=_positive_int)
    parser.add_argument("--token-num", type=_positive_int, default=200)
    parser.add_argument("--rdf-runs", type=_positive_int, default=100)
    parser.add_argument(
        "--schemes",
        nargs="+",
        choices=("vow", "lefthash", "selfhash", "rdf"),
        default=("vow", "lefthash", "selfhash", "rdf"),
    )
    parser.add_argument("--vow-workers", type=_positive_int)
    parser.add_argument("--kgw-workers", type=_positive_int)
    parser.add_argument("--rdf-workers", type=_positive_int)
    parser.add_argument("--rdf-intraop-threads", type=_positive_int)
    parser.add_argument("--kgw-device")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--environment-only", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    environment = inspect_environment()
    recommendations = recommend_parallelism(environment)
    _override_recommendations(recommendations, arguments)
    report: dict[str, Any] = {
        "environment": environment,
        "recommendations": recommendations,
        "benchmark": {},
    }
    print("Environment:")
    print(json.dumps(environment, indent=2, sort_keys=True))
    print("\nRecommended plan settings:")
    print(yaml.safe_dump(recommendations, sort_keys=False).rstrip())

    if not arguments.environment_only:
        tokenizer = AutoTokenizer.from_pretrained(
            arguments.tokenizer,
            local_files_only=arguments.local_files_only,
        )
        requested = max(arguments.sample_num, arguments.rdf_sample_num or 0)
        windows, tokenization_seconds = _load_windows(
            arguments.dataset,
            tokenizer,
            sample_num=requested,
            token_num=arguments.token_num,
        )
        report["input"] = {
            "dataset": str(arguments.dataset.resolve()),
            "tokenizer": arguments.tokenizer,
            "sample_num": arguments.sample_num,
            "rdf_sample_num": arguments.rdf_sample_num or arguments.sample_num,
            "token_num": arguments.token_num,
            "tokenization_seconds": tokenization_seconds,
            "vocabulary_size": len(tokenizer.get_vocab()),
        }
        for method in arguments.schemes:
            selected_num = (
                arguments.rdf_sample_num or arguments.sample_num
                if method == "rdf"
                else arguments.sample_num
            )
            selected_windows = windows[:selected_num]
            print(f"\nBenchmarking {method} on {selected_num} samples...", flush=True)
            if method == "rdf":
                result = benchmark_rdf(
                    tokenizer,
                    selected_windows,
                    recommendations["detect_rdf_null"],
                    n_runs=arguments.rdf_runs,
                )
            else:
                result = benchmark_standard_detector(
                    method,
                    tokenizer,
                    selected_windows,
                    recommendations[f"detect_{method}_null"],
                )
            report["benchmark"][method] = result
            print(json.dumps(result, indent=2, sort_keys=True), flush=True)

    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nWrote {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
