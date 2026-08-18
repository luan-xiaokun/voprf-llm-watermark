from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path("experiments/benchmark_fpr_calibration.py").resolve()
SPEC = importlib.util.spec_from_file_location(
    "benchmark_fpr_calibration", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)
_load_windows = BENCHMARK._load_windows
_override_recommendations = BENCHMARK._override_recommendations
recommend_parallelism = BENCHMARK.recommend_parallelism


class TinyTokenizer:
    def __call__(self, texts, **kwargs):
        assert kwargs == {
            "add_special_tokens": False,
            "padding": False,
            "truncation": False,
        }
        return {
            "input_ids": [
                [int(piece) for piece in text.split()] for text in texts
            ]
        }


def _environment(cpu_budget, gpus=()):
    return {
        "cpu": {"recommended_thread_budget": cpu_budget},
        "gpus": list(gpus),
    }


def test_recommendations_fill_a_32_thread_server_without_oversubscription():
    recommendations = recommend_parallelism(
        _environment(
            32,
            (
                {"index": 0, "free_memory_gib": 10.0},
                {"index": 1, "free_memory_gib": 20.0},
            ),
        )
    )

    rdf = recommendations["detect_rdf_null"]
    assert rdf["concurrent_batches"] == 8
    assert rdf["intraop_threads"] == 4
    assert rdf["concurrent_batches"] * rdf["intraop_threads"] == 32
    assert recommendations["detect_lefthash_null"]["device"] == "cuda:1"
    assert recommendations["detect_selfhash_null"]["device"] == "cuda:1"


def test_recommendations_scale_down_to_a_small_cpu_host():
    recommendations = recommend_parallelism(_environment(2))

    rdf = recommendations["detect_rdf_null"]
    assert rdf["concurrent_batches"] == 1
    assert rdf["intraop_threads"] == 2
    assert recommendations["detect_vow_null"]["concurrent_batches"] == 2
    assert recommendations["detect_lefthash_null"]["device"] == "cpu"


def test_explicit_overrides_are_used_by_the_benchmark():
    recommendations = recommend_parallelism(_environment(32))
    arguments = argparse.Namespace(
        kgw_device="cuda:0",
        vow_workers=2,
        kgw_workers=3,
        rdf_workers=5,
        rdf_intraop_threads=2,
    )

    _override_recommendations(recommendations, arguments)

    assert recommendations["detect_vow_null"]["concurrent_batches"] == 2
    assert recommendations["detect_lefthash_null"]["concurrent_batches"] == 3
    assert recommendations["detect_selfhash_null"]["device"] == "cuda:0"
    assert recommendations["detect_rdf_null"]["concurrent_batches"] == 5
    assert recommendations["detect_rdf_null"]["intraop_threads"] == 2


def test_window_loader_keeps_only_fixed_length_eligible_windows(tmp_path):
    path = tmp_path / "c4.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"text": text})
            for text in ("1 2", "3 4 5 6", "7 8 9 10")
        )
        + "\n",
        encoding="utf-8",
    )

    windows, elapsed = _load_windows(
        path,
        TinyTokenizer(),
        sample_num=2,
        token_num=3,
    )

    assert windows == [[3, 4, 5], [7, 8, 9]]
    assert elapsed >= 0
