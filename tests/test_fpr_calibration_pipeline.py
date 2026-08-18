from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

from watermark_suite.experiments import ExperimentRuns
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
)
from watermark_suite.schemes.rdf.detection import permutation_test_compact


MODEL = {
    "checkpoint": "test-tokenizer",
    "revision": "test-revision",
    "location": "/models/test-tokenizer",
    "verification": {
        "kind": "huggingface-cache",
        "commit": "test-revision",
    },
    "tokenizer_checkpoint": "test-tokenizer",
    "tokenizer_revision": "test-revision",
    "tokenizer_location": "/models/test-tokenizer",
    "tokenizer_verification": {
        "kind": "huggingface-cache",
        "commit": "test-revision",
    },
}


class TinyTokenizer:
    bos_token_id = None

    def __call__(self, texts, **kwargs):
        assert kwargs["add_special_tokens"] is False
        assert kwargs["padding"] is False
        assert kwargs["truncation"] is False
        return {
            "input_ids": [
                [int(piece) for piece in text.split()] for text in texts
            ]
        }

    def get_vocab(self):
        return {str(index): index for index in range(64)}


class FakeRuntime:
    def __init__(self):
        self.value = TinyTokenizer()

    def tokenizer(self, model, *, padding_side):
        del model, padding_side
        return self.value

    def release(self):
        pass


class FakeStream:
    def __init__(self, records):
        self.records = records
        self.shuffle_call = None

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_call = (seed, buffer_size)
        return self

    def __iter__(self):
        yield from self.records


class ConcurrentDetector:
    def __init__(self, state):
        self.state = state

    def local_batch_detect_tokens(self, token_lists):
        with self.state["lock"]:
            self.state["active"] += 1
            self.state["max_active"] = max(
                self.state["max_active"], self.state["active"]
            )
        time.sleep(0.02)
        with self.state["lock"]:
            self.state["active"] -= 1
        return [
            {
                "total_token_num": len(tokens),
                "effective_token_num": len(tokens) - 1,
                "green_token_num": 1,
                "green_ratio": 1 / (len(tokens) - 1),
                "p_value": 0.005 if tokens[0] % 2 else 0.5,
            }
            for tokens in token_lists
        ]


class FailOnceDetector(ConcurrentDetector):
    def local_batch_detect_tokens(self, token_lists):
        with self.state["lock"]:
            if not self.state["failed"]:
                self.state["failed"] = True
                raise RuntimeError("synthetic concurrent batch failure")
        return super().local_batch_detect_tokens(token_lists)


def _plan(path: Path) -> Path:
    value = {
        "version": 1,
        "name": "tiny-fpr-calibration",
        "models": {"tokenizer": {"checkpoint": "test-tokenizer"}},
        "datasets": {
            "stream": {
                "kind": "huggingface-stream",
                "path": "allenai/c4",
                "config": "realnewslike",
                "split": "train",
                "revision": "fixed-revision",
                "text_field": "text",
            }
        },
        "stages": {
            "windows": {
                "kind": "token-window-corpus",
                "dataset": "stream",
                "tokenizer": "tokenizer",
                "sample_num": 6,
                "window_token_num": 4,
                "max_windows_per_document": 1,
                "selection_seed": 7,
                "shuffle_buffer_size": 3,
                "stream_batch_size": 2,
                "checkpoint_sample_num": 2,
            },
            "detect": {
                "kind": "null-detection",
                "input": "windows",
                "detector_watermark": {
                    "method": "vow",
                    "enabled": True,
                    "window_size": 1,
                    "gamma": 0.5,
                    "delta": 2.5,
                    "server_seed_path": "data/server_seed",
                    "naive_baseline": False,
                },
                "sample_num": 6,
                "detector_batch_size": 1,
                "concurrent_batches": 3,
                "max_in_flight_batches": 3,
                "intraop_threads": 1,
                "significance_levels": [0.01],
                "device": "cpu",
            },
            "assemble": {
                "kind": "result-aggregation",
                "inputs": ["detect"],
                "recipe": "fpr-calibration",
                "significance_levels": [0.01],
                "confidence_level": 0.95,
            },
        },
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def test_streamed_token_windows_feed_concurrent_detector_batches(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.null_detection as detection_module
    import watermark_suite.experiments.stages.token_window_corpus as corpus_module

    monkeypatch.setattr(corpus_module, "resolve_model", lambda *a, **k: MODEL)
    stream = FakeStream(
        [
            {
                "text": f"{index} 10 11 12 13 14",
                "url": f"https://example.test/{index}",
                "timestamp": "2026-01-01",
            }
            for index in range(1, 7)
        ]
    )
    monkeypatch.setattr(
        corpus_module, "load_streaming_dataset", lambda source: stream
    )
    state = {
        "lock": threading.Lock(),
        "active": 0,
        "max_active": 0,
    }
    monkeypatch.setattr(
        detection_module.WATERMARK_SCHEMES,
        "detector",
        lambda *args, **kwargs: (ConcurrentDetector(state), {}),
    )
    workspace = tmp_path / "workspace"
    runs = ExperimentRuns.local(
        repository=Path.cwd(),
        workspace=workspace,
        enforce_clean=False,
        runtime=FakeRuntime(),
    )

    report = runs.execute(_plan(tmp_path / "plan.yaml"))

    assert report.succeeded
    assert stream.shuffle_call == (7, 3)
    assert state["max_active"] >= 2
    status = runs.status(tmp_path / "plan.yaml")
    artifacts = {
        item["instance_name"]: item["canonical_artifact_identity"]
        for item in status.runs
    }
    corpus = runs.workspace.artifact(ArtifactIdentity(artifacts["windows"]))
    corpus_summary = json.loads(
        (corpus.path / "summary.json").read_text(encoding="utf-8")
    )
    assert corpus_summary["sample_num"] == 6
    assert corpus_summary["window"]["token_num"] == 4
    assert corpus_summary["retention"]["raw_text_retained"] is False
    part_records = [
        json.loads(line)
        for line in (corpus.path / "records.jsonl").read_text().splitlines()
    ]
    first_part = pq.read_table(corpus.path / part_records[0]["part"])
    assert first_part.column("token_ids").to_pylist()[0] == [1, 10, 11, 12]
    assert "text" not in first_part.column_names

    detection = runs.workspace.artifact(ArtifactIdentity(artifacts["detect"]))
    detection_summary = json.loads(
        (detection.path / "summary.json").read_text(encoding="utf-8")
    )
    assert detection_summary["false_positive_counts"]["1e-02"] == {
        "positive_num": 3,
        "sample_num": 6,
    }
    assert detection_summary["retention"]["sample_p_values_retained"] is True
    detection_records = [
        json.loads(line)
        for line in (detection.path / "records.jsonl").read_text().splitlines()
    ]
    assert len(detection_records) == 6
    assert all("p_value" in record["detection"] for record in detection_records)
    assert all(record["source_token_sha256"] for record in detection_records)
    assembled = runs.workspace.artifact(ArtifactIdentity(artifacts["assemble"]))
    row = json.loads(
        (assembled.path / "records.jsonl").read_text(encoding="utf-8")
    )
    assert row["value"] == 0.5
    assert row["uncertainty"]["method"] == "clopper-pearson"


def test_compact_rdf_permutation_is_schedule_independent():
    keyed_xi = np.random.default_rng(9).random((4, 8), dtype=np.float32)
    tokens = [1, 2, 1, 3]

    first = permutation_test_compact(
        tokens,
        keyed_xi=keyed_xi,
        n=4,
        n_runs=7,
        rng=np.random.default_rng(123),
        intraop_threads=1,
    )
    second = permutation_test_compact(
        tokens,
        keyed_xi=keyed_xi,
        n=4,
        n_runs=7,
        rng=np.random.default_rng(123),
        intraop_threads=2,
    )

    assert first == second
    assert first[0] == (first[1] + 1) / 8


def test_concurrent_null_detection_resumes_without_duplicate_samples(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.null_detection as detection_module
    import watermark_suite.experiments.stages.token_window_corpus as corpus_module

    monkeypatch.setattr(corpus_module, "resolve_model", lambda *a, **k: MODEL)
    records = [
        {"text": f"{index} 10 11 12 13", "url": str(index)}
        for index in range(1, 7)
    ]
    monkeypatch.setattr(
        corpus_module,
        "load_streaming_dataset",
        lambda source: FakeStream(records),
    )
    state = {
        "lock": threading.Lock(),
        "active": 0,
        "max_active": 0,
        "failed": False,
    }
    monkeypatch.setattr(
        detection_module.WATERMARK_SCHEMES,
        "detector",
        lambda *args, **kwargs: (FailOnceDetector(state), {}),
    )
    runs = ExperimentRuns.local(
        repository=Path.cwd(),
        workspace=tmp_path / "workspace",
        enforce_clean=False,
        runtime=FakeRuntime(),
    )
    plan = _plan(tmp_path / "plan.yaml")

    first = runs.execute(plan)
    second = runs.execute(plan)

    assert first.failed_stages == ("detect",)
    assert second.succeeded
    assert second.resumed_attempts
    status = runs.status(plan)
    detection_id = next(
        run["canonical_artifact_identity"]
        for run in status.runs
        if run["instance_name"] == "detect"
    )
    detection = runs.workspace.artifact(ArtifactIdentity(detection_id))
    sample_ids = [
        json.loads(line)["sample_id"]
        for line in (detection.path / "records.jsonl").read_text().splitlines()
    ]
    assert len(sample_ids) == len(set(sample_ids)) == 6


def test_fpr_assembler_marks_unattainable_rdf_level_unsupported(tmp_path):
    from watermark_suite.experiments.fpr_calibration import FprCalibrationAssembler

    artifact_path = tmp_path / "rdf"
    artifact_path.mkdir()
    summary = {
        "sample_num": 10,
        "population": {"identity": "population_test"},
        "window": {"token_num": 200},
        "source_artifact": "artifact_windows",
        "detector_scheme": {"method": "rdf"},
        "false_positive_counts": {
            "1e-02": {"positive_num": 0, "sample_num": 10}
        },
        "p_value": {"minimum_attainable": 1 / 101},
    }
    (artifact_path / "summary.json").write_text(json.dumps(summary))
    artifact = ArtifactRef(
        identity=ArtifactIdentity("artifact_rdf"),
        path=artifact_path,
        schema_revision="null-detection-v1",
        run_identity=RunIdentity("run_rdf"),
        attempt_identity=AttemptIdentity("attempt_rdf"),
        source_artifacts=(ArtifactIdentity("artifact_windows"),),
        manifest={
            "kind": "null-detection",
            "record_count": 10,
        },
    )

    rows, _ = FprCalibrationAssembler(
        (artifact,),
        significance_levels=[0.01, 0.001],
        confidence_level=0.95,
    ).assemble()

    by_alpha = {row["dimensions"]["nominal_alpha"]: row for row in rows}
    assert by_alpha[0.01]["status"] == "measured"
    assert by_alpha[0.001]["status"] == "unsupported"
    assert by_alpha[0.001]["value"] is None
    assert "minimum attainable" in by_alpha[0.001]["reason"]


def test_usenix_fpr_plan_uses_large_200_token_streaming_population():
    plan = yaml.safe_load(
        Path(
            "experiments/usenix-plans/fpr-calibration-qwen25-3b-c4.yaml"
        ).read_text(encoding="utf-8")
    )

    dataset = plan["datasets"]["c4-null-stream"]
    corpus = plan["stages"]["build_c4_null_windows"]
    vow = plan["stages"]["detect_vow_null"]
    lefthash = plan["stages"]["detect_lefthash_null"]
    selfhash = plan["stages"]["detect_selfhash_null"]
    rdf = plan["stages"]["detect_rdf_null"]
    assert dataset["kind"] == "huggingface-stream"
    assert dataset["config"] == "realnewslike"
    assert dataset["revision"] == "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"
    assert corpus["sample_num"] == 1000000
    assert corpus["window_token_num"] == 200
    assert vow["sample_num"] == 1000000
    assert vow["detector_batch_size"] == 2048
    assert lefthash["sample_num"] == 1000000
    assert lefthash["detector_batch_size"] == 256
    assert lefthash["concurrent_batches"] == 4
    assert selfhash["sample_num"] == 1000000
    assert selfhash["detector_batch_size"] == 256
    assert selfhash["concurrent_batches"] == 4
    assert rdf["sample_num"] == 100000
    assert rdf["detector_batch_size"] == 32
    assert rdf["concurrent_batches"] == 10
    assert rdf["intraop_threads"] == 4
    assert rdf["significance_levels"] == [0.01]
