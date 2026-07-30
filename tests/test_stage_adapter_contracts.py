from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_suite.experiments.adapters import ResolutionContext
from watermark_suite.experiments.errors import PlanValidationError
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
)
from watermark_suite.experiments.stages.adaptive_forgery import (
    AdaptiveForgeryStageAdapter,
)
from watermark_suite.experiments.stages.detection import (
    DetectionStageAdapter,
    _DetectionExecution,
)
from watermark_suite.experiments.stages.downstream import (
    DownstreamStageAdapter,
)
from watermark_suite.experiments.stages.generation import (
    GenerationStageAdapter,
)
from watermark_suite.experiments.stages.perplexity import (
    PerplexityStageAdapter,
)


MODEL = {
    "checkpoint": "model",
    "revision": "commit",
    "location": "/models/model",
    "tokenizer_checkpoint": "model",
    "tokenizer_revision": "commit",
    "tokenizer_location": "/models/model",
}
DATASET = {
    "kind": "c4",
    "selection": {"count": 2, "sample_ids": ["c4:1", "c4:2"]},
}
WATERMARK = {
    "method": "vow",
    "enabled": True,
    "window_size": 4,
    "delta": 2.5,
    "gamma": 0.375,
    "server_seed_path": "/data/server_seed",
    "server_seed_sha256": "seed-digest",
    "naive_baseline": False,
}


def context(tmp_path: Path) -> ResolutionContext:
    return ResolutionContext(
        repository=tmp_path,
        models={},
        datasets={},
        defaults={},
        code_revision="commit",
    )


def test_generation_contract_resolves_model_dataset_and_watermark(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.generation as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        module, "resolve_dataset", lambda *args, **kwargs: DATASET
    )
    monkeypatch.setattr(
        module, "resolve_watermark", lambda *args, **kwargs: WATERMARK
    )
    settings = {
        "model": "main",
        "dataset": "c4",
        "num_samples": 2,
        "batch_size": 2,
        "seed": 42,
        "max_new_tokens": 10,
        "do_sample": True,
        "top_p": None,
        "top_k": 50,
        "temperature": 0.7,
        "suppress_eos": False,
        "stop_strings": None,
        "watermark": {"method": "vow"},
        "device": "cuda",
        "dtype": "bfloat16",
    }

    resolved = GenerationStageAdapter().resolve(
        settings, context(tmp_path)
    )

    assert resolved.semantic_settings["model"] == MODEL
    assert resolved.semantic_settings["sample_manifest"] == DATASET["selection"]
    assert resolved.semantic_settings["watermark"] == WATERMARK
    assert resolved.execution_settings == {
        "device": "cuda",
        "dtype": "bfloat16",
    }


def test_adaptive_contract_excludes_irrelevant_delta_from_run_semantics(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.adaptive_forgery as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        module, "resolve_dataset", lambda *args, **kwargs: DATASET
    )
    monkeypatch.setattr(
        module, "resolve_watermark", lambda *args, **kwargs: WATERMARK
    )
    settings = {
        "model": "main",
        "dataset": "c4",
        "num_samples": 2,
        "batch_size": 1,
        "seed": 42,
        "max_new_tokens": 10,
        "max_candidates": 8,
        "watermark": {"method": "vow"},
        "allow_special_tokens": False,
        "trace_level": "compact",
        "device": "cuda",
        "dtype": "bfloat16",
    }

    resolved = AdaptiveForgeryStageAdapter().resolve(
        settings, context(tmp_path)
    )

    assert resolved.semantic_settings["watermark"]["gamma"] == 0.375
    assert "delta" not in resolved.semantic_settings["watermark"]
    assert "naive_baseline" not in resolved.semantic_settings["watermark"]


def test_detection_contract_derives_watermark_and_tokenizer_from_source():
    adapter = DetectionStageAdapter()
    definition = adapter.resolve(
        {
            "batch_size": 4,
            "target_field": "generated_text",
            "token_num": None,
            "significance_levels": [0.01],
            "use_local": True,
            "step_size": None,
            "device": "cpu",
        },
        ResolutionContext(
            repository=Path("/repo"),
            models={},
            datasets={},
            defaults={},
            code_revision="commit",
        ),
    )
    source = ArtifactRef(
        identity=ArtifactIdentity("artifact_source"),
        path=Path("/artifact"),
        schema_revision="generated-text-v1",
        run_identity=RunIdentity("run_source"),
        attempt_identity=AttemptIdentity("attempt_source"),
        manifest={
            "semantic_settings": {
                "model": MODEL,
                "watermark": WATERMARK,
            }
        },
    )

    bound = adapter.bind_inputs(definition, (source,))

    assert bound.semantic_settings["watermark"] == WATERMARK
    assert bound.semantic_settings["tokenizer"] == {
        "tokenizer_checkpoint": "model",
        "tokenizer_revision": "commit",
        "tokenizer_location": "/models/model",
    }


def test_detection_summary_aggregates_actual_milestones():
    execution = object.__new__(_DetectionExecution)
    execution.context = SimpleNamespace(
        semantic_settings={
            "significance_levels": [0.00001],
            "watermark": WATERMARK,
        }
    )
    records = [
        {
            "detection": {
                "p_value": 0.000001,
                "total_token_num": 10,
                "milestones": [5, 10],
                "step_p_values": [0.01, 0.000001],
            }
        },
        {
            "detection": {
                "p_value": 0.1,
                "total_token_num": 5,
                "milestones": [5],
                "step_p_values": [0.1],
            }
        },
    ]

    summary = execution.summarize(records)

    assert summary["milestone_detection_rate"] == [
        {
            "token_num": 5,
            "eligible_sample_num": 2,
            "detection_rate": {"1e-05": 0.0},
        },
        {
            "token_num": 10,
            "eligible_sample_num": 1,
            "detection_rate": {"1e-05": 1.0},
        },
    ]


def test_perplexity_contract_uses_evaluator_model_tokenizer(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.perplexity as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    resolved = PerplexityStageAdapter().resolve(
        {
            "model": "ppl",
            "batch_size": 2,
            "max_length": 2048,
            "prompt_field": "prompt_text",
            "target_field": "generated_text",
            "device": "cuda",
            "dtype": "bfloat16",
        },
        context(tmp_path),
    )

    assert resolved.semantic_settings["model"] == MODEL
    assert resolved.semantic_settings["tokenizer_policy"] == "evaluator-owned"

    source = ArtifactRef(
        identity=ArtifactIdentity("artifact_source"),
        path=Path("/artifact"),
        schema_revision="generated-text-v1",
        run_identity=RunIdentity("run_source"),
        attempt_identity=AttemptIdentity("attempt_source"),
        manifest={"semantic_settings": {"watermark": WATERMARK}},
    )
    bound = PerplexityStageAdapter().bind_inputs(resolved, (source,))

    assert bound.semantic_settings["watermark"] == WATERMARK


def test_downstream_contract_pins_full_task_and_greedy_decoding(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.downstream as module

    task_dataset = {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "sample_num": 1319,
        "sample_ids": ["gsm8k:0"],
        "snapshot": "dataset_snapshot",
    }
    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        module, "resolve_watermark", lambda *args, **kwargs: WATERMARK
    )
    monkeypatch.setattr(
        module, "resolve_task_dataset", lambda task: task_dataset
    )

    resolved = DownstreamStageAdapter().resolve(
        {
            "task": "gsm8k",
            "model": "instruct",
            "batch_size": 8,
            "seed": 42,
            "max_new_tokens": 1024,
            "num_shots": 4,
            "watermark": {"method": "vow"},
            "evaluation_workers": 4,
            "execution_timeout": 3.0,
            "device": "cuda",
            "dtype": "bfloat16",
        },
        context(tmp_path),
    )

    assert resolved.semantic_settings["dataset"] == task_dataset
    assert resolved.semantic_settings["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
    }
    assert resolved.semantic_settings["task_evaluation"]["num_shots"] == 4
    assert "execution_timeout" not in resolved.semantic_settings[
        "task_evaluation"
    ]
    assert resolved.execution_settings["evaluation_workers"] == 4


def test_downstream_contract_rejects_wrong_shot_count(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.downstream as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        module, "resolve_watermark", lambda *args, **kwargs: WATERMARK
    )
    monkeypatch.setattr(
        module,
        "resolve_task_dataset",
        lambda task: {"sample_num": 164},
    )

    with pytest.raises(PlanValidationError, match="num_shots=0"):
        DownstreamStageAdapter().resolve(
            {
                "task": "humaneval",
                "model": "instruct",
                "batch_size": 8,
                "seed": 42,
                "max_new_tokens": 1024,
                "num_shots": 4,
                "watermark": {"method": "vow"},
                "evaluation_workers": 4,
                "execution_timeout": 3.0,
                "device": "cuda",
                "dtype": "bfloat16",
            },
            context(tmp_path),
        )
