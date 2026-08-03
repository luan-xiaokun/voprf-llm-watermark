from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from watermark_suite.experiments.adapters import ResolutionContext
from watermark_suite.experiments.dataset_prompt import (
    DATASET_PROMPTS,
    ResolvedPromptPopulation,
)
from watermark_suite.experiments.errors import PlanValidationError
from watermark_suite.experiments.identity import identity_for
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
)
from watermark_suite.experiments.scheme_registry import WATERMARK_SCHEMES
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
    "verification": {"kind": "huggingface-cache", "commit": "commit"},
    "tokenizer_checkpoint": "model",
    "tokenizer_revision": "commit",
    "tokenizer_location": "/models/model",
    "tokenizer_verification": {
        "kind": "huggingface-cache",
        "commit": "commit",
    },
}
DATASET = {
    "kind": "c4",
    "split": "train",
    "snapshot": "dataset-snapshot",
    "selection": {"count": 2, "sample_ids": ["c4:0", "c4:1"]},
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


def prompt_population(
    kind: str = "c4",
    *,
    count: int = 2,
) -> ResolvedPromptPopulation:
    parameters = (
        {"num_shots": 4 if kind == "gsm8k" else 0}
        if kind in {"gsm8k", "humaneval"}
        else {}
    )
    policy = {
        "name": f"{kind}-test",
        "revision": f"{kind}-test-v1",
        "parameters": parameters,
    }
    return ResolvedPromptPopulation(
        kind=kind,
        dataset={
            "kind": kind,
            "split": (
                "test" if kind in {"gsm8k", "humaneval"} else "train"
            ),
            "snapshot": "dataset-snapshot",
            "selection": {
                "count": count,
                "sample_ids": [
                    f"{kind}:{index}" for index in range(count)
                ],
            },
        },
        source={"kind": "test"},
        prompt_policy={
            **policy,
            "identity": identity_for(policy, prefix="prompt-policy"),
        },
        generation=(
            {"stop_strings": ["Question:"]}
            if kind == "gsm8k"
            else {}
        ),
    )


def context(tmp_path: Path) -> ResolutionContext:
    return ResolutionContext(
        repository=tmp_path,
        models={},
        datasets={},
        defaults={},
        code_revision="commit",
    )


def test_watermark_scheme_registry_is_the_single_paired_interface(tmp_path):
    assert WATERMARK_SCHEMES.methods == (
        "lefthash",
        "none",
        "pdw",
        "rdf",
        "selfhash",
        "upv",
        "vow",
    )

    resolved = WATERMARK_SCHEMES.resolve(
        {"method": "none", "enabled": False},
        tmp_path,
    )

    assert resolved == {"method": "none", "enabled": False}


def test_generation_contract_resolves_model_dataset_and_watermark(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.generation as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        DATASET_PROMPTS,
        "resolve",
        lambda *args, **kwargs: prompt_population(),
    )
    monkeypatch.setattr(
        module.WATERMARK_SCHEMES,
        "resolve",
        lambda *args, **kwargs: WATERMARK,
    )
    settings = {
        "model": "main",
        "dataset": "c4",
        "num_samples": 2,
        "repetitions": 1,
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
    assert resolved.semantic_settings["prompt_population"]["dataset"] == DATASET
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
        DATASET_PROMPTS,
        "resolve",
        lambda *args, **kwargs: prompt_population(),
    )
    monkeypatch.setattr(
        module.WATERMARK_SCHEMES,
        "resolve",
        lambda *args, **kwargs: WATERMARK,
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
        schema_revision="generated-text-v3",
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
        "tokenizer_verification": {
            "kind": "huggingface-cache",
            "commit": "commit",
        },
    }


def test_upv_detection_binding_discards_generic_p_value_thresholds():
    adapter = DetectionStageAdapter()
    definition = adapter.resolve(
        {
            "batch_size": 4,
            "target_field": "generated_text",
            "token_num": None,
            "significance_levels": [0.00001],
            "use_local": True,
            "step_size": 10,
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
    watermark = {
        "method": "upv",
        "enabled": True,
        "detection_operating_point": {
            "kind": "classifier-threshold",
            "decision_threshold": 0.5,
        },
    }
    source = ArtifactRef(
        identity=ArtifactIdentity("artifact_upv"),
        path=Path("/artifact"),
        schema_revision="generated-text-v3",
        run_identity=RunIdentity("run_upv"),
        attempt_identity=AttemptIdentity("attempt_upv"),
        manifest={
            "semantic_settings": {
                "model": MODEL,
                "watermark": watermark,
            }
        },
    )

    bound = adapter.bind_inputs(definition, (source,))

    assert bound.semantic_settings["significance_levels"] == []
    assert bound.semantic_settings["watermark"] == watermark


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
            "detection_counts": {
                "1e-05": {"positive_num": 0, "sample_num": 2}
            },
        },
        {
            "token_num": 10,
            "eligible_sample_num": 1,
            "detection_rate": {"1e-05": 1.0},
            "detection_counts": {
                "1e-05": {"positive_num": 1, "sample_num": 1}
            },
        },
    ]


def test_upv_detection_summary_uses_classifier_decisions_and_empirical_fpr():
    operating_point = {
        "schema_revision": "upv-classifier-calibration-v1",
        "kind": "classifier-threshold",
        "score_type": "classifier_confidence",
        "decision_operator": ">",
        "decision_threshold": 0.5,
        "empirical_fpr": {
            "positive_num": 7920,
            "sample_num": 1_000_000,
            "rate": 0.00792,
        },
        "private_detector_sha256": "private-detector",
        "calibration": {"token_num": 255},
    }
    execution = object.__new__(_DetectionExecution)
    execution.context = SimpleNamespace(
        semantic_settings={
            "significance_levels": [0.00001],
            "watermark": {
                "method": "upv",
                "enabled": True,
                "detection_operating_point": operating_point,
            },
        }
    )
    records = [
        {
            "detection": {
                "p_value": None,
                "confidence": 0.9,
                "score_type": "classifier_confidence",
                "decision_threshold": 0.5,
                "predicted": True,
                "total_token_num": 10,
                "milestones": [5, 10],
                "step_scores": [0.4, 0.9],
                "step_predictions": [False, True],
            }
        },
        {
            "detection": {
                "p_value": None,
                "confidence": 0.1,
                "score_type": "classifier_confidence",
                "decision_threshold": 0.5,
                "predicted": False,
                "total_token_num": 5,
                "milestones": [5],
                "step_scores": [0.1],
                "step_predictions": [False],
            }
        },
    ]

    summary = execution.summarize(records)

    operating_point_id = "classifier_confidence>0.5"
    assert summary["detection_operating_points"] == {
        operating_point_id: operating_point,
    }
    assert summary["detection_counts"] == {
        operating_point_id: {"positive_num": 1, "sample_num": 2},
    }
    assert summary["classifier_confidence_distribution"]["median"] == 0.5
    assert "p_value_distribution" not in summary
    assert summary["milestone_detection_rate"] == [
        {
            "token_num": 5,
            "eligible_sample_num": 2,
            "detection_rate": {operating_point_id: 0.0},
            "detection_counts": {
                operating_point_id: {
                    "positive_num": 0,
                    "sample_num": 2,
                }
            },
        },
        {
            "token_num": 10,
            "eligible_sample_num": 1,
            "detection_rate": {operating_point_id: 1.0},
            "detection_counts": {
                operating_point_id: {
                    "positive_num": 1,
                    "sample_num": 1,
                }
            },
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
        schema_revision="generated-text-v3",
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

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(
        module.WATERMARK_SCHEMES,
        "resolve",
        lambda *args, **kwargs: WATERMARK,
    )
    monkeypatch.setattr(
        DATASET_PROMPTS,
        "resolve",
        lambda *args, **kwargs: prompt_population("gsm8k", count=1319),
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

    assert resolved.semantic_settings["prompt_population"]["kind"] == "gsm8k"
    assert resolved.semantic_settings["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
    }
    assert resolved.semantic_settings["prompt_population"]["prompt_policy"][
        "parameters"
    ]["num_shots"] == 4
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
        module.WATERMARK_SCHEMES,
        "resolve",
        lambda *args, **kwargs: WATERMARK,
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
