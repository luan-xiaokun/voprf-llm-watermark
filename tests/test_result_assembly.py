from __future__ import annotations

import json
from pathlib import Path

import pytest

from watermark_suite.experiments.adapters import (
    ResolutionContext,
    StageExecutionContext,
)
from watermark_suite.experiments.artifact_interpretation import (
    interpret_artifacts,
)
from watermark_suite.experiments.errors import PlanValidationError
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
)
from watermark_suite.experiments.result_assembly import ResultAssembler
from watermark_suite.experiments.stages.result_aggregation import (
    ResultAggregationStageAdapter,
)
from watermark_suite.experiments.workspace import ExperimentWorkspace


MODEL = {
    "checkpoint": "Qwen/Qwen2.5-7B",
    "revision": "model-commit",
}
EVAL_MODEL = {
    "checkpoint": "Qwen/Qwen2.5-14B",
    "revision": "eval-commit",
}
VOW = {
    "method": "vow",
    "enabled": True,
    "window_size": 4,
    "gamma": 0.5,
    "delta": 2.5,
    "naive_baseline": False,
}
UPV_OPERATING_POINT = {
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
    "calibration": {
        "dataset": "allenai/c4:realnewslike",
        "split": "train",
        "tokenizer": "Qwen/Qwen2.5-0.5B",
        "token_num": 255,
    },
}
UPV = {
    "method": "upv",
    "enabled": True,
    "window_size": 4,
    "gamma": 0.5,
    "delta": 2.0,
    "bit_number": 18,
    "layers": 5,
    "beam_size": 0,
    "detection_operating_point": UPV_OPERATING_POINT,
}
NONE = {"method": "none", "enabled": False}
DATASET = {
    "kind": "c4",
    "split": "train",
    "snapshot": "dataset-snapshot",
    "selection": {
        "count": 2,
        "sample_ids": ["c4:1", "c4:2"],
    },
}
GENERATION = {
    "max_new_tokens": 210,
    "do_sample": True,
    "top_p": None,
    "top_k": None,
    "temperature": 0.7,
    "suppress_eos": False,
    "stop_strings": None,
}
PROMPT_POPULATION = {
    "revision": "dataset-prompt-v1",
    "kind": "c4",
    "dataset": DATASET,
    "source": {"format": "jsonl", "path": "/data/c4.jsonl"},
    "prompt_policy": {
        "name": "c4-plain-text",
        "revision": "c4-plain-text-v1",
        "parameters": {"source_field": "prompt_text"},
        "identity": "prompt-policy-c4",
    },
    "generation": {},
}


def artifact(
    root: Path,
    name: str,
    *,
    kind: str,
    schema: str,
    semantic: dict,
    summary: dict,
    sources: tuple[ArtifactRef, ...] = (),
    records: list[dict] | None = None,
) -> ArtifactRef:
    identity = f"artifact_{name}"
    path = root / "artifacts" / identity
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    record_count = int(summary.get("sample_num", 0))
    if records is None:
        records = [
            {"sample_id": f"{name}:{index}"}
            for index in range(record_count)
        ]
    (path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "kind": kind,
        "artifact_schema_revision": schema,
        "semantic_settings": semantic,
        "source_artifacts": [source.identity.value for source in sources],
        "record_count": len(records),
        "files": {},
    }
    (path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return ArtifactRef(
        identity=ArtifactIdentity(identity),
        path=path,
        schema_revision=schema,
        run_identity=RunIdentity(f"run_{name}"),
        attempt_identity=AttemptIdentity(f"attempt_{name}"),
        source_artifacts=tuple(source.identity for source in sources),
        manifest=manifest,
    )


def generation(
    root: Path,
    name: str,
    watermark: dict,
) -> ArtifactRef:
    return artifact(
        root,
        name,
        kind="generation",
        schema="generated-text-v3",
        semantic={
            "model": MODEL,
            "prompt_population": PROMPT_POPULATION,
            "generation": GENERATION,
            "watermark": watermark,
        },
        summary={"sample_num": 2},
    )


def detection(
    root: Path,
    name: str,
    source: ArtifactRef,
    *,
    watermark: dict = VOW,
    detector_watermark: dict | None = None,
    positive: int = 1,
    token_num: int | None = 100,
) -> ArtifactRef:
    detector_watermark = detector_watermark or watermark
    p_values = [
        0.000001 if index < positive else 0.5
        for index in range(2)
    ]
    milestones = []
    if token_num is not None:
        milestones.append(
            {
                "token_num": token_num,
                "eligible_sample_num": 2,
                "detection_counts": {
                    "1e-05": {
                        "positive_num": positive,
                        "sample_num": 2,
                    },
                    "1e-02": {
                        "positive_num": positive,
                        "sample_num": 2,
                    },
                },
            }
        )
    return artifact(
        root,
        name,
        kind="detection",
        schema="watermark-detection-v4",
        semantic={
            "watermark": watermark,
            "detector_watermark": detector_watermark,
        },
        sources=(source,),
        records=[
            {
                "sample_id": f"{name}:{index}",
                "detection": {"p_value": p_value},
            }
            for index, p_value in enumerate(p_values)
        ],
        summary={
            "sample_num": 2,
            "watermark": watermark,
            "detector_watermark": detector_watermark,
            "detection_operating_points": {
                "1e-05": {
                    "kind": "p-value-threshold",
                    "score_type": "p_value",
                    "decision_operator": "<",
                    "decision_threshold": 0.00001,
                    "target_fpr": 0.00001,
                },
                "1e-02": {
                    "kind": "p-value-threshold",
                    "score_type": "p_value",
                    "decision_operator": "<",
                    "decision_threshold": 0.01,
                    "target_fpr": 0.01,
                },
            },
            "detection_counts": {
                "1e-05": {"positive_num": positive, "sample_num": 2},
                "1e-02": {"positive_num": positive, "sample_num": 2},
            },
            "p_value_distribution": {
                "count": 2,
                "mean": sum(p_values) / 2,
                "median": sum(p_values) / 2,
                "min": min(p_values),
                "max": max(p_values),
            },
            "milestone_detection_rate": milestones,
        },
    )


def upv_detection(
    root: Path,
    name: str,
    source: ArtifactRef,
) -> ArtifactRef:
    operating_point_id = "classifier_confidence>0.5"
    return artifact(
        root,
        name,
        kind="detection",
        schema="watermark-detection-v4",
        semantic={"watermark": UPV},
        sources=(source,),
        summary={
            "sample_num": 2,
            "watermark": UPV,
            "detection_operating_points": {
                operating_point_id: UPV_OPERATING_POINT,
            },
            "detection_rate": {operating_point_id: 0.5},
            "detection_counts": {
                operating_point_id: {
                    "positive_num": 1,
                    "sample_num": 2,
                },
            },
            "classifier_confidence_distribution": {
                "count": 2,
                "mean": 0.5,
                "median": 0.5,
                "min": 0.1,
                "max": 0.9,
            },
            "milestone_detection_rate": [
                {
                    "token_num": 100,
                    "eligible_sample_num": 2,
                    "detection_rate": {operating_point_id: 0.5},
                    "detection_counts": {
                        operating_point_id: {
                            "positive_num": 1,
                            "sample_num": 2,
                        }
                    },
                }
            ],
        },
    )


def perplexity(
    root: Path,
    name: str,
    source: ArtifactRef,
    watermark: dict,
    value: float,
) -> ArtifactRef:
    return artifact(
        root,
        name,
        kind="perplexity",
        schema="conditional-perplexity-v2",
        semantic={"model": EVAL_MODEL, "watermark": watermark},
        sources=(source,),
        summary={
            "sample_num": 2,
            "watermark": watermark,
            "conditional_perplexity": value,
            "sample_conditional_perplexity": {
                "count": 2,
                "mean": value,
                "median": value,
            },
        },
    )


def assembler(
    tmp_path: Path,
    all_artifacts: tuple[ArtifactRef, ...],
    roots: tuple[ArtifactRef, ...],
) -> ResultAssembler:
    register_artifacts(tmp_path, all_artifacts)
    return ResultAssembler(
        interpret_artifacts(tmp_path, roots),
        confidence_level=0.95,
        threshold_policy={"default": 0.00001, "rdf": 0.01},
    )


def assemble_stage(
    tmp_path: Path,
    all_artifacts: tuple[ArtifactRef, ...],
    roots: tuple[ArtifactRef, ...],
    *,
    recipe: str,
) -> tuple[list[dict], dict]:
    register_artifacts(tmp_path, all_artifacts)
    adapter = ResultAggregationStageAdapter()
    definition = adapter.resolve(
        {
            "recipe": recipe,
            "confidence_level": 0.95,
            "threshold_policy": {
                "default": 0.00001,
                "rdf": 0.01,
            },
        },
        ResolutionContext(
            repository=tmp_path,
            models={},
            datasets={},
            defaults={},
            code_revision="commit",
        ),
    )
    definition = adapter.bind_inputs(definition, roots)
    execution = adapter.prepare(
        StageExecutionContext(
            repository=tmp_path,
            workspace=tmp_path,
            stage_name="assemble",
            run_identity=f"run_assemble_{recipe}",
            attempt_identity=f"attempt_assemble_{recipe}",
            settings=definition.settings,
            semantic_settings=definition.semantic_settings,
            execution_settings=definition.execution_settings,
            inputs=roots,
            runtime=None,
        )
    )
    result = execution.execute(execution.work_items()[0])
    return list(result.records), execution.summarize(list(result.records))


def register_artifacts(
    root: Path,
    artifacts: tuple[ArtifactRef, ...],
) -> None:
    workspace = ExperimentWorkspace(root)
    with workspace.connect() as connection:
        for ref in artifacts:
            manifest_json = json.dumps(ref.manifest, sort_keys=True)
            connection.execute(
                """
                INSERT INTO runs
                    (run_identity, stage_name, stage_kind, canonical_json,
                     canonical_artifact_identity, created_at)
                VALUES (?, ?, ?, '{}', ?, 'now')
                """,
                (
                    ref.run_identity.value,
                    ref.manifest["kind"],
                    ref.manifest["kind"],
                    ref.identity.value,
                ),
            )
            connection.execute(
                """
                INSERT INTO attempts
                    (attempt_identity, run_identity, state, provenance_json,
                     error_json, created_at, updated_at)
                VALUES (?, ?, 'succeeded', '{}', NULL, 'now', 'now')
                """,
                (ref.attempt_identity.value, ref.run_identity.value),
            )
            connection.execute(
                """
                INSERT INTO artifacts
                    (artifact_identity, run_identity, attempt_identity,
                     schema_revision, path, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'now')
                """,
                (
                    ref.identity.value,
                    ref.run_identity.value,
                    ref.attempt_identity.value,
                    ref.schema_revision,
                    str(ref.path.relative_to(root)),
                    manifest_json,
                ),
            )


def test_tpr_token_report_emits_exact_counts_and_uncertainty(tmp_path):
    generated = generation(tmp_path, "generated", VOW)
    detected = detection(tmp_path, "detected", generated)

    rows, summary = assemble_stage(
        tmp_path,
        (generated, detected),
        (detected,),
        recipe="tpr-vs-token-length",
    )

    assert len(rows) == 1
    assert rows[0]["metric"] == "true_positive_rate"
    assert rows[0]["value"] == 0.5
    assert rows[0]["population"] == {
        "sample_num": 2,
        "positive_num": 1,
    }
    assert rows[0]["dimensions"]["token_num"] == 100
    assert rows[0]["dimensions"]["target_fpr"] == 0.00001
    assert rows[0]["uncertainty"]["method"] == "wilson"
    assert summary["source_artifacts"] == ["artifact_detected"]


def test_upv_tpr_report_uses_its_empirical_classifier_operating_point(
    tmp_path,
):
    generated = generation(tmp_path, "generated_upv", UPV)
    detected = upv_detection(tmp_path, "detected_upv", generated)

    rows, summary = assemble_stage(
        tmp_path,
        (generated, detected),
        (detected,),
        recipe="tpr-vs-token-length",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["value"] == 0.5
    assert row["dimensions"]["target_fpr"] is None
    operating_point = row["dimensions"]["detection_operating_point"]
    assert operating_point["score_type"] == "classifier_confidence"
    assert operating_point["decision_threshold"] == 0.5
    assert operating_point["empirical_fpr"] == {
        "positive_num": 7920,
        "sample_num": 1_000_000,
        "rate": 0.00792,
    }
    assert summary["detection_operating_points"] == [UPV_OPERATING_POINT]


def test_tpr_ppl_report_joins_by_lineage_and_keeps_baseline(tmp_path):
    watermarked = generation(tmp_path, "watermarked", VOW)
    detected = detection(tmp_path, "detected", watermarked, positive=2)
    watermarked_ppl = perplexity(
        tmp_path, "watermarked_ppl", detected, VOW, 6.25
    )
    baseline = generation(tmp_path, "baseline", NONE)
    baseline_ppl = perplexity(
        tmp_path, "baseline_ppl", baseline, NONE, 5.0
    )
    all_artifacts = (
        watermarked,
        detected,
        watermarked_ppl,
        baseline,
        baseline_ppl,
    )

    rows, _ = assemble_stage(
        tmp_path,
        all_artifacts,
        (detected, watermarked_ppl, baseline_ppl),
        recipe="tpr-vs-ppl",
    )

    assert sorted(
        (row["dimensions"]["scheme"], row["metric"], row["value"])
        for row in rows
    ) == [
        ("none", "conditional_perplexity", 5.0),
        ("vow", "conditional_perplexity", 6.25),
        ("vow", "true_positive_rate", 1.0),
    ]
    tpr = next(row for row in rows if row["metric"] == "true_positive_rate")
    assert tpr["source_artifacts"] == [
        "artifact_detected",
        "artifact_watermarked_ppl",
    ]


def test_interpretation_loads_transitive_ancestors_from_ledger(tmp_path):
    watermarked = generation(tmp_path, "watermarked", VOW)
    detected = detection(tmp_path, "detected", watermarked, positive=2)
    watermarked_ppl = perplexity(
        tmp_path, "watermarked_ppl", detected, VOW, 6.25
    )
    baseline = generation(tmp_path, "baseline", NONE)
    baseline_ppl = perplexity(
        tmp_path, "baseline_ppl", baseline, NONE, 5.0
    )
    all_artifacts = (
        watermarked,
        detected,
        watermarked_ppl,
        baseline,
        baseline_ppl,
    )
    register_artifacts(tmp_path, all_artifacts)
    direct = (detected, watermarked_ppl, baseline_ppl)
    interpretations = interpret_artifacts(tmp_path, direct)
    result = ResultAssembler(
        interpretations,
        confidence_level=0.95,
        threshold_policy={"default": 0.00001, "rdf": 0.01},
    )

    rows, _ = result.assemble(recipe="tpr-vs-ppl")
    expected_population = interpretations.artifact(
        watermarked_ppl.identity
    ).dimensions.population_identity

    assert len(rows) == 3
    assert {row["dimensions"]["population_identity"] for row in rows} == {
        expected_population
    }


def test_result_aggregation_stage_executes_and_resumes_from_sources(tmp_path):
    generated = generation(tmp_path, "generated", VOW)
    detected = detection(tmp_path, "detected", generated)
    register_artifacts(tmp_path, (generated, detected))
    adapter = ResultAggregationStageAdapter()
    definition = adapter.resolve(
        {
            "recipe": "tpr-vs-token-length",
            "confidence_level": 0.95,
            "threshold_policy": {
                "default": 0.00001,
                "rdf": 0.01,
            },
        },
        ResolutionContext(
            repository=tmp_path,
            models={},
            datasets={},
            defaults={},
            code_revision="commit",
        ),
    )
    definition = adapter.bind_inputs(definition, (detected,))
    execution_context = StageExecutionContext(
        repository=tmp_path,
        workspace=tmp_path,
        stage_name="assemble",
        run_identity="run_assemble",
        attempt_identity="attempt_assemble",
        settings=definition.settings,
        semantic_settings=definition.semantic_settings,
        execution_settings=definition.execution_settings,
        inputs=(detected,),
        runtime=None,
    )

    execution = adapter.prepare(execution_context)
    result = execution.execute(execution.work_items()[0])
    summary = execution.summarize(list(result.records))
    resumed = adapter.prepare(execution_context)
    resumed_summary = resumed.summarize(list(result.records))

    assert summary == resumed_summary
    assert summary["recipe"] == "tpr-vs-token-length"
    assert summary["metric_row_num"] == 1
    assert result.records[0]["source_artifacts"] == ["artifact_detected"]


def test_tpr_ppl_rejects_an_unpaired_detection(tmp_path):
    generated = generation(tmp_path, "generated", VOW)
    detected = detection(tmp_path, "detected", generated)
    other_detection = detection(tmp_path, "other_detected", generated)
    ppl = perplexity(tmp_path, "ppl", detected, VOW, 6.0)
    baseline = generation(tmp_path, "baseline", NONE)
    baseline_ppl = perplexity(
        tmp_path, "baseline_ppl", baseline, NONE, 5.0
    )
    all_artifacts = (
        generated,
        detected,
        other_detection,
        ppl,
        baseline,
        baseline_ppl,
    )

    with pytest.raises(PlanValidationError, match="did not pair"):
        assembler(
            tmp_path,
            all_artifacts,
            (detected, other_detection, ppl, baseline_ppl),
        ).assemble(recipe="tpr-vs-ppl")


def test_comparative_report_rejects_mixed_generation_contexts(tmp_path):
    first_generation = generation(tmp_path, "first_generation", VOW)
    second_generation = artifact(
        tmp_path,
        "second_generation",
        kind="generation",
        schema="generated-text-v3",
        semantic={
            "model": {
                "checkpoint": "Qwen/Qwen2.5-3B",
                "revision": "other-model-commit",
            },
            "prompt_population": PROMPT_POPULATION,
            "generation": GENERATION,
            "watermark": VOW,
        },
        summary={"sample_num": 2},
    )
    first_detection = detection(
        tmp_path, "first_detection", first_generation
    )
    second_detection = detection(
        tmp_path, "second_detection", second_generation
    )

    with pytest.raises(
        PlanValidationError,
        match="requires one generation context",
    ):
        assembler(
            tmp_path,
            (
                first_generation,
                second_generation,
                first_detection,
                second_detection,
            ),
            (first_detection, second_detection),
        ).assemble(recipe="tpr-vs-token-length")


def test_downstream_report_normalizes_accuracy_and_pass_at_1(tmp_path):
    gsm = artifact(
        tmp_path,
        "gsm",
        kind="downstream",
        schema="gsm8k-evaluation-v2",
        semantic={
            "task": "gsm8k",
            "model": MODEL,
            "prompt_population": {
                "dataset": {
                    "name": "gsm8k",
                    "split": "test",
                    "snapshot": "gsm",
                    "selection": {
                        "count": 2,
                        "sample_ids": ["gsm:1", "gsm:2"],
                    },
                },
                "prompt_policy": {
                    "name": "gsm8k-policy",
                    "revision": "gsm8k-policy-v1",
                    "parameters": {"num_shots": 4},
                },
                "generation": {},
            },
            "watermark": VOW,
        },
        summary={
            "task": "gsm8k",
            "sample_num": 2,
            "correct_num": 1,
        },
    )
    humaneval = artifact(
        tmp_path,
        "humaneval",
        kind="downstream",
        schema="humaneval-evaluation-v2",
        semantic={
            "task": "humaneval",
            "model": MODEL,
            "prompt_population": {
                "dataset": {
                    "name": "humaneval",
                    "split": "test",
                    "snapshot": "humaneval",
                    "selection": {
                        "count": 2,
                        "sample_ids": ["he:1", "he:2"],
                    },
                },
                "prompt_policy": {
                    "name": "humaneval-policy",
                    "revision": "humaneval-policy-v1",
                    "parameters": {"num_shots": 0},
                },
                "generation": {},
            },
            "watermark": VOW,
        },
        summary={
            "task": "humaneval",
            "sample_num": 2,
            "passed_num": 2,
        },
    )

    rows, _ = assemble_stage(
        tmp_path,
        (gsm, humaneval),
        (gsm, humaneval),
        recipe="downstream-performance",
    )

    assert sorted(
        (row["metric"], row["value"]) for row in rows
    ) == [
        ("accuracy", 0.5),
        ("pass_at_1", 1.0),
    ]


def test_robustness_report_joins_transformation_detection_and_similarity(
    tmp_path,
):
    generated = generation(tmp_path, "generated", VOW)
    unwatermarked = generation(tmp_path, "unwatermarked", NONE)
    transformed = artifact(
        tmp_path,
        "transformed",
        kind="robustness",
        schema="robustness-text-v2",
        semantic={
            "model": MODEL,
            "watermark": VOW,
            "transformation": {"method": "word-deletion", "rate": 0.1},
        },
        sources=(generated,),
        summary={"sample_num": 2},
    )
    robust_detection = detection(
        tmp_path, "robust_detection", transformed, token_num=None
    )
    negative_detection = detection(
        tmp_path,
        "negative_detection",
        unwatermarked,
        watermark=NONE,
        detector_watermark=VOW,
        positive=0,
        token_num=None,
    )
    similarity = artifact(
        tmp_path,
        "similarity",
        kind="text-evaluation",
        schema="text-evaluation-v1",
        semantic={
            "metric_set": ["similarity"],
            "embedding_model": EVAL_MODEL,
        },
        sources=(transformed,),
        summary={
            "sample_num": 2,
            "similarity": {
                "count": 2,
                "mean": 0.8,
                "median": 0.8,
                "std": 0.0,
                "min": 0.8,
                "max": 0.8,
            },
        },
    )
    all_artifacts = (
        generated,
        unwatermarked,
        transformed,
        robust_detection,
        negative_detection,
        similarity,
    )

    rows, _ = assemble_stage(
        tmp_path,
        all_artifacts,
        (robust_detection, negative_detection, similarity),
        recipe="robustness",
    )

    assert sorted(row["metric"] for row in rows) == [
        "cosine_similarity",
        "false_positive_rate",
        "roc_auc",
        "true_positive_rate",
    ]
    assert sum(
        row["dimensions"]["transformation"] is None for row in rows
    ) == 1
    assert any(
        row["metric"] == "roc_auc" and row["value"] == 0.75
        for row in rows
    )
    assert any(
        row["metric"] == "cosine_similarity" and row["value"] == 0.8
        for row in rows
    )


def test_adaptive_forgery_and_diversity_recipes(tmp_path):
    forged = artifact(
        tmp_path,
        "forged",
        kind="adaptive-forgery",
        schema="adaptive-forgery-v2",
        semantic={
            "model": MODEL,
            "prompt_population": PROMPT_POPULATION,
            "max_new_tokens": 350,
            "target_scored_pairs": 300,
            "max_candidates": 2,
            "watermark": VOW,
        },
        summary={
            "sample_num": 2,
            "theoretical_green_probability": 0.75,
            "theoretical_queries_per_scored_token": 1.5,
        },
    )
    detected = detection(tmp_path, "forge_detection", forged)
    detected_summary = json.loads(
        (detected.path / "summary.json").read_text(encoding="utf-8")
    )
    detected_summary.update(
        {
            "p_value_distribution": {
                "count": 2,
                "mean": 1.5e-8,
                "median": 1.5e-8,
                "min": 1e-8,
                "max": 2e-8,
            },
            "negative_log10_p_value_distribution": {
                "count": 2,
                "mean": 7.85,
                "median": 7.85,
                "min": 7.7,
                "max": 8.0,
            },
            "green_token_count_distribution": {
                "count": 2,
                "mean": 47.5,
                "median": 47.5,
                "min": 47,
                "max": 48,
            },
            "effective_token_count_distribution": {
                "count": 2,
                "mean": 50.0,
                "median": 50.0,
                "min": 50,
                "max": 50,
            },
            "oracle_query_count": 180,
            "oracle_query_count_distribution": {
                "count": 2,
                "mean": 90.0,
                "median": 90.0,
                "min": 88,
                "max": 92,
            },
            "green_ratio": 0.95,
            "green_count_thresholds": {"1e-05": 188},
            "query_overhead_vs_honest_audit": 1.8,
        }
    )
    (detected.path / "summary.json").write_text(
        json.dumps(detected_summary),
        encoding="utf-8",
    )
    ppl = perplexity(tmp_path, "forge_ppl", detected, VOW, 6.5)
    forged_k3 = artifact(
        tmp_path,
        "forged_k3",
        kind="adaptive-forgery",
        schema="adaptive-forgery-v3",
        semantic={
            "model": MODEL,
            "prompt_population": PROMPT_POPULATION,
            "max_new_tokens": 350,
            "target_scored_pairs": 300,
            "max_candidates": 3,
            "watermark": VOW,
        },
        summary={
            "sample_num": 2,
            "theoretical_green_probability": 0.875,
            "theoretical_queries_per_scored_token": 1.75,
        },
    )
    detected_k3 = detection(
        tmp_path,
        "forge_detection_k3",
        forged_k3,
        positive=2,
        token_num=None,
    )
    ppl_k3 = perplexity(
        tmp_path,
        "forge_ppl_k3",
        detected_k3,
        VOW,
        6.8,
    )
    unwatermarked = generation(tmp_path, "forge_baseline", NONE)
    baseline_ppl = perplexity(
        tmp_path,
        "forge_baseline_ppl",
        unwatermarked,
        NONE,
        5.1,
    )
    honest_watermarked = generation(
        tmp_path,
        "honest_watermarked",
        VOW,
    )
    honest_watermarked_ppl = perplexity(
        tmp_path,
        "honest_watermarked_ppl",
        honest_watermarked,
        VOW,
        5.7,
    )

    forgery_rows, _ = assemble_stage(
        tmp_path,
        (
            forged,
            detected,
            ppl,
            forged_k3,
            detected_k3,
            ppl_k3,
            unwatermarked,
            baseline_ppl,
            honest_watermarked,
            honest_watermarked_ppl,
        ),
        (
            detected,
            detected_k3,
            ppl,
            ppl_k3,
            baseline_ppl,
            honest_watermarked_ppl,
        ),
        recipe="adaptive-forgery",
    )

    assert {
        "attack_success_rate",
        "p_value",
        "negative_log10_p_value",
        "green_token_count",
        "effective_token_count",
        "oracle_query_count",
        "total_oracle_query_count",
        "green_ratio",
        "green_count_threshold",
        "query_overhead_vs_honest_audit",
        "theoretical_green_probability",
        "theoretical_queries_per_scored_token",
        "conditional_perplexity",
    } == {row["metric"] for row in forgery_rows}
    assert sum(
        row["metric"] == "conditional_perplexity"
        for row in forgery_rows
    ) == 4
    honest_rows = [
        row
        for row in forgery_rows
        if row["metric"] == "conditional_perplexity"
        and row["dimensions"]["scheme"] == "vow"
        and row["dimensions"]["generation"]["max_candidates"] is None
    ]
    assert len(honest_rows) == 1
    assert honest_rows[0]["value"] == 5.7
    assert sum(
        row["metric"] == "attack_success_rate"
        for row in forgery_rows
    ) == 2

    generated = generation(tmp_path, "diverse_generation", VOW)
    diversity = artifact(
        tmp_path,
        "diversity",
        kind="text-evaluation",
        schema="text-evaluation-v1",
        semantic={
            "metric_set": ["diversity"],
            "embedding_model": EVAL_MODEL,
        },
        sources=(generated,),
        summary={
            "sample_num": 2,
            "diversity": {
                "group_num": 2,
                "aggregate": {
                    "distinct_1": 0.9,
                    "distinct_2": 0.8,
                    "distinct_3": 0.7,
                    "self_bleu_4": 0.2,
                    "vendi_score": 1.8,
                },
            },
        },
    )

    diversity_rows, _ = assemble_stage(
        tmp_path,
        (generated, diversity),
        (diversity,),
        recipe="diversity",
    )

    assert len(diversity_rows) == 5
    assert {row["metric"] for row in diversity_rows} == {
        "distinct_1",
        "distinct_2",
        "distinct_3",
        "self_bleu_4",
        "vendi_score",
    }
