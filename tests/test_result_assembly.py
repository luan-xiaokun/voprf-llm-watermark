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
    positive: int = 1,
    token_num: int = 100,
) -> ArtifactRef:
    return artifact(
        root,
        name,
        kind="detection",
        schema="watermark-detection-v3",
        semantic={"watermark": watermark},
        sources=(source,),
        summary={
            "sample_num": 2,
            "watermark": watermark,
            "detection_counts": {
                "1e-05": {"positive_num": positive, "sample_num": 2},
                "1e-02": {"positive_num": positive, "sample_num": 2},
            },
            "milestone_detection_rate": [
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
    clean = detection(tmp_path, "clean", generated, token_num=50)
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
        tmp_path, "robust_detection", transformed, token_num=50
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
        clean,
        transformed,
        robust_detection,
        similarity,
    )

    rows, _ = assemble_stage(
        tmp_path,
        all_artifacts,
        (clean, robust_detection, similarity),
        recipe="robustness",
    )

    assert sorted(row["metric"] for row in rows) == [
        "cosine_similarity",
        "detection_rate",
        "detection_rate",
    ]
    assert sum(
        row["dimensions"]["transformation"] is None for row in rows
    ) == 1
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
            "watermark": VOW,
        },
        summary={"sample_num": 2},
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
            "query_overhead_vs_honest_audit": 1.8,
        }
    )
    detected_summary["adaptive_forgery_curve"] = [
        {
            "token_num": 50,
            "eligible_sample_num": 2,
            "mean_oracle_query_count": 90.0,
            "median_oracle_query_count": 88.0,
            "mean_queries_per_token": 1.8,
            "mean_selected_green_token_count": 47.5,
            "mean_selected_green_ratio": 0.95,
            "median_p_value": 1e-8,
            "attack_success_counts": {
                "1e-05": {"positive_num": 2, "sample_num": 2}
            },
        }
    ]
    (detected.path / "summary.json").write_text(
        json.dumps(detected_summary),
        encoding="utf-8",
    )
    ppl = perplexity(tmp_path, "forge_ppl", detected, VOW, 6.5)

    forgery_rows, _ = assemble_stage(
        tmp_path,
        (forged, detected, ppl),
        (detected, ppl),
        recipe="adaptive-forgery",
    )

    assert {
        "mean_oracle_query_count",
        "median_oracle_query_count",
        "mean_queries_per_token",
        "mean_selected_green_token_count",
        "mean_selected_green_ratio",
        "median_p_value",
        "attack_success_rate",
        "p_value",
        "negative_log10_p_value",
        "green_token_count",
        "effective_token_count",
        "oracle_query_count",
        "total_oracle_query_count",
        "green_ratio",
        "query_overhead_vs_honest_audit",
        "conditional_perplexity",
    } == {row["metric"] for row in forgery_rows}

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
