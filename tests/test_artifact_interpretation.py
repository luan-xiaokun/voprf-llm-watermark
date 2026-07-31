from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from watermark_suite.experiments.artifact_interpretation import (
    ArtifactInterpretationError,
    MetricScope,
    interpret_artifacts,
)
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
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
DATASET = {
    "kind": "c4",
    "split": "train",
    "snapshot": "dataset-snapshot",
    "selection": {
        "count": 2,
        "sample_ids": ["c4:1", "c4:2"],
    },
}
VOW = {
    "method": "vow",
    "enabled": True,
    "window_size": 4,
    "gamma": 0.5,
    "delta": 2.5,
    "naive_baseline": False,
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


def make_artifact(
    root: Path,
    name: str,
    *,
    kind: str,
    schema: str,
    semantic: dict,
    summary: dict,
    sources: tuple[ArtifactRef, ...] = (),
    records: list[dict] | None = None,
    manifest_record_count: int | None = None,
) -> ArtifactRef:
    identity = f"artifact_{name}"
    path = root / "artifacts" / identity
    path.mkdir(parents=True)
    if records is None:
        records = [
            {"sample_id": f"{name}:{index}"}
            for index in range(int(summary.get("sample_num", 0)))
        ]
    (path / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (path / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = {
        "kind": kind,
        "artifact_schema_revision": schema,
        "semantic_settings": semantic,
        "source_artifacts": [source.identity.value for source in sources],
        "record_count": (
            len(records)
            if manifest_record_count is None
            else manifest_record_count
        ),
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


def register(root: Path, artifacts: tuple[ArtifactRef, ...]) -> None:
    workspace = ExperimentWorkspace(root)
    with workspace.connect() as connection:
        for ref in artifacts:
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
                    json.dumps(ref.manifest, sort_keys=True),
                ),
            )


def generation(root: Path, name: str = "generation") -> ArtifactRef:
    return make_artifact(
        root,
        name,
        kind="generation",
        schema="generated-text-v3",
        semantic={
            "model": MODEL,
            "prompt_population": PROMPT_POPULATION,
            "generation": GENERATION,
            "watermark": VOW,
        },
        summary={
            "sample_num": 2,
            "generated_token_num": 200,
            "mean_generated_token_num": 100.0,
        },
    )


def detection(
    root: Path,
    source: ArtifactRef,
    name: str = "detection",
    *,
    watermark: dict = VOW,
    records: list[dict] | None = None,
    rate: float = 0.5,
) -> ArtifactRef:
    return make_artifact(
        root,
        name,
        kind="detection",
        schema="watermark-detection-v3",
        semantic={"watermark": watermark},
        sources=(source,),
        records=records,
        summary={
            "sample_num": 2,
            "detection_rate": {"1e-05": rate},
            "detection_counts": {
                "1e-05": {"positive_num": 1, "sample_num": 2},
            },
            "p_value_distribution": {
                "count": 2,
                "mean": 0.15,
                "median": 0.15,
                "min": 0.1,
                "max": 0.2,
                "p05": 0.1,
                "p25": 0.1,
                "p75": 0.2,
                "p95": 0.2,
            },
            "milestone_detection_rate": [
                {
                    "token_num": 100,
                    "eligible_sample_num": 2,
                    "detection_rate": {"1e-05": rate},
                    "detection_counts": {
                        "1e-05": {
                            "positive_num": 1,
                            "sample_num": 2,
                        }
                    },
                }
            ],
        },
    )


def test_normalizes_lineage_facts_relations_and_streams_samples(tmp_path):
    generated = generation(tmp_path)
    records = [
        {
            "sample_id": "c4:1",
            "detection": {
                "p_value": 0.1,
                "green_token_num": 50,
                "effective_token_num": 100,
            },
        },
        {
            "sample_id": "c4:2",
            "detection": {
                "p_value": 0.2,
                "green_token_num": 45,
                "effective_token_num": 100,
            },
        },
    ]
    detected = detection(tmp_path, generated, records=records)
    register(tmp_path, (generated, detected))

    result = interpret_artifacts(tmp_path, (detected,))
    interpreted = result.artifact(detected.identity)

    assert interpreted.dimensions.scheme == "vow"
    assert interpreted.dimensions.generation_model == MODEL
    assert interpreted.dimensions.population_identity is not None
    assert interpreted.direct_source_identities == (generated.identity,)
    assert {
        (fact.metric, fact.dimensions.token_num)
        for fact in interpreted.aggregate_facts
        if fact.metric == "detection_rate"
    } == {("detection_rate", None), ("detection_rate", 100)}

    stream = result.stream_sample_facts(
        detected.identity,
        metrics={"p_value"},
    )
    assert iter(stream) is stream
    facts = list(stream)
    assert [fact.value for fact in facts] == [0.1, 0.2]
    assert all(fact.scope == MetricScope.SAMPLE for fact in facts)
    assert all(fact.metric == "p_value" for fact in facts)


def test_supports_exact_current_kind_schema_pairs(tmp_path):
    generated = generation(tmp_path)
    forged = make_artifact(
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
    transformed = make_artifact(
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
        summary={"sample_num": 2, "mean_character_ratio": 0.9},
    )
    detected = detection(tmp_path, generated)
    ppl = make_artifact(
        tmp_path,
        "ppl",
        kind="perplexity",
        schema="conditional-perplexity-v2",
        semantic={"model": EVAL_MODEL, "watermark": VOW},
        sources=(generated,),
        summary={
            "sample_num": 2,
            "conditional_perplexity": 6.0,
            "sample_conditional_perplexity": {
                "mean": 6.0,
                "median": 6.0,
            },
        },
    )
    text_evaluation = make_artifact(
        tmp_path,
        "text_evaluation",
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
                "min": 0.7,
                "max": 0.9,
            },
        },
    )
    downstream = []
    for task, schema, positive_field in (
        ("gsm8k", "gsm8k-evaluation-v2", "correct_num"),
        ("humaneval", "humaneval-evaluation-v2", "passed_num"),
    ):
        task_dataset = {
            "name": task,
            "split": "test",
            "snapshot": f"{task}-snapshot",
            "selection": {
                "count": 2,
                "sample_ids": [f"{task}:1", f"{task}:2"],
            },
        }
        downstream.append(
            make_artifact(
                tmp_path,
                task,
                kind="downstream",
                schema=schema,
                semantic={
                    "task": task,
                    "model": MODEL,
                    "prompt_population": {
                        "revision": "dataset-prompt-v1",
                        "kind": task,
                        "dataset": task_dataset,
                        "source": {"name": task},
                        "prompt_policy": {
                            "name": f"{task}-policy",
                            "revision": f"{task}-policy-v1",
                            "parameters": {
                                "num_shots": (
                                    4 if task == "gsm8k" else 0
                                )
                            },
                            "identity": f"prompt-policy-{task}",
                        },
                        "generation": {},
                    },
                    "decoding": {"do_sample": False, "num_beams": 1},
                    "task_evaluation": {},
                    "watermark": VOW,
                },
                summary={
                    "task": task,
                    "sample_num": 2,
                    positive_field: 1,
                },
            )
        )
    artifacts = (
        generated,
        forged,
        transformed,
        detected,
        ppl,
        text_evaluation,
        *downstream,
    )
    register(tmp_path, artifacts)

    result = interpret_artifacts(tmp_path, artifacts)

    assert {
        (artifact.kind, artifact.schema_revision)
        for artifact in result.root_artifacts
    } == {
        ("generation", "generated-text-v3"),
        ("adaptive-forgery", "adaptive-forgery-v2"),
        ("robustness", "robustness-text-v2"),
        ("detection", "watermark-detection-v3"),
        ("perplexity", "conditional-perplexity-v2"),
        ("text-evaluation", "text-evaluation-v1"),
        ("downstream", "gsm8k-evaluation-v2"),
        ("downstream", "humaneval-evaluation-v2"),
    }
    ppl_fact = result.artifact(ppl.identity).facts(
        "conditional_perplexity"
    )[0]
    assert ppl_fact.distribution is not None
    assert ppl_fact.distribution.count == 2


def test_preflight_aggregates_unknown_and_kind_schema_errors(tmp_path):
    unknown = make_artifact(
        tmp_path,
        "unknown",
        kind="generation",
        schema="generated-text-v99",
        semantic={},
        summary={"sample_num": 0},
    )
    wrong_kind = make_artifact(
        tmp_path,
        "wrong_kind",
        kind="robustness",
        schema="generated-text-v3",
        semantic={},
        summary={"sample_num": 0},
    )
    detected = detection(tmp_path, unknown)
    register(tmp_path, (unknown, wrong_kind, detected))

    with pytest.raises(ArtifactInterpretationError) as captured:
        interpret_artifacts(tmp_path, (detected, wrong_kind))

    codes = [issue.code for issue in captured.value.issues]
    assert "unknown-schema" in codes
    assert "kind-schema-mismatch" in codes
    assert captured.value.issues == tuple(
        sorted(captured.value.issues, key=lambda issue: issue.sort_key())
    )


def test_repeated_lineage_invariant_cannot_be_overridden(tmp_path):
    generated = generation(tmp_path)
    conflicting = {
        **VOW,
        "gamma": 0.625,
    }
    detected = detection(
        tmp_path,
        generated,
        watermark=conflicting,
    )
    register(tmp_path, (generated, detected))

    with pytest.raises(
        ArtifactInterpretationError,
        match="Lineage Invariant 'scheme' conflicts",
    ) as captured:
        interpret_artifacts(tmp_path, (detected,))

    assert {
        issue.code for issue in captured.value.issues
    } == {"lineage-invariant-conflict"}


def test_preflight_validates_count_rate_and_record_consistency(tmp_path):
    generated = generation(tmp_path)
    detected = detection(
        tmp_path,
        generated,
        rate=0.75,
    )
    mismatched = make_artifact(
        tmp_path,
        "mismatched",
        kind="generation",
        schema="generated-text-v3",
        semantic={
            "model": MODEL,
            "prompt_population": PROMPT_POPULATION,
            "generation": GENERATION,
            "watermark": VOW,
        },
        summary={"sample_num": 2},
        records=[{"sample_id": "only-one"}],
        manifest_record_count=1,
    )
    register(tmp_path, (generated, detected, mismatched))

    with pytest.raises(ArtifactInterpretationError) as captured:
        interpret_artifacts(tmp_path, (detected, mismatched))

    assert {
        issue.code for issue in captured.value.issues
    } >= {"schema-interpretation", "population-count-mismatch"}
    assert "disagrees with exact count ratio" in str(captured.value)


def test_sample_validation_is_deferred_and_rejects_duplicate_identity(
    tmp_path,
):
    generated = generation(tmp_path)
    detected = detection(
        tmp_path,
        generated,
        records=[
            {"sample_id": "duplicate", "detection": {"p_value": 0.1}},
            {"sample_id": "duplicate", "detection": {"p_value": 0.2}},
        ],
    )
    register(tmp_path, (generated, detected))
    result = interpret_artifacts(tmp_path, (detected,))

    stream = result.stream_sample_facts(
        detected.identity,
        metrics={"p_value"},
    )
    assert next(stream).sample_identity == "duplicate"
    with pytest.raises(ValueError, match="duplicate Sample identity"):
        next(stream)


def test_supplied_root_relation_must_match_verified_manifest(tmp_path):
    generated = generation(tmp_path)
    detected = detection(tmp_path, generated)
    register(tmp_path, (generated, detected))
    stale = replace(detected, source_artifacts=())

    with pytest.raises(
        ArtifactInterpretationError,
        match="ArtifactRef and manifest sources disagree",
    ) as captured:
        interpret_artifacts(tmp_path, (stale,))

    assert {
        issue.code for issue in captured.value.issues
    } == {"source-relation-mismatch"}


def test_missing_ancestor_is_reported_without_short_circuiting(tmp_path):
    generated = generation(tmp_path)
    detected = detection(tmp_path, generated)
    register(tmp_path, (detected,))

    with pytest.raises(ArtifactInterpretationError) as captured:
        interpret_artifacts(tmp_path, (detected,))

    codes = {issue.code for issue in captured.value.issues}
    assert "missing-artifact" in codes
    assert "missing-scientific-role" in codes


def test_lineage_cycle_is_aggregated_as_a_preflight_issue(tmp_path):
    generated = generation(tmp_path)
    transformed = make_artifact(
        tmp_path,
        "transformed_cycle",
        kind="robustness",
        schema="robustness-text-v2",
        semantic={
            "model": MODEL,
            "watermark": VOW,
            "transformation": {"method": "word-deletion", "rate": 0.1},
        },
        sources=(generated,),
        summary={"sample_num": 2, "mean_character_ratio": 0.9},
    )
    cyclic_manifest = {
        **generated.manifest,
        "source_artifacts": [transformed.identity.value],
    }
    (generated.path / "manifest.json").write_text(
        json.dumps(cyclic_manifest),
        encoding="utf-8",
    )
    cyclic_generation = replace(
        generated,
        source_artifacts=(transformed.identity,),
        manifest=cyclic_manifest,
    )
    register(tmp_path, (cyclic_generation, transformed))

    with pytest.raises(ArtifactInterpretationError) as captured:
        interpret_artifacts(tmp_path, (cyclic_generation,))

    assert "lineage-cycle" in {
        issue.code for issue in captured.value.issues
    }
