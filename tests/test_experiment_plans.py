from __future__ import annotations

from pathlib import Path

from watermark_suite.experiments.plan import resolve_plan
from watermark_suite.experiments.stages import default_stage_adapters


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_DIRECTORY = REPOSITORY / "experiments" / "plans"


def _resolve_checkpoint(
    checkpoint: str,
    revision: str | None,
    repository: Path,
) -> tuple[str, str, str, dict]:
    del repository
    selected_revision = revision or "test-model-revision"
    location = f"/models/{checkpoint.replace('/', '--')}"
    return (
        checkpoint,
        selected_revision,
        location,
        {
            "kind": "huggingface-cache",
            "commit": selected_revision,
        },
    )


def _resolve_task_dataset(task: str) -> dict:
    sample_num = 1319 if task == "gsm8k" else 164
    return {
        "name": task,
        "config": "main" if task == "gsm8k" else None,
        "split": "test",
        "sample_num": sample_num,
        "sample_ids": [f"{task}:{index}" for index in range(sample_num)],
        "snapshot": f"{task}-snapshot",
    }


def resolve(name: str, monkeypatch):
    import watermark_suite.experiments.stages.common as common
    import watermark_suite.experiments.stages.downstream as downstream

    monkeypatch.setattr(common, "_resolve_checkpoint", _resolve_checkpoint)
    monkeypatch.setattr(
        downstream, "resolve_task_dataset", _resolve_task_dataset
    )
    return resolve_plan(
        PLAN_DIRECTORY / name,
        repository=REPOSITORY,
        adapters=default_stage_adapters(),
    )


def test_tpr_token_length_plan_contract(monkeypatch):
    plan = resolve("figure-tpr-vs-token-length.yaml", monkeypatch)

    assert len(plan.stages) == 10
    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    assert len(generations) == 5
    assert all(
        stage.semantic_settings["dataset"]["selection"]["count"] == 1000
        for stage in generations
    )
    assert all(
        stage.semantic_settings["generation"] == {
            "max_new_tokens": 210,
            "do_sample": True,
            "top_p": None,
            "top_k": None,
            "temperature": 0.7,
            "suppress_eos": False,
            "stop_strings": None,
        }
        for stage in generations
    )
    detections = {
        stage.stage_name: stage for stage in plan.stages
        if stage.kind == "detection"
    }
    assert detections["detect_vow_default"].semantic_settings[
        "step_size"
    ] == 5
    assert detections["detect_lefthash"].semantic_settings["step_size"] == 5
    assert detections["detect_selfhash"].semantic_settings["step_size"] == 5
    assert detections["detect_upv"].semantic_settings["step_size"] == 10
    assert detections["detect_rdf"].semantic_settings["step_size"] == 20
    assert detections["detect_rdf"].semantic_settings[
        "significance_levels"
    ] == [0.01]


def test_adaptive_forgery_plan_has_1000_samples_and_length_curve(
    monkeypatch,
):
    plan = resolve("adaptive-forgery-qwen25-7b.yaml", monkeypatch)

    forge = next(
        stage for stage in plan.stages
        if stage.kind == "adaptive-forgery"
    )
    detection = next(
        stage for stage in plan.stages if stage.kind == "detection"
    )

    assert forge.semantic_settings["sample_manifest"]["count"] == 1000
    assert forge.semantic_settings["watermark"]["gamma"] == 0.5
    assert forge.semantic_settings["max_candidates"] == 8
    assert forge.semantic_settings["trace_level"] == "compact"
    assert detection.semantic_settings["step_size"] == 10
    assert 0.00001 in detection.semantic_settings["significance_levels"]


def test_robustness_plan_is_an_immutable_stage_matrix(monkeypatch):
    import watermark_suite.experiments.stages.common as common

    monkeypatch.setattr(
        common,
        "_load_dataset_records",
        lambda spec, limit=None: [
            {"question": f"question {index}"}
            for index in range(limit or 500)
        ],
    )
    plan = resolve(
        "robustness-qwen25-7b-instruct.yaml",
        monkeypatch,
    )

    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    assert all(
        stage.semantic_settings["sample_manifest"]["count"] == 1000
        for stage in generations
    )
    assert len(plan.stages) == 77
    assert sum(stage.kind == "robustness" for stage in plan.stages) == 21
    assert sum(stage.kind == "detection" for stage in plan.stages) == 28
    assert sum(
        stage.kind == "text-evaluation" for stage in plan.stages
    ) == 21
    assert {
        stage.semantic_settings["transformation"]["method"]
        for stage in plan.stages
        if stage.kind == "robustness"
    } == {"word-deletion", "openai-paraphrase"}
    paraphrases = [
        stage.semantic_settings["transformation"]
        for stage in plan.stages
        if (
            stage.kind == "robustness"
            and stage.semantic_settings["transformation"]["method"]
            == "openai-paraphrase"
        )
    ]
    assert {value["model"] for value in paraphrases} == {
        "gpt-3.5-turbo-0125",
        "gpt-5.6-sol",
    }
    assert {
        value["model"]: value["reasoning_effort"]
        for value in paraphrases
    } == {
        "gpt-3.5-turbo-0125": None,
        "gpt-5.6-sol": "low",
    }
    assert {value["temperature"] for value in paraphrases} == {0.7}
    clean_detection = [
        stage
        for stage in plan.stages
        if (
            stage.kind == "detection"
            and stage.settings["target_field"] == "generated_text"
        )
    ]
    assert len(clean_detection) == 7


def test_diversity_plan_repeats_each_prompt_and_uses_unified_evaluation(
    monkeypatch,
):
    plan = resolve("diversity-qwen25-7b.yaml", monkeypatch)

    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    evaluations = [
        stage for stage in plan.stages
        if stage.kind == "text-evaluation"
    ]

    assert len(generations) == 7
    assert len(evaluations) == 7
    assert all(
        stage.semantic_settings["repetitions"] == 50
        for stage in generations
    )
    assert all(
        stage.semantic_settings["metric_set"] == ["diversity"]
        and stage.semantic_settings["group_field"] == "source_sample_id"
        for stage in evaluations
    )


def test_tpr_ppl_plan_reuses_shared_generation_semantics(monkeypatch):
    token_plan = resolve("figure-tpr-vs-token-length.yaml", monkeypatch)
    ppl_plan = resolve("figure-tpr-vs-ppl.yaml", monkeypatch)

    assert len(ppl_plan.stages) == 29
    token_generation = {
        stage.semantic_settings["watermark"]["method"]: (
            stage.semantic_settings
        )
        for stage in token_plan.stages
        if stage.kind == "generation"
    }
    ppl_generation = [
        stage for stage in ppl_plan.stages
        if stage.stage_name == "generate_watermarked"
    ]
    assert len(ppl_generation) == 9
    for stage in ppl_generation:
        watermark = stage.semantic_settings["watermark"]
        if (
            watermark["method"] in token_generation
            and (
                watermark["method"] != "vow"
                or (
                    watermark["delta"] == 2.5
                    and watermark["gamma"] == 0.5
                )
            )
        ):
            assert stage.semantic_settings == token_generation[
                watermark["method"]
            ]

    perplexity = [
        stage for stage in ppl_plan.stages if stage.kind == "perplexity"
    ]
    assert len(perplexity) == 10
    assert all(
        stage.semantic_settings["batch_size"] == 2
        for stage in perplexity
    )
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-14B"
        for stage in perplexity
    )


def test_downstream_plan_contract(monkeypatch):
    plan = resolve("table-downstream-performance.yaml", monkeypatch)

    assert len(plan.stages) == 16
    gsm8k = [
        stage for stage in plan.stages
        if stage.stage_name == "evaluate_gsm8k"
    ]
    humaneval = [
        stage for stage in plan.stages
        if stage.stage_name == "evaluate_humaneval"
    ]
    assert len(gsm8k) == len(humaneval) == 8
    assert all(stage.semantic_settings["batch_size"] == 8 for stage in plan.stages)
    assert all(
        stage.semantic_settings["max_new_tokens"] == 1024
        for stage in plan.stages
    )
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-7B-Instruct"
        for stage in plan.stages
    )
    assert all(
        stage.semantic_settings["task_evaluation"]["num_shots"] == 4
        for stage in gsm8k
    )
    assert all(
        stage.semantic_settings["task_evaluation"]["num_shots"] == 0
        for stage in humaneval
    )
    methods = {
        stage.semantic_settings["watermark"]["method"] for stage in gsm8k
    }
    assert methods == {
        "none",
        "lefthash",
        "selfhash",
        "rdf",
        "pdw",
        "upv",
        "vow",
    }
    vow_deltas = {
        stage.semantic_settings["watermark"]["delta"]
        for stage in gsm8k
        if stage.semantic_settings["watermark"]["method"] == "vow"
    }
    assert vow_deltas == {2.0, 2.5}
