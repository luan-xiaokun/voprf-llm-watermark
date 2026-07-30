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
) -> tuple[str, str, str]:
    del repository
    selected_revision = revision or "test-model-revision"
    location = f"/models/{checkpoint.replace('/', '--')}"
    return checkpoint, selected_revision, location


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
