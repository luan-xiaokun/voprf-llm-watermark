from __future__ import annotations

from pathlib import Path

from watermark_suite.experiments.dataset_prompt import (
    DATASET_PROMPTS,
    ResolvedPromptPopulation,
)
from watermark_suite.experiments.identity import identity_for
from watermark_suite.experiments.plan import resolve_plan
from watermark_suite.experiments.stages import default_stage_adapters


REPOSITORY = Path(__file__).resolve().parents[1]
PLAN_DIRECTORY = REPOSITORY / "experiments" / "plans"
USENIX_PLAN_DIRECTORY = REPOSITORY / "experiments" / "usenix-plans"


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


def _resolve_prompt_population(
    value,
    *,
    catalog,
    repository,
    sample_num=None,
) -> ResolvedPromptPopulation:
    del repository
    raw = catalog.get(value, value) if isinstance(value, str) else value
    kind = raw.get("kind") if isinstance(raw, dict) else raw
    count = sample_num or (1319 if kind == "gsm8k" else 164)
    parameters = {
        "num_shots": 4 if kind == "gsm8k" else 0
    } if kind in {"gsm8k", "humaneval"} else {}
    policy_document = {
        "name": f"{kind}-test",
        "revision": f"{kind}-test-v1",
        "parameters": parameters,
    }
    return ResolvedPromptPopulation(
        kind=kind,
        dataset={
            "kind": kind,
            "split": "test" if kind in {"gsm8k", "humaneval"} else "train",
            "snapshot": f"{kind}-snapshot",
            "selection": {
                "mode": "full",
                "count": count,
                "sample_ids": [
                    f"{kind}:{index}" for index in range(count)
                ],
            },
        },
        source={"kind": "test"},
        prompt_policy={
            **policy_document,
            "identity": identity_for(
                policy_document,
                prefix="prompt-policy",
            ),
        },
        generation=(
            {"stop_strings": ["Question:"]}
            if kind == "gsm8k"
            else {}
        ),
    )


def resolve_path(path: Path, monkeypatch):
    import watermark_suite.experiments.stages.common as common

    monkeypatch.setattr(common, "_resolve_checkpoint", _resolve_checkpoint)
    monkeypatch.setattr(
        DATASET_PROMPTS, "resolve", _resolve_prompt_population
    )
    return resolve_plan(
        path,
        repository=REPOSITORY,
        adapters=default_stage_adapters(),
    )


def resolve(name: str, monkeypatch):
    return resolve_path(PLAN_DIRECTORY / name, monkeypatch)


def test_usenix_vow_h2_parameter_grid_plan_contract(monkeypatch):
    plan = resolve_path(
        USENIX_PLAN_DIRECTORY
        / "vow-h2-parameter-grid-qwen25-3b-1000.yaml",
        monkeypatch,
    )

    expected_grid = {
        (gamma, delta)
        for gamma in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
        for delta in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    }
    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    watermarked = [
        stage
        for stage in generations
        if stage.semantic_settings["watermark"]["method"] == "vow"
    ]
    baselines = [
        stage
        for stage in generations
        if stage.semantic_settings["watermark"]["method"] == "none"
    ]

    assert len(plan.stages) == 150
    assert len(generations) == 50
    assert len(watermarked) == 49
    assert len(baselines) == 1
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-3B"
        for stage in generations
    )
    assert all(
        stage.semantic_settings["batch_size"] == 64
        for stage in generations
    )
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"][
            "selection"
        ]["count"]
        == 1000
        for stage in generations
    )
    assert all(
        stage.semantic_settings["watermark"]["window_size"] == 2
        for stage in watermarked
    )

    assert {
        (
            stage.semantic_settings["watermark"]["gamma"],
            stage.semantic_settings["watermark"]["delta"],
        )
        for stage in watermarked
    } == expected_grid
    assert all(
        stage.semantic_settings["generation"]
        == {
            "max_new_tokens": 210,
            "do_sample": True,
            "top_p": None,
            "top_k": None,
            "temperature": 0.7,
            "suppress_eos": True,
            "stop_strings": None,
        }
        for stage in watermarked
    )

    detections = [
        stage for stage in plan.stages if stage.kind == "detection"
    ]
    assert len(detections) == 49
    assert all(
        stage.semantic_settings["significance_levels"] == [0.00001]
        and stage.semantic_settings["step_size"] is None
        for stage in detections
    )

    perplexities = [
        stage for stage in plan.stages if stage.kind == "perplexity"
    ]
    assert len(perplexities) == 50
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-7B"
        for stage in perplexities
    )
    assert all(
        stage.semantic_settings["batch_size"] == 16
        for stage in perplexities
    )

    reports = [
        stage
        for stage in plan.stages
        if stage.kind == "result-aggregation"
    ]
    assert len(reports) == 1
    assert all(len(stage.input_instances) == 99 for stage in reports)
    assert all(
        stage.semantic_settings == {
            "recipe": "tpr-vs-ppl",
            "confidence_level": 0.95,
            "threshold_policy": {"default": 0.00001},
            "significance_levels": [],
        }
        for stage in reports
    )


def test_usenix_vow_h2_remaining_experiments_plan_contract(monkeypatch):
    plan = resolve_path(
        USENIX_PLAN_DIRECTORY
        / "vow-h2-fpr-token-top50-qwen25-3b-1000.yaml",
        monkeypatch,
    )
    multinomial_grid = resolve_path(
        USENIX_PLAN_DIRECTORY
        / "vow-h2-parameter-grid-qwen25-3b-1000.yaml",
        monkeypatch,
    )
    existing_fpr = resolve_path(
        USENIX_PLAN_DIRECTORY / "fpr-calibration-qwen25-3b-c4.yaml",
        monkeypatch,
    )

    assert len(plan.stages) == 156
    null_corpus = next(
        stage
        for stage in plan.stages
        if stage.kind == "token-window-corpus"
    )
    assert null_corpus.semantic_settings["sample_num"] == 1_000_000
    assert null_corpus.semantic_settings["window"] == {
        "token_num": 200,
        "max_per_document": 1,
        "stride": 200,
        "drop_incomplete": True,
    }
    existing_null_corpus = next(
        stage
        for stage in existing_fpr.stages
        if stage.kind == "token-window-corpus"
    )
    assert (
        null_corpus.semantic_settings
        == existing_null_corpus.semantic_settings
    )

    null_detection = next(
        stage for stage in plan.stages if stage.kind == "null-detection"
    )
    assert null_detection.semantic_settings["sample_num"] == 1_000_000
    assert null_detection.semantic_settings["significance_levels"] == [
        0.01,
        0.001,
        0.0001,
        0.00001,
    ]
    assert null_detection.semantic_settings["detector_watermark"][
        "window_size"
    ] == 2

    token_generation = next(
        stage
        for stage in plan.stages
        if stage.stage_name == "generate_vow_h2_multinomial_default"
    )
    assert token_generation.semantic_settings["batch_size"] == 64
    assert token_generation.semantic_settings["generation"] == {
        "max_new_tokens": 210,
        "do_sample": True,
        "top_p": None,
        "top_k": None,
        "temperature": 0.7,
        "suppress_eos": True,
        "stop_strings": None,
    }
    assert token_generation.semantic_settings["watermark"] == {
        "method": "vow",
        "enabled": True,
        "window_size": 2,
        "gamma": 0.5,
        "delta": 2.5,
        "server_seed_path": token_generation.semantic_settings[
            "watermark"
        ]["server_seed_path"],
        "server_seed_sha256": token_generation.semantic_settings[
            "watermark"
        ]["server_seed_sha256"],
        "naive_baseline": False,
    }
    existing_default_generation = next(
        stage
        for stage in multinomial_grid.stages
        if stage.stage_name == "generate_vow_multinomial"
        and stage.semantic_settings["watermark"]["gamma"] == 0.5
        and stage.semantic_settings["watermark"]["delta"] == 2.5
    )
    assert (
        token_generation.semantic_settings
        == existing_default_generation.semantic_settings
    )
    token_detection = next(
        stage
        for stage in plan.stages
        if stage.stage_name == "detect_vow_h2_token_length"
    )
    assert token_detection.semantic_settings["step_size"] == 5
    assert token_detection.semantic_settings["significance_levels"] == [
        0.00001
    ]

    top50_generations = [
        stage
        for stage in plan.stages
        if stage.stage_name == "generate_vow_h2_top50_grid"
    ]
    assert len(top50_generations) == 49
    assert {
        (
            stage.semantic_settings["watermark"]["gamma"],
            stage.semantic_settings["watermark"]["delta"],
        )
        for stage in top50_generations
    } == {
        (gamma, delta)
        for gamma in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875)
        for delta in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    }
    assert all(
        stage.semantic_settings["batch_size"] == 32
        and stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-3B"
        and stage.semantic_settings["prompt_population"]["dataset"]
        ["selection"]["count"]
        == 1000
        and stage.semantic_settings["watermark"]["window_size"] == 2
        and stage.semantic_settings["generation"]
        == {
            "max_new_tokens": 210,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 50,
            "temperature": 0.7,
            "suppress_eos": True,
            "stop_strings": None,
        }
        for stage in top50_generations
    )

    top50_baseline = next(
        stage
        for stage in plan.stages
        if stage.stage_name == "generate_unwatermarked_top50"
    )
    assert top50_baseline.semantic_settings["batch_size"] == 32
    assert top50_baseline.semantic_settings["watermark"]["method"] == "none"
    top50_detections = [
        stage
        for stage in plan.stages
        if stage.stage_name == "detect_vow_h2_top50_grid"
    ]
    assert len(top50_detections) == 49
    assert all(
        stage.semantic_settings["significance_levels"] == [0.00001]
        and stage.semantic_settings["step_size"] is None
        for stage in top50_detections
    )
    top50_ppl = [
        stage
        for stage in plan.stages
        if stage.stage_name
        in {
            "evaluate_vow_h2_top50_ppl",
            "evaluate_unwatermarked_top50_ppl",
        }
    ]
    assert len(top50_ppl) == 50
    assert all(
        stage.semantic_settings["batch_size"] == 16
        and stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-7B"
        for stage in top50_ppl
    )

    reports = {
        stage.stage_name: stage
        for stage in plan.stages
        if stage.kind == "result-aggregation"
    }
    assert set(reports) == {
        "assemble_vow_h2_fpr_calibration",
        "assemble_vow_h2_tpr_vs_token_length",
        "assemble_vow_h2_top50_tpr_vs_ppl",
    }
    assert len(
        reports["assemble_vow_h2_fpr_calibration"].input_instances
    ) == 1
    assert len(
        reports["assemble_vow_h2_tpr_vs_token_length"].input_instances
    ) == 1
    assert len(
        reports["assemble_vow_h2_top50_tpr_vs_ppl"].input_instances
    ) == 99


def test_usenix_vow_h2_downstream_plan_contract(monkeypatch):
    plan = resolve_path(
        USENIX_PLAN_DIRECTORY
        / "vow-h2-downstream-qwen25-3b-instruct.yaml",
        monkeypatch,
    )

    assert len(plan.stages) == 5
    downstream = [
        stage for stage in plan.stages if stage.kind == "downstream"
    ]
    assert len(downstream) == 4
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-3B-Instruct"
        for stage in downstream
    )
    assert all(
        stage.semantic_settings["batch_size"] == 32
        and stage.semantic_settings["max_new_tokens"] == 1024
        and stage.semantic_settings["decoding"]
        == {"do_sample": False, "num_beams": 1}
        for stage in downstream
    )
    assert all(
        stage.semantic_settings["watermark"]["method"] == "vow"
        and stage.semantic_settings["watermark"]["window_size"] == 2
        and stage.semantic_settings["watermark"]["gamma"] == 0.5
        for stage in downstream
    )
    assert {
        stage.semantic_settings["watermark"]["delta"]
        for stage in downstream
    } == {2.0, 2.5}

    gsm8k = [
        stage
        for stage in downstream
        if stage.semantic_settings["task"] == "gsm8k"
    ]
    humaneval = [
        stage
        for stage in downstream
        if stage.semantic_settings["task"] == "humaneval"
    ]
    assert len(gsm8k) == len(humaneval) == 2
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"]
        ["selection"]["count"]
        == 1319
        and stage.semantic_settings["prompt_population"]["prompt_policy"]
        ["parameters"]["num_shots"]
        == 4
        for stage in gsm8k
    )
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"]
        ["selection"]["count"]
        == 164
        and stage.semantic_settings["prompt_population"]["prompt_policy"]
        ["parameters"]["num_shots"]
        == 0
        for stage in humaneval
    )

    report = next(
        stage
        for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 4
    assert report.semantic_settings == {
        "recipe": "downstream-performance",
        "confidence_level": 0.95,
        "threshold_policy": {"default": 0.00001},
        "significance_levels": [],
    }


def test_usenix_robustness_plan_matches_the_500_plus_500_design(monkeypatch):
    plan = resolve_path(
        USENIX_PLAN_DIRECTORY / "robustness-qwen25-3b-instruct.yaml",
        monkeypatch,
    )

    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    assert len(generations) == 9
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"][
            "selection"
        ]["count"]
        == 500
        for stage in generations
    )
    assert sum(
        stage.semantic_settings["watermark"]["method"] == "none"
        for stage in generations
    ) == 1
    assert sum(stage.kind == "robustness" for stage in plan.stages) == 24
    assert sum(stage.kind == "detection" for stage in plan.stages) == 32
    assert sum(stage.kind == "text-evaluation" for stage in plan.stages) == 24
    assert len(plan.stages) == 90

    negative_detections = [
        stage
        for stage in plan.stages
        if stage.stage_name
        in {
            "detect_unwatermarked",
            "detect_rdf_unwatermarked",
            "detect_pdw_unwatermarked",
        }
    ]
    assert len(negative_detections) == 8
    assert {
        stage.semantic_settings["detector_watermark"]["method"]
        for stage in negative_detections
    } == {
        "vow",
        "lefthash",
        "selfhash",
        "rdf",
        "pdw",
        "upv",
    }
    detections = [stage for stage in plan.stages if stage.kind == "detection"]
    assert {
        stage.stage_name: stage.semantic_settings["step_size"]
        for stage in detections
        if stage.stage_name
        in {
            "detect_robustness",
            "detect_rdf_robustness",
            "detect_pdw_robustness",
            "detect_unwatermarked",
            "detect_rdf_unwatermarked",
            "detect_pdw_unwatermarked",
        }
    } == {
        "detect_robustness": 20,
        "detect_rdf_robustness": 50,
        "detect_pdw_robustness": None,
        "detect_unwatermarked": 20,
        "detect_rdf_unwatermarked": 50,
        "detect_pdw_unwatermarked": None,
    }
    assert all(
        stage.semantic_settings["significance_levels"] == [0.01]
        for stage in detections
    )

    evaluations = [
        stage for stage in plan.stages if stage.kind == "text-evaluation"
    ]
    assert all(
        stage.semantic_settings["embedding_provider"] == "openai"
        and stage.semantic_settings["embedding_model"]["checkpoint"]
        == "text-embedding-3-large"
        for stage in evaluations
    )
    pdw_attacks = [
        stage
        for stage in plan.stages
        if stage.stage_name == "transform_pdw_robustness"
    ]
    assert len(pdw_attacks) == 3
    assert {
        stage.semantic_settings["transformation"].get("model")
        for stage in pdw_attacks
        if stage.semantic_settings["transformation"]["method"]
        == "openai-paraphrase"
    } == {"gpt-3.5-turbo-0125", "gpt-5.6-luna"}
    luna_attacks = [
        stage
        for stage in plan.stages
        if stage.kind == "robustness"
        and stage.semantic_settings["transformation"].get("model")
        == "gpt-5.6-luna"
    ]
    assert len(luna_attacks) == 8
    assert all(
        stage.semantic_settings["transformation"]["max_output_tokens"]
        == 2048
        for stage in luna_attacks
    )
    assert all(
        stage.semantic_settings["transformation"]["temperature"] == 0.7
        and stage.semantic_settings["transformation"]["reasoning_effort"]
        == "none"
        for stage in luna_attacks
    )
    paraphrase_instruction = (
        "As an expert copy-editor, please rewrite the following text in your "
        "own voice while ensuring that the final output contains the same "
        "information as the original text and has roughly the same length. "
        "Please paraphrase all sentences and do not omit any crucial details. "
        "Additionally, please take care to provide any relevant information "
        "about public figures, organizations, or other entities mentioned in "
        "the text to avoid any potential misunderstandings or biases."
    )
    paraphrase_attacks = [
        stage
        for stage in plan.stages
        if stage.kind == "robustness"
        and stage.semantic_settings["transformation"]["method"]
        == "openai-paraphrase"
    ]
    assert len(paraphrase_attacks) == 16
    assert all(
        stage.semantic_settings["transformation"]["max_output_tokens"]
        == 2048
        for stage in paraphrase_attacks
    )
    assert {
        stage.semantic_settings["transformation"]["instruction"]
        for stage in paraphrase_attacks
    } == {paraphrase_instruction}

    report = next(
        stage
        for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 56
    assert report.semantic_settings["threshold_policy"] == {"default": 0.01}


def test_usenix_forgery_plan_matches_the_table_design(monkeypatch):
    plan = resolve_path(
        USENIX_PLAN_DIRECTORY / "adaptive-forgery-qwen25-3b-1000.yaml",
        monkeypatch,
    )

    forgeries = [
        stage for stage in plan.stages if stage.kind == "adaptive-forgery"
    ]
    assert len(forgeries) == 7
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-3B-Instruct"
        and stage.semantic_settings["max_new_tokens"] == 350
        and stage.semantic_settings["target_scored_pairs"] == 300
        and stage.semantic_settings["prompt_population"]["dataset"][
            "selection"
        ]["count"]
        == 1000
        for stage in forgeries
    )
    vow = [
        stage
        for stage in forgeries
        if stage.semantic_settings["watermark"]["method"] == "vow"
    ]
    assert sorted(stage.semantic_settings["max_candidates"] for stage in vow) == [
        1,
        2,
        3,
        4,
        5,
    ]
    baselines = [stage for stage in forgeries if stage not in vow]
    assert {
        stage.semantic_settings["watermark"]["method"]
        for stage in baselines
    } == {"lefthash", "selfhash"}
    assert all(stage.semantic_settings["max_candidates"] == 2 for stage in baselines)

    detections = [stage for stage in plan.stages if stage.kind == "detection"]
    assert len(detections) == 7
    assert all(
        stage.semantic_settings["significance_levels"] == [0.00001]
        and stage.semantic_settings["step_size"] is None
        for stage in detections
    )
    perplexities = [
        stage for stage in plan.stages if stage.kind == "perplexity"
    ]
    assert len(perplexities) == 9
    controls = [stage for stage in plan.stages if stage.kind == "generation"]
    assert len(controls) == 2
    assert {
        stage.semantic_settings["watermark"]["method"]
        for stage in controls
    } == {"none", "vow"}
    expected_control_generation = {
        "max_new_tokens": 300,
        "do_sample": True,
        "top_p": None,
        "top_k": 50,
        "temperature": 0.7,
        "suppress_eos": True,
        "stop_strings": None,
    }
    assert all(
        stage.semantic_settings["generation"]
        == expected_control_generation
        for stage in controls
    )
    honest_vow = next(
        stage
        for stage in controls
        if stage.semantic_settings["watermark"]["method"] == "vow"
    )
    assert honest_vow.semantic_settings["watermark"]["window_size"] == 4
    assert honest_vow.semantic_settings["watermark"]["gamma"] == 0.5
    assert honest_vow.semantic_settings["watermark"]["delta"] == 2.5

    report = next(
        stage
        for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 16
    assert report.semantic_settings["threshold_policy"] == {
        "default": 0.00001
    }


def test_tpr_token_length_plan_contract(monkeypatch):
    plan = resolve("figure-tpr-vs-token-length.yaml", monkeypatch)

    assert len(plan.stages) == 11
    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    assert len(generations) == 5
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"][
            "selection"
        ]["count"] == 1000
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
    report = next(
        stage for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 5
    assert report.semantic_settings["recipe"] == "tpr-vs-token-length"


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

    assert forge.semantic_settings["prompt_population"]["dataset"][
        "selection"
    ]["count"] == 1000
    assert forge.semantic_settings["watermark"]["gamma"] == 0.5
    assert forge.semantic_settings["max_candidates"] == 8
    assert forge.semantic_settings["trace_level"] == "compact"
    assert detection.semantic_settings["step_size"] == 10
    assert 0.00001 in detection.semantic_settings["significance_levels"]
    report = next(
        stage for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert report.semantic_settings["recipe"] == "adaptive-forgery"
    assert len(report.input_instances) == 2


def test_robustness_plan_is_an_immutable_stage_matrix(monkeypatch):
    plan = resolve(
        "robustness-qwen25-7b-instruct.yaml",
        monkeypatch,
    )

    generations = [
        stage for stage in plan.stages if stage.kind == "generation"
    ]
    assert all(
        stage.semantic_settings["prompt_population"]["dataset"][
            "selection"
        ]["count"] == 1000
        for stage in generations
    )
    assert len(plan.stages) == 78
    assert sum(stage.kind == "robustness" for stage in plan.stages) == 21
    assert sum(stage.kind == "detection" for stage in plan.stages) == 28
    assert sum(
        stage.kind == "text-evaluation" for stage in plan.stages
    ) == 21
    assert {
        stage.semantic_settings["transformation"]["method"]
        for stage in plan.stages
        if stage.kind == "robustness"
    } == {"masked-lm-replacement", "openai-paraphrase"}
    replacements = [
        stage.semantic_settings["transformation"]
        for stage in plan.stages
        if (
            stage.kind == "robustness"
            and stage.semantic_settings["transformation"]["method"]
            == "masked-lm-replacement"
        )
    ]
    assert all(value["replacement_rate"] == 0.3 for value in replacements)
    assert all(value["top_k"] == 15 for value in replacements)
    assert all(
        value["candidate_sampling"] == "score-weighted"
        for value in replacements
    )
    assert {
        value["implementation_revision"] for value in replacements
    } == {"masked-lm-roundtrip-window-v2"}
    assert {
        value["model"]["checkpoint"] for value in replacements
    } == {"distilbert/distilbert-base-uncased"}
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
    assert {
        value["model"]: value["temperature"] for value in paraphrases
    } == {
        "gpt-3.5-turbo-0125": 0.7,
        "gpt-5.6-sol": None,
    }
    assert all(
        "implementation_revision" not in value for value in paraphrases
    )
    clean_detection = [
        stage
        for stage in plan.stages
        if (
            stage.kind == "detection"
            and stage.settings["target_field"] == "generated_text"
        )
    ]
    assert len(clean_detection) == 7
    report = next(
        stage for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 49


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
    reports = [
        stage for stage in plan.stages
        if stage.kind == "result-aggregation"
    ]
    assert len(reports) == 1
    assert len(reports[0].input_instances) == 7
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

    assert len(ppl_plan.stages) == 30
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
    report = next(
        stage for stage in ppl_plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 19


def test_downstream_plan_contract(monkeypatch):
    plan = resolve("table-downstream-performance.yaml", monkeypatch)

    assert len(plan.stages) == 17
    gsm8k = [
        stage for stage in plan.stages
        if stage.stage_name == "evaluate_gsm8k"
    ]
    humaneval = [
        stage for stage in plan.stages
        if stage.stage_name == "evaluate_humaneval"
    ]
    assert len(gsm8k) == len(humaneval) == 8
    downstream = [
        stage for stage in plan.stages if stage.kind == "downstream"
    ]
    assert all(
        stage.semantic_settings["batch_size"] == 8 for stage in downstream
    )
    assert all(
        stage.semantic_settings["max_new_tokens"] == 1024
        for stage in downstream
    )
    assert all(
        stage.semantic_settings["model"]["checkpoint"]
        == "Qwen/Qwen2.5-7B-Instruct"
        for stage in downstream
    )
    assert all(
        stage.semantic_settings["prompt_population"]["prompt_policy"][
            "parameters"
        ]["num_shots"] == 4
        for stage in gsm8k
    )
    assert all(
        stage.semantic_settings["prompt_population"]["prompt_policy"][
            "parameters"
        ]["num_shots"] == 0
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
    pdw = next(
        stage
        for stage in gsm8k
        if stage.semantic_settings["watermark"]["method"] == "pdw"
    )
    assert pdw.semantic_settings["decoding"] == {
        "do_sample": False,
        "num_beams": 1,
        "effective_sampling": "multinomial",
    }
    assert (
        pdw.semantic_settings["watermark"]["max_generation_attempts"] == 3
    )
    assert all(
        stage.semantic_settings["decoding"]
        == {"do_sample": False, "num_beams": 1}
        for stage in gsm8k
        if stage.semantic_settings["watermark"]["method"] != "pdw"
    )
    report = next(
        stage for stage in plan.stages
        if stage.kind == "result-aggregation"
    )
    assert len(report.input_instances) == 16
