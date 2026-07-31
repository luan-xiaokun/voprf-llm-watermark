from __future__ import annotations

import json

import pytest

from watermark_suite.experiments.dataset_prompt import (
    DATASET_PROMPTS,
    ResolvedPromptPopulation,
)
from watermark_suite.experiments.errors import (
    PlanValidationError,
    ResolutionError,
)


class ChatTokenizer:
    chat_template = "test-template"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return "CHAT:" + "|".join(
            f"{message['role']}={message['content']}"
            for message in messages
        )


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("kind", "field", "value"),
    (
        ("c4", "prompt_text", "C4 prompt"),
        ("eli5", "question", "Why is the sky blue?"),
    ),
)
def test_text_prompt_populations_resolve_and_render(
    tmp_path,
    kind,
    field,
    value,
):
    path = tmp_path / f"{kind}.jsonl"
    write_jsonl(path, [{field: value, "original_index": 7}])

    resolved = DATASET_PROMPTS.resolve(
        {
            "kind": kind,
            "format": "jsonl",
            "path": str(path),
            "prompt_field": field,
        },
        catalog={},
        repository=tmp_path,
        sample_num=1,
    )
    materialized = DATASET_PROMPTS.materialize(
        resolved,
        tokenizer=ChatTokenizer() if kind == "eli5" else None,
    )

    sample = materialized.samples[0]
    assert sample.source_prompt == value
    assert sample.sample_id.startswith(f"{kind}:")
    if kind == "eli5":
        assert sample.model_prompt.startswith("CHAT:system=")
        assert "user=Why is the sky blue?" in sample.model_prompt
    else:
        assert sample.model_prompt == value


def test_repetition_identity_is_owned_by_prompt_population(tmp_path):
    path = tmp_path / "c4.jsonl"
    write_jsonl(
        path,
        [{"original_index": 12, "prompt_text": "A prompt"}],
    )
    resolved = DATASET_PROMPTS.resolve(
        {
            "kind": "c4",
            "format": "jsonl",
            "path": str(path),
        },
        catalog={},
        repository=tmp_path,
        sample_num=1,
    )

    repeated = DATASET_PROMPTS.materialize(resolved).repeated(2)

    assert [sample.sample_id for sample in repeated.samples] == [
        "c4:12:repeat:0",
        "c4:12:repeat:1",
    ]
    assert all(
        sample.source_sample_id == "c4:12"
        for sample in repeated.samples
    )


def test_materialization_rejects_dataset_mutation(tmp_path):
    path = tmp_path / "c4.jsonl"
    write_jsonl(path, [{"original_index": 1, "prompt_text": "before"}])
    resolved = DATASET_PROMPTS.resolve(
        {
            "kind": "c4",
            "format": "jsonl",
            "path": str(path),
        },
        catalog={},
        repository=tmp_path,
        sample_num=1,
    )
    write_jsonl(path, [{"original_index": 1, "prompt_text": "after"}])

    with pytest.raises(
        ResolutionError,
        match="changed after Plan resolution",
    ):
        DATASET_PROMPTS.materialize(resolved)


def test_resolved_population_rejects_prompt_policy_tampering(tmp_path):
    path = tmp_path / "c4.jsonl"
    write_jsonl(path, [{"original_index": 1, "prompt_text": "prompt"}])
    resolved = DATASET_PROMPTS.resolve(
        {
            "kind": "c4",
            "format": "jsonl",
            "path": str(path),
        },
        catalog={},
        repository=tmp_path,
        sample_num=1,
    ).to_dict()
    resolved["prompt_policy"]["parameters"]["source_field"] = "changed"

    with pytest.raises(
        PlanValidationError,
        match="Prompt Policy identity",
    ):
        ResolvedPromptPopulation.from_dict(resolved)


def test_gsm8k_adapter_owns_official_population_and_prompt(
    tmp_path,
    monkeypatch,
):
    import datasets

    records = [
        {"question": f"question {index}", "answer": "#### 1"}
        for index in range(1319)
    ]
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: records,
    )

    resolved = DATASET_PROMPTS.resolve(
        {"kind": "gsm8k"},
        catalog={},
        repository=tmp_path,
    )
    materialized = DATASET_PROMPTS.materialize(resolved)

    assert resolved.sample_num == 1319
    assert resolved.prompt_policy["parameters"]["num_shots"] == 4
    assert resolved.generation["stop_strings"]
    assert materialized.samples[0].sample_id == "gsm8k:0"
    assert "question 0" in materialized.samples[0].model_prompt


def test_humaneval_adapter_owns_official_population_and_prompt(
    tmp_path,
    monkeypatch,
):
    import human_eval.data

    problems = {
        f"HumanEval/{index}": {
            "task_id": f"HumanEval/{index}",
            "prompt": f"def function_{index}():\n    pass",
        }
        for index in range(164)
    }
    monkeypatch.setattr(
        human_eval.data,
        "read_problems",
        lambda: problems,
    )

    resolved = DATASET_PROMPTS.resolve(
        {"kind": "humaneval"},
        catalog={},
        repository=tmp_path,
    )
    materialized = DATASET_PROMPTS.materialize(resolved)

    assert resolved.sample_num == 164
    assert resolved.prompt_policy["parameters"]["num_shots"] == 0
    assert materialized.samples[0].sample_id == "humaneval:HumanEval/0"
    assert "def function_0" in materialized.samples[0].model_prompt
