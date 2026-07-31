from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from watermark_suite.experiments.adapters import (
    ResolutionContext,
    StageExecutionContext,
)
from watermark_suite.experiments.stages.adaptive_forgery import (
    AdaptiveForgeryStageAdapter,
)
from watermark_suite.experiments.stages.downstream import (
    DownstreamStageAdapter,
)
from watermark_suite.experiments.stages.generation import (
    GenerationStageAdapter,
)


MODEL = {
    "checkpoint": "test-model",
    "revision": "test-model-revision",
    "location": "/models/test-model",
    "verification": {
        "kind": "huggingface-cache",
        "commit": "test-model-revision",
    },
    "tokenizer_checkpoint": "test-model",
    "tokenizer_revision": "test-model-revision",
    "tokenizer_location": "/models/test-model",
    "tokenizer_verification": {
        "kind": "huggingface-cache",
        "commit": "test-model-revision",
    },
}


class TinyTokenizer:
    eos_token_id = 0
    bos_token_id = 0
    pad_token_id = 0
    all_special_ids = [0]

    def __call__(self, prompt: str, return_tensors: str):
        assert prompt
        assert return_tensors == "pt"
        return {
            "input_ids": torch.tensor([[5]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return ",".join(str(token_id) for token_id in token_ids)

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if "," in text:
            return [int(piece) for piece in text.split(",")]
        return list(range(len(text.split())))


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rankings = (
            (1, 2, 3),
            (2, 3, 4),
            (4, 2, 1),
            (1, 2, 3),
        )

    @property
    def device(self):
        return torch.device("cpu")

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=True,
        return_dict=True,
    ):
        del attention_mask, use_cache, return_dict
        step = 0 if past_key_values is None else past_key_values
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], 6),
            -100.0,
        )
        for score, token_id in zip(
            (3.0, 2.0, 1.0),
            self.rankings[step],
        ):
            logits[:, -1, token_id] = score
        return SimpleNamespace(
            logits=logits,
            past_key_values=step + 1,
        )


class FakeRuntime:
    def __init__(self, model=None, tokenizer=None):
        self.model = model if model is not None else object()
        self.tokenizer = tokenizer or TinyTokenizer()
        self.calls = []

    def get(self, model, **settings):
        self.calls.append((model, settings))
        return self.model, self.tokenizer


class RecordingWatermarker:
    def __init__(self, output_text: str):
        self.output_text = output_text
        self.calls = []

    def __call__(self, *, prompts, **settings):
        self.calls.append(
            {
                "prompts": list(prompts),
                "settings": settings,
            }
        )
        return [self.output_text for _ in prompts]


def write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def resolution_context(tmp_path, *, datasets=None):
    return ResolutionContext(
        repository=tmp_path,
        models={},
        datasets=datasets or {},
        defaults={},
        code_revision="commit",
    )


def execution_context(tmp_path, definition, runtime):
    return StageExecutionContext(
        repository=tmp_path,
        workspace=tmp_path / "workspace",
        stage_name="test-stage",
        run_identity="run_test",
        attempt_identity="attempt_test",
        settings=definition.settings,
        semantic_settings=definition.semantic_settings,
        execution_settings=definition.execution_settings,
        inputs=(),
        runtime=runtime,
    )


def execute_all(execution):
    records = []
    for item in execution.work_items():
        records.extend(execution.execute(item).records)
    return records


def test_generation_pipeline_materializes_repeats_and_emits_records(
    tmp_path,
    monkeypatch,
):
    import watermark_suite.experiments.stages.generation as module

    dataset_path = tmp_path / "c4.jsonl"
    write_jsonl(
        dataset_path,
        [
            {
                "original_index": 10,
                "prompt_text": "first prompt",
                "url": "https://example.test/first",
            },
            {
                "original_index": 11,
                "prompt_text": "second prompt",
                "url": "https://example.test/second",
            },
        ],
    )
    monkeypatch.setattr(
        module,
        "resolve_model",
        lambda *args, **kwargs: MODEL,
    )
    watermarker = RecordingWatermarker("generated text")
    monkeypatch.setattr(
        module.WATERMARK_SCHEMES,
        "generator",
        lambda *args, **kwargs: (watermarker, True),
    )
    adapter = GenerationStageAdapter()
    definition = adapter.resolve(
        {
            "model": "test-model",
            "dataset": "local-c4",
            "num_samples": 2,
            "repetitions": 2,
            "batch_size": 3,
            "seed": 7,
            "max_new_tokens": 12,
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 40,
            "temperature": 0.7,
            "suppress_eos": False,
            "stop_strings": ["STOP"],
            "watermark": {"method": "none", "enabled": False},
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(
            tmp_path,
            datasets={
                "local-c4": {
                    "kind": "c4",
                    "format": "jsonl",
                    "path": str(dataset_path),
                }
            },
        ),
    )
    runtime = FakeRuntime()

    execution = adapter.prepare(
        execution_context(tmp_path, definition, runtime)
    )
    items = execution.work_items()
    records = execute_all(execution)
    summary = execution.summarize(records)

    assert [item.sample_identities for item in items] == [
        (
            "c4:10:repeat:0",
            "c4:10:repeat:1",
            "c4:11:repeat:0",
        ),
        ("c4:11:repeat:1",),
    ]
    assert [record["sample_id"] for record in records] == [
        "c4:10:repeat:0",
        "c4:10:repeat:1",
        "c4:11:repeat:0",
        "c4:11:repeat:1",
    ]
    assert [record["prompt_text"] for record in records] == [
        "first prompt",
        "first prompt",
        "second prompt",
        "second prompt",
    ]
    assert records[0]["url"] == "https://example.test/first"
    assert all(record["generated_token_num"] == 2 for record in records)
    assert [call["prompts"] for call in watermarker.calls] == [
        ["first prompt", "first prompt", "second prompt"],
        ["second prompt"],
    ]
    assert all(
        call["settings"]["stop_strings"] == ["STOP"]
        and call["settings"]["no_watermark"] is True
        for call in watermarker.calls
    )
    assert runtime.calls[0][1]["padding_side"] == "left"
    assert summary["sample_num"] == 4
    assert summary["source_sample_num"] == 2
    assert summary["repetitions"] == 2


def test_adaptive_forgery_pipeline_uses_materialized_prompt_and_summarizes(
    tmp_path,
    monkeypatch,
):
    import watermark_suite.experiments.stages.adaptive_forgery as module

    dataset_path = tmp_path / "c4.jsonl"
    write_jsonl(
        dataset_path,
        [{"original_index": 20, "prompt_text": "forge this prompt"}],
    )
    seed_path = tmp_path / "server.seed"
    seed_path.write_text(bytes(range(64)).hex(), encoding="utf-8")
    monkeypatch.setattr(
        module,
        "resolve_model",
        lambda *args, **kwargs: MODEL,
    )
    adapter = AdaptiveForgeryStageAdapter()
    definition = adapter.resolve(
        {
            "model": "test-model",
            "dataset": {
                "kind": "c4",
                "format": "jsonl",
                "path": str(dataset_path),
            },
            "num_samples": 1,
            "batch_size": 1,
            "seed": 9,
            "max_new_tokens": 4,
            "max_candidates": 3,
            "watermark": {
                "method": "vow",
                "enabled": True,
                "window_size": 1,
                "delta": 2.5,
                "gamma": 0.5,
                "server_seed_path": str(seed_path),
            },
            "allow_special_tokens": False,
            "trace_level": "compact",
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )
    runtime = FakeRuntime(model=TinyCausalModel())

    execution = adapter.prepare(
        execution_context(tmp_path, definition, runtime)
    )
    items = execution.work_items()
    records = execute_all(execution)
    summary = execution.summarize(records)

    assert len(items) == 1
    assert items[0].sample_identities == ("c4:20",)
    assert records[0]["source_prompt"] == "forge this prompt"
    assert records[0]["prompt_text"] == "forge this prompt"
    assert records[0]["generated_text"]
    assert records[0]["adaptive_forgery"]["generated_token_num"] == 4
    assert records[0]["adaptive_forgery"]["oracle_query_count"] > 0
    assert records[0]["sample_metrics"]["scored_position_num"] == 3
    assert summary["sample_num"] == 1
    assert summary["generated_token_num"] == 4
    assert summary["oracle_query_count"] > 0
    assert summary["theoretical_green_probability"] == pytest.approx(
        0.875
    )


@pytest.mark.parametrize(
    ("task", "sample_num", "num_shots"),
    (
        ("gsm8k", 1319, 4),
        ("humaneval", 164, 0),
    ),
)
def test_downstream_pipeline_materializes_official_task_and_scores(
    tmp_path,
    monkeypatch,
    task,
    sample_num,
    num_shots,
):
    import datasets
    import human_eval.data
    import watermark_suite.experiments.stages.downstream as module

    gsm8k_records = [
        {
            "question": f"question {index}",
            "answer": "#### 1",
        }
        for index in range(1319)
    ]
    humaneval_records = {
        f"HumanEval/{index}": {
            "task_id": f"HumanEval/{index}",
            "prompt": f"def function_{index}():\n    pass",
        }
        for index in range(164)
    }
    monkeypatch.setattr(
        datasets,
        "load_dataset",
        lambda *args, **kwargs: gsm8k_records,
    )
    monkeypatch.setattr(
        human_eval.data,
        "read_problems",
        lambda: humaneval_records,
    )
    monkeypatch.setattr(
        module,
        "resolve_model",
        lambda *args, **kwargs: MODEL,
    )
    monkeypatch.setattr(
        module,
        "check_correctness",
        lambda *args, **kwargs: {
            "passed": True,
            "result": "passed",
        },
    )
    output = (
        "The answer is 1. #### 1"
        if task == "gsm8k"
        else "return 1"
    )
    watermarker = RecordingWatermarker(output)
    monkeypatch.setattr(
        module.WATERMARK_SCHEMES,
        "generator",
        lambda *args, **kwargs: (watermarker, True),
    )
    adapter = DownstreamStageAdapter()
    definition = adapter.resolve(
        {
            "task": task,
            "model": "test-model",
            "batch_size": sample_num,
            "seed": 11,
            "max_new_tokens": 32,
            "num_shots": num_shots,
            "watermark": {"method": "none", "enabled": False},
            "evaluation_workers": 2,
            "execution_timeout": 3.0,
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )
    runtime = FakeRuntime()

    execution = adapter.prepare(
        execution_context(tmp_path, definition, runtime)
    )
    items = execution.work_items()
    records = execute_all(execution)
    summary = execution.summarize(records)

    assert len(items) == 1
    assert len(items[0].sample_identities) == sample_num
    assert len(records) == sample_num
    assert watermarker.calls[0]["prompts"] == [
        record["prompt_text"] for record in records
    ]
    assert watermarker.calls[0]["settings"]["no_watermark"] is True
    assert summary["sample_num"] == sample_num
    assert summary["num_shots"] == num_shots
    if task == "gsm8k":
        assert watermarker.calls[0]["settings"]["stop_strings"]
        assert records[0]["sample_id"] == "gsm8k:0"
        assert records[0]["prediction"] == "1"
        assert records[0]["is_correct"] is True
        assert summary["accuracy"] == 1.0
    else:
        assert watermarker.calls[0]["settings"]["stop_strings"] is None
        assert records[0]["sample_id"] == "humaneval:HumanEval/0"
        assert records[0]["passed"] is True
        assert summary["pass@1"] == 1.0
