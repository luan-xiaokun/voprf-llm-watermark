from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from watermark_suite.experiments.adapters import (
    ResolutionContext,
    StageExecutionContext,
)
from watermark_suite.experiments.errors import PlanValidationError
from watermark_suite.experiments.models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    RunIdentity,
)
from watermark_suite.experiments.openai_paraphrase import (
    OpenAIParaphraseResult,
    OpenAIParaphraser,
)
from watermark_suite.experiments.stages.robustness import (
    RobustnessStageAdapter,
)
from watermark_suite.experiments.stages.detection import DetectionStageAdapter
from watermark_suite.experiments.stages.text_evaluation import (
    TextEvaluationStageAdapter,
)


MODEL = {
    "checkpoint": "encoder",
    "revision": "commit",
    "location": "/models/encoder",
    "verification": {"kind": "huggingface-cache", "commit": "commit"},
    "tokenizer_checkpoint": "encoder",
    "tokenizer_revision": "commit",
    "tokenizer_location": "/models/encoder",
    "tokenizer_verification": {
        "kind": "huggingface-cache",
        "commit": "commit",
    },
}
WATERMARK = {
    "method": "vow",
    "enabled": True,
    "window_size": 4,
    "delta": 2.5,
    "gamma": 0.5,
    "server_seed_path": "/data/server_seed",
    "server_seed_sha256": "digest",
    "naive_baseline": False,
}


def resolution_context(tmp_path: Path) -> ResolutionContext:
    return ResolutionContext(
        repository=tmp_path,
        models={},
        datasets={},
        defaults={},
        code_revision="commit",
    )


def source_artifact(
    tmp_path: Path,
    records: list[dict],
    *,
    watermark: dict = WATERMARK,
) -> ArtifactRef:
    path = tmp_path / "artifact"
    path.mkdir()
    records_path = path / "records.jsonl"
    records_path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return ArtifactRef(
        identity=ArtifactIdentity("artifact_source"),
        path=path,
        schema_revision="generated-text-v3",
        run_identity=RunIdentity("run_source"),
        attempt_identity=AttemptIdentity("attempt_source"),
        manifest={
            "semantic_settings": {
                "model": MODEL,
                "watermark": watermark,
            }
        },
    )


def test_detection_can_score_one_unwatermarked_artifact_with_an_override(
    tmp_path,
):
    source = source_artifact(
        tmp_path,
        [{"sample_id": "sample:1", "generated_text": "plain text"}],
        watermark={"method": "none", "enabled": False},
    )
    detector = {
        "method": "lefthash",
        "enabled": True,
        "gamma": 0.25,
        "delta": 2.0,
    }
    adapter = DetectionStageAdapter()
    definition = adapter.resolve(
        {
            "batch_size": 8,
            "target_field": "generated_text",
            "token_num": None,
            "significance_levels": [0.01],
            "use_local": True,
            "step_size": None,
            "detector_watermark": detector,
            "device": "cpu",
        },
        resolution_context(tmp_path),
    )

    bound = adapter.bind_inputs(definition, (source,))

    assert bound.semantic_settings["watermark"] == {
        "method": "none",
        "enabled": False,
    }
    assert bound.semantic_settings["detector_watermark"] == detector


def execution_context(
    tmp_path: Path,
    *,
    definition,
    source: ArtifactRef,
    runtime,
) -> StageExecutionContext:
    return StageExecutionContext(
        repository=tmp_path,
        workspace=tmp_path / "workspace",
        stage_name="stage",
        run_identity="run",
        attempt_identity="attempt",
        settings=definition.settings,
        semantic_settings=definition.semantic_settings,
        execution_settings=definition.execution_settings,
        inputs=(source,),
        runtime=runtime,
    )


def test_robustness_stage_preserves_source_and_transformation_lineage(
    tmp_path,
):
    source = source_artifact(
        tmp_path,
        [
            {
                "sample_id": "sample:1",
                "prompt_text": "prompt",
                "generated_text": "one two three four five six",
            }
        ],
    )
    original = (source.path / "records.jsonl").read_bytes()
    adapter = RobustnessStageAdapter()
    definition = adapter.resolve(
        {
            "transformation": {
                "method": "word-deletion",
                "rate": 0.5,
            },
            "batch_size": 1,
            "target_field": "generated_text",
            "seed": 42,
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )
    bound = adapter.bind_inputs(definition, (source,))
    execution = adapter.prepare(
        execution_context(
            tmp_path,
            definition=bound,
            source=source,
            runtime=object(),
        )
    )

    result = execution.execute(execution.work_items()[0])
    record = result.records[0]

    assert (source.path / "records.jsonl").read_bytes() == original
    assert record["sample_id"] == "sample:1"
    assert record["original_text"] == "one two three four five six"
    assert record["transformed_text"] != record["original_text"]
    assert record["robustness"]["method"] == "word-deletion"
    assert bound.semantic_settings["model"] == MODEL
    assert bound.semantic_settings["watermark"] == WATERMARK


def test_robustness_stage_records_masked_lm_replacement_metrics(
    tmp_path,
    monkeypatch,
):
    import watermark_suite.experiments.stages.robustness as module

    class FakeReplacement:
        text = "Cats pursue mice."

        @staticmethod
        def provenance():
            return {
                "eligible_word_count": 3,
                "selected_word_count": 1,
                "replaced_word_count": 1,
                "failed_replacement_count": 0,
                "target_replacement_rate": 0.3,
                "realized_replacement_rate": 1 / 3,
                "mean_selected_candidate_rank": 2.0,
            }

    class FakeReplacer:
        def __init__(self, tokenizer, unmasker):
            assert tokenizer == "tokenizer"
            assert unmasker == "unmasker"

        def replace_many(self, texts, **settings):
            assert texts == ["Cats chase mice."]
            assert len(settings["seeds"]) == 1
            assert settings["replacement_rate"] == 0.3
            assert settings["top_k"] == 15
            assert settings["candidate_sampling"] == "score-weighted"
            return [FakeReplacement()]

    class FakeMaskedLMRuntime:
        def fill_mask(self, model, *, device, dtype):
            assert model == MODEL
            assert device == "cpu"
            assert dtype == "float32"
            return "unmasker", "tokenizer"

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    monkeypatch.setattr(module, "SynonymReplacer", FakeReplacer)
    source = source_artifact(
        tmp_path,
        [
            {
                "sample_id": "sample:1",
                "generated_text": "Cats chase mice.",
            }
        ],
    )
    adapter = RobustnessStageAdapter()
    definition = adapter.resolve(
        {
            "transformation": {
                "method": "masked-lm-replacement",
                "model": "synonym-replacer",
                "replacement_rate": 0.3,
                "top_k": 15,
                "candidate_sampling": "score-weighted",
            },
            "batch_size": 4,
            "target_field": "generated_text",
            "seed": 42,
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )
    bound = adapter.bind_inputs(definition, (source,))
    execution = adapter.prepare(
        execution_context(
            tmp_path,
            definition=bound,
            source=source,
            runtime=FakeMaskedLMRuntime(),
        )
    )

    result = execution.execute(execution.work_items()[0])
    summary = execution.summarize(list(result.records))

    assert result.records[0]["transformed_text"] == "Cats pursue mice."
    assert result.records[0]["robustness"]["replaced_word_count"] == 1
    assert summary["masked_lm_replacement"] == {
        "eligible_word_count": 3,
        "selected_word_count": 1,
        "replaced_word_count": 1,
        "failed_replacement_count": 0,
        "realized_replacement_rate": 1 / 3,
        "mean_selected_candidate_rank": 2.0,
    }


class FakeResponses:
    def __init__(self):
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        return SimpleNamespace(
            id=f"response:{len(self.requests)}",
            model=request["model"],
            status="completed",
            created_at=123,
            service_tier="default",
            output_text="rewritten text",
            usage=SimpleNamespace(
                input_tokens=12,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=3,
                    cache_write_tokens=2,
                ),
                output_tokens=7,
                output_tokens_details=SimpleNamespace(
                    reasoning_tokens=4,
                ),
                total_tokens=19,
            ),
        )


class FakeOpenAIError(RuntimeError):
    def __init__(self, status_code):
        super().__init__(f"OpenAI status {status_code}")
        self.status_code = status_code


def test_openai_paraphraser_retries_transient_server_errors():
    class EventuallySuccessfulResponses(FakeResponses):
        def create(self, **request):
            if len(self.requests) < 3:
                self.requests.append(request)
                raise FakeOpenAIError(500)
            return super().create(**request)

    responses = EventuallySuccessfulResponses()
    paraphraser = OpenAIParaphraser(
        SimpleNamespace(responses=responses),
        max_attempts=4,
        initial_retry_delay_seconds=0,
    )

    result = paraphraser.paraphrase(
        "original",
        model="gpt-5.6-luna",
        instruction="rewrite",
        max_output_tokens=600,
        temperature=None,
        reasoning_effort="low",
        request_identity="sample:1",
    )

    assert result.text == "rewritten text"
    assert result.provenance["application_attempts"] == 4
    assert len(responses.requests) == 4


def test_openai_paraphraser_retries_incomplete_token_limited_responses():
    class EventuallyCompleteResponses(FakeResponses):
        def create(self, **request):
            if len(self.requests) < 2:
                self.requests.append(request)
                return SimpleNamespace(
                    id=f"response:{len(self.requests)}",
                    model=request["model"],
                    status="incomplete",
                    incomplete_details=SimpleNamespace(
                        reason="max_output_tokens"
                    ),
                    output_text="",
                    usage=None,
                )
            return super().create(**request)

    responses = EventuallyCompleteResponses()
    paraphraser = OpenAIParaphraser(
        SimpleNamespace(responses=responses),
        max_attempts=3,
        initial_retry_delay_seconds=0,
    )

    result = paraphraser.paraphrase(
        "original",
        model="gpt-5.6-luna",
        instruction="rewrite",
        max_output_tokens=600,
        temperature=None,
        reasoning_effort="low",
        request_identity="sample:1",
    )

    assert result.text == "rewritten text"
    assert result.provenance["application_attempts"] == 3
    assert len(responses.requests) == 3


def test_openai_paraphraser_reports_non_retryable_incomplete_reason():
    class IncompleteResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **request):
            self.calls += 1
            return SimpleNamespace(
                id="response:1",
                model=request["model"],
                status="incomplete",
                incomplete_details=SimpleNamespace(reason="content_filter"),
                output_text="",
                usage=None,
            )

    responses = IncompleteResponses()
    paraphraser = OpenAIParaphraser(
        SimpleNamespace(responses=responses),
        max_attempts=3,
        initial_retry_delay_seconds=0,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "model='gpt-5.6-luna'.*sample_id='sample:2'.*"
            "reason='content_filter'"
        ),
    ):
        paraphraser.paraphrase(
            "original",
            model="gpt-5.6-luna",
            instruction="rewrite",
            max_output_tokens=600,
            temperature=None,
            reasoning_effort="low",
            request_identity="sample:2",
        )

    assert responses.calls == 1


def test_openai_paraphraser_does_not_retry_bad_requests():
    class InvalidResponses:
        def __init__(self):
            self.calls = 0

        def create(self, **request):
            del request
            self.calls += 1
            raise FakeOpenAIError(400)

    responses = InvalidResponses()
    paraphraser = OpenAIParaphraser(
        SimpleNamespace(responses=responses),
        max_attempts=4,
        initial_retry_delay_seconds=0,
    )

    with pytest.raises(
        RuntimeError,
        match="model='gpt-3.5-turbo'.*sample_id='sample:2'",
    ):
        paraphraser.paraphrase(
            "original",
            model="gpt-3.5-turbo",
            instruction="rewrite",
            max_output_tokens=600,
            temperature=0.7,
            reasoning_effort=None,
            request_identity="sample:2",
        )

    assert responses.calls == 1


def test_openai_paraphraser_only_sends_reasoning_for_reasoning_models():
    responses = FakeResponses()
    paraphraser = OpenAIParaphraser(
        SimpleNamespace(responses=responses)
    )

    gpt35 = paraphraser.paraphrase(
        "original",
        model="gpt-3.5-turbo",
        instruction="rewrite",
        max_output_tokens=600,
        temperature=0.7,
        reasoning_effort=None,
    )
    sol = paraphraser.paraphrase(
        "original",
        model="gpt-5.6-sol",
        instruction="rewrite",
        max_output_tokens=600,
        temperature=None,
        reasoning_effort="low",
    )

    assert "reasoning" not in responses.requests[0]
    assert responses.requests[0]["temperature"] == 0.7
    assert responses.requests[1]["reasoning"] == {"effort": "low"}
    assert "temperature" not in responses.requests[1]
    assert gpt35.provenance["usage"]["cached_input_tokens"] == 3
    assert sol.provenance["usage"]["reasoning_output_tokens"] == 4


@pytest.mark.parametrize(
    ("model", "temperature", "reasoning_effort", "message"),
    [
        (
            "gpt-3.5-turbo-0125",
            0.7,
            "low",
            "does not support reasoning_effort",
        ),
        (
            "gpt-5.6-luna",
            0.7,
            "low",
            "does not support temperature",
        ),
    ],
)
def test_robustness_stage_rejects_model_incompatible_openai_parameters(
    tmp_path,
    model,
    temperature,
    reasoning_effort,
    message,
):
    adapter = RobustnessStageAdapter()

    with pytest.raises(PlanValidationError, match=message):
        adapter.resolve(
            {
                "transformation": {
                    "method": "openai-paraphrase",
                    "model": model,
                    "max_output_tokens": 600,
                    "temperature": temperature,
                    "reasoning_effort": reasoning_effort,
                    "instruction": "rewrite",
                },
                "batch_size": 8,
                "target_field": "generated_text",
                "seed": 42,
                "device": "cpu",
                "dtype": "float32",
            },
            resolution_context(tmp_path),
        )


def test_gpt56_temperature_is_allowed_when_reasoning_is_disabled(tmp_path):
    definition = RobustnessStageAdapter().resolve(
        {
            "transformation": {
                "method": "openai-paraphrase",
                "model": "gpt-5.6-luna",
                "max_output_tokens": 600,
                "temperature": 0.7,
                "reasoning_effort": "none",
                "instruction": "rewrite",
            },
            "batch_size": 8,
            "target_field": "generated_text",
            "seed": 42,
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )

    transformation = definition.semantic_settings["transformation"]
    assert transformation["temperature"] == 0.7
    assert transformation["reasoning_effort"] == "none"


def test_robustness_stage_records_openai_response_provenance(
    tmp_path,
    monkeypatch,
):
    import watermark_suite.experiments.stages.robustness as module

    class FakeParaphraser:
        def __init__(self, *, max_attempts):
            assert max_attempts == 4

        def paraphrase_many(self, texts, **settings):
            assert texts == ["source text"]
            assert settings["request_identities"] == ["sample:1"]
            assert settings["model"] == "gpt-5.6-sol"
            assert settings["reasoning_effort"] == "low"
            assert settings["concurrency"] == 4
            return [
                OpenAIParaphraseResult(
                    text="rewritten text",
                    provenance={
                        "provider": "openai",
                        "endpoint": "responses",
                        "requested_model": "gpt-5.6-sol",
                        "response_model": "gpt-5.6-sol",
                        "response_id": "response:1",
                        "response_status": "completed",
                        "response_created_at": 123,
                        "service_tier": "default",
                        "reasoning_effort": "low",
                        "latency_seconds": 0.25,
                        "usage": {
                            "input_tokens": 10,
                            "cached_input_tokens": 0,
                            "cache_write_input_tokens": 0,
                            "output_tokens": 5,
                            "reasoning_output_tokens": 2,
                            "total_tokens": 15,
                        },
                    },
                )
            ]

    monkeypatch.setattr(module, "OpenAIParaphraser", FakeParaphraser)
    source = source_artifact(
        tmp_path,
        [
            {
                "sample_id": "sample:1",
                "generated_text": "source text",
            }
        ],
    )
    adapter = RobustnessStageAdapter()
    definition = adapter.resolve(
        {
            "transformation": {
                "method": "openai-paraphrase",
                "model": "gpt-5.6-sol",
                "max_output_tokens": 600,
                "temperature": None,
                "reasoning_effort": "low",
                "instruction": "rewrite",
            },
            "batch_size": 8,
            "openai_concurrency": 4,
            "openai_max_attempts": 4,
            "target_field": "generated_text",
            "seed": 42,
            "device": "cpu",
            "dtype": "float32",
        },
        resolution_context(tmp_path),
    )
    bound = adapter.bind_inputs(definition, (source,))
    assert "openai_concurrency" not in bound.semantic_settings
    assert bound.execution_settings["openai_concurrency"] == 4
    assert bound.execution_settings["openai_max_attempts"] == 4
    execution = adapter.prepare(
        execution_context(
            tmp_path,
            definition=bound,
            source=source,
            runtime=object(),
        )
    )

    result = execution.execute(execution.work_items()[0])
    summary = execution.summarize(list(result.records))

    assert result.records[0]["transformed_text"] == "rewritten text"
    assert result.records[0]["robustness"]["response_id"] == "response:1"
    assert summary["openai"]["request_num"] == 1
    assert summary["openai"]["usage"]["total_tokens"] == 15


class FakeEncoder:
    def encode(self, texts, **kwargs):
        del kwargs
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float(lowered.count("cat")),
                    float(lowered.count("dog")),
                    float(len(lowered.split())),
                ]
            )
        return np.asarray(vectors, dtype=float)


class FakeRuntime:
    def encoder(self, model, *, device):
        assert model == MODEL
        assert device == "cpu"
        return FakeEncoder()


def test_text_evaluation_emits_sample_similarity_and_diversity_summary(
    tmp_path, monkeypatch
):
    import watermark_suite.experiments.stages.text_evaluation as module

    monkeypatch.setattr(module, "resolve_model", lambda *args, **kwargs: MODEL)
    source = source_artifact(
        tmp_path,
        [
            {
                "sample_id": "sample:1",
                "group": "prompt:1",
                "original_text": "cat cat",
                "transformed_text": "cat dog",
            },
            {
                "sample_id": "sample:2",
                "group": "prompt:1",
                "original_text": "dog dog",
                "transformed_text": "dog cat",
            },
        ],
    )
    adapter = TextEvaluationStageAdapter()
    definition = adapter.resolve(
        {
            "metric_set": ["similarity", "diversity"],
            "embedding_model": "encoder",
            "batch_size": 2,
            "reference_field": "original_text",
            "target_field": "transformed_text",
            "group_field": "group",
            "device": "cpu",
        },
        resolution_context(tmp_path),
    )
    bound = adapter.bind_inputs(definition, (source,))
    execution = adapter.prepare(
        execution_context(
            tmp_path,
            definition=bound,
            source=source,
            runtime=FakeRuntime(),
        )
    )

    result = execution.execute(execution.work_items()[0])
    summary = execution.summarize(list(result.records))

    assert all(
        "cosine_similarity" in record["text_evaluation"]
        for record in result.records
    )
    assert summary["metric_set"] == ["similarity", "diversity"]
    assert summary["similarity"]["count"] == 2
    assert summary["diversity"]["group_num"] == 1
    assert summary["diversity"]["groups"][0]["group"] == "prompt:1"


def test_text_evaluation_supports_openai_embedding_models(
    tmp_path,
    monkeypatch,
):
    import watermark_suite.experiments.stages.text_evaluation as module

    class FakeOpenAIEmbedder:
        def embed_many(self, texts, *, model, dimensions):
            assert model == "text-embedding-3-large"
            assert dimensions is None
            return [
                [
                    float(text.lower().count("cat")),
                    float(text.lower().count("dog")),
                ]
                for text in texts
            ]

    monkeypatch.setattr(module, "OpenAIEmbedder", FakeOpenAIEmbedder)
    source = source_artifact(
        tmp_path,
        [
            {
                "sample_id": "sample:1",
                "original_text": "cat cat",
                "transformed_text": "cat dog",
            }
        ],
    )
    adapter = TextEvaluationStageAdapter()
    definition = adapter.resolve(
        {
            "metric_set": ["similarity"],
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": None,
            "batch_size": 128,
            "reference_field": "original_text",
            "target_field": "transformed_text",
            "group_field": None,
            "device": "cuda",
        },
        resolution_context(tmp_path),
    )
    bound = adapter.bind_inputs(definition, (source,))
    execution = adapter.prepare(
        execution_context(
            tmp_path,
            definition=bound,
            source=source,
            runtime=object(),
        )
    )

    result = execution.execute(execution.work_items()[0])

    assert definition.semantic_settings["embedding_model"] == {
        "provider": "openai",
        "checkpoint": "text-embedding-3-large",
        "revision": "api",
    }
    assert 0 < result.records[0]["text_evaluation"]["cosine_similarity"] < 1
