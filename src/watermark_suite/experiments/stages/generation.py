from __future__ import annotations

import statistics

import torch

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import (
    batches,
    format_prompt,
    load_selected_samples,
    require,
    resolve_dataset,
    resolve_model,
    validate_execution_settings,
)


class GenerationStageAdapter:
    kind = "generation"
    revision = "generation-v2"
    accepted_settings = {
        "model",
        "dataset",
        "num_samples",
        "repetitions",
        "batch_size",
        "seed",
        "max_new_tokens",
        "do_sample",
        "top_p",
        "top_k",
        "temperature",
        "suppress_eos",
        "stop_strings",
        "watermark",
        "device",
        "dtype",
    }
    _required = accepted_settings

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        validate_execution_settings(settings)
        for field in (
            "num_samples",
            "repetitions",
            "batch_size",
            "max_new_tokens",
        ):
            if not isinstance(settings[field], int) or settings[field] <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if not isinstance(settings["seed"], int):
            raise PlanValidationError("seed must be an integer")
        if not isinstance(settings["do_sample"], bool):
            raise PlanValidationError("do_sample must be boolean")
        if not isinstance(settings["suppress_eos"], bool):
            raise PlanValidationError("suppress_eos must be boolean")
        if (
            settings["temperature"] is not None
            and settings["temperature"] <= 0
        ):
            raise PlanValidationError("temperature must be positive or null")
        model = resolve_model(
            settings["model"],
            models=context.models,
            repository=context.repository,
        )
        dataset = resolve_dataset(
            settings["dataset"],
            datasets=context.datasets,
            repository=context.repository,
            num_samples=settings["num_samples"],
        )
        watermark = WATERMARK_SCHEMES.resolve(
            settings["watermark"], context.repository
        )
        semantic = {
            "model": model,
            "dataset": dataset,
            "sample_manifest": dataset["selection"],
            "repetitions": settings["repetitions"],
            "batch_size": settings["batch_size"],
            "seed": settings["seed"],
            "generation": {
                key: settings[key]
                for key in (
                    "max_new_tokens",
                    "do_sample",
                    "top_p",
                    "top_k",
                    "temperature",
                    "suppress_eos",
                    "stop_strings",
                )
            },
            "watermark": watermark,
            "prompt_revision": "c4-or-eli5-v1",
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
        }
        resolved = {**settings, "model": model, "dataset": dataset}
        return ResolvedStageDefinition(
            settings=resolved,
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision="generated-text-v2",
            resource_key=(
                f"causal:{model['checkpoint']}@{model['revision']}:"
                f"{settings['device']}:{settings['dtype']}"
            ),
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if inputs:
            raise ValueError("generation does not accept an input Artifact")
        return definition

    def prepare(
        self, context: StageExecutionContext
    ) -> "_GenerationExecution":
        return _GenerationExecution(context)


class _GenerationExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        model_spec = context.semantic_settings["model"]
        self.model, self.tokenizer = context.runtime.get(
            model_spec,
            device=context.execution_settings["device"],
            dtype=context.execution_settings["dtype"],
            padding_side="left",
        )
        self.watermarker, self.no_watermark = WATERMARK_SCHEMES.generator(
            context.semantic_settings["watermark"],
            self.model,
            self.tokenizer,
        )
        source_samples = load_selected_samples(
            context.semantic_settings["dataset"]
        )
        repetitions = context.semantic_settings["repetitions"]
        self.samples = []
        for sample in source_samples:
            source_sample_id = sample["sample_id"]
            for repetition in range(repetitions):
                sample_id = (
                    source_sample_id
                    if repetitions == 1
                    else f"{source_sample_id}:repeat:{repetition}"
                )
                self.samples.append(
                    {
                        **sample,
                        "sample_id": sample_id,
                        "source_sample_id": source_sample_id,
                        "repetition": repetition,
                    }
                )

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.samples,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item["sample_id"] for item in batch)
            identity = identity_for(
                {"ordinal": ordinal, "sample_ids": sample_ids},
                prefix="batch",
            )
            result.append(
                WorkItem(
                    identity=identity,
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=batch,
                )
            )
        return result

    def execute(self, item: WorkItem) -> WorkResult:
        dataset = self.context.semantic_settings["dataset"]
        prompts = [
            format_prompt(sample, dataset, self.tokenizer)
            for sample in item.payload
        ]
        generation = self.context.semantic_settings["generation"]
        suppress_tokens = (
            [self.tokenizer.eos_token_id]
            if generation["suppress_eos"]
            else None
        )
        seed = derived_seed(
            self.context.semantic_settings["seed"], item.identity
        )
        with torch.no_grad():
            texts = self.watermarker(
                prompts=[model_prompt for _, model_prompt in prompts],
                max_new_tokens=generation["max_new_tokens"],
                stop_strings=generation["stop_strings"],
                seed=seed,
                do_sample=generation["do_sample"],
                top_p=generation["top_p"],
                top_k=generation["top_k"],
                temperature=generation["temperature"],
                suppress_tokens=suppress_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                no_watermark=self.no_watermark,
            )
        if not isinstance(texts, list):
            texts = [texts]
        records = []
        for sample, (source_prompt, model_prompt), text in zip(
            item.payload, prompts, texts
        ):
            token_num = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
            record = {
                "sample_id": sample["sample_id"],
                "source_prompt": source_prompt,
                "prompt_text": model_prompt,
                "generated_text": text,
                "generated_token_num": token_num,
                "source_sample_id": sample["source_sample_id"],
                "repetition": sample["repetition"],
            }
            for field in ("index", "original_index", "url"):
                if field in sample:
                    record[field] = sample[field]
            records.append(record)
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        lengths = [record["generated_token_num"] for record in records]
        return {
            "sample_num": len(records),
            "source_sample_num": len(
                {record["source_sample_id"] for record in records}
            ),
            "repetitions": self.context.semantic_settings["repetitions"],
            "generated_token_num": sum(lengths),
            "mean_generated_token_num": (
                statistics.mean(lengths) if lengths else 0.0
            ),
            "watermark": self.context.semantic_settings["watermark"],
        }

    def close(self) -> None:
        pass
