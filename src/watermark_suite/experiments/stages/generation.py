from __future__ import annotations

import statistics

import torch

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..dataset_prompt import DATASET_PROMPTS, PromptSample
from ..errors import PlanValidationError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import (
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)


class GenerationStageAdapter:
    kind = "generation"
    revision = "generation-v3"
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
        prompt_population = DATASET_PROMPTS.resolve(
            settings["dataset"],
            catalog=context.datasets,
            repository=context.repository,
            sample_num=settings["num_samples"],
        )
        watermark = WATERMARK_SCHEMES.resolve(
            settings["watermark"], context.repository
        )
        semantic = {
            "model": model,
            "prompt_population": prompt_population.to_dict(),
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
        }
        execution = {
            "device": settings["device"],
            "dtype": settings["dtype"],
        }
        resolved = {
            **settings,
            "model": model,
            "dataset": prompt_population.to_dict(),
        }
        return ResolvedStageDefinition(
            settings=resolved,
            semantic_settings=semantic,
            execution_settings=execution,
            artifact_schema_revision="generated-text-v3",
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
        population = DATASET_PROMPTS.materialize(
            context.semantic_settings["prompt_population"],
            tokenizer=self.tokenizer,
        )
        self.samples = list(
            population.repeated(
                context.semantic_settings["repetitions"]
            ).samples
        )

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.samples,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(item.sample_id for item in batch)
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
        samples: list[PromptSample] = item.payload
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
                prompts=[sample.model_prompt for sample in samples],
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
        for sample, text in zip(samples, texts):
            token_num = len(
                self.tokenizer.encode(text, add_special_tokens=False)
            )
            record = {
                "sample_id": sample.sample_id,
                "source_prompt": sample.source_prompt,
                "prompt_text": sample.model_prompt,
                "generated_text": text,
                "generated_token_num": token_num,
                "source_sample_id": sample.source_sample_id,
                "repetition": sample.repetition,
            }
            for field in ("index", "original_index", "url"):
                if field in sample.payload:
                    record[field] = sample.payload[field]
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
