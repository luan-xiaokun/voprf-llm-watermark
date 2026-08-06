from __future__ import annotations

import random
import re
import statistics
from typing import Any

from ...attacks.synonym_replacement import SynonymReplacer
from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import derived_seed, identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..openai_paraphrase import OpenAIParaphraser
from .common import (
    artifact_records,
    batches,
    require,
    resolve_model,
    validate_execution_settings,
)


_MASKED_LM_REPLACEMENT_FIELDS = {
    "method",
    "model",
    "replacement_rate",
    "top_k",
    "candidate_sampling",
}
_OPENAI_PARAPHRASE_FIELDS = {
    "method",
    "model",
    "max_output_tokens",
    "temperature",
    "reasoning_effort",
    "instruction",
}
_REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}


def _resolve_transformation(
    value: Any,
    context: ResolutionContext,
) -> JsonObject:
    if not isinstance(value, dict):
        raise PlanValidationError("transformation must be a mapping")
    method = value.get("method")
    if method == "masked-lm-replacement":
        unknown = sorted(set(value) - _MASKED_LM_REPLACEMENT_FIELDS)
        missing = sorted(_MASKED_LM_REPLACEMENT_FIELDS - set(value))
        if unknown or missing:
            messages = []
            if unknown:
                messages.append("unknown: " + ", ".join(unknown))
            if missing:
                messages.append("missing: " + ", ".join(missing))
            raise PlanValidationError(
                "masked-lm-replacement settings are invalid ("
                + "; ".join(messages)
                + ")"
            )
        rate = value["replacement_rate"]
        if (
            not isinstance(rate, (int, float))
            or isinstance(rate, bool)
            or not 0 < rate < 1
        ):
            raise PlanValidationError(
                "masked-lm-replacement replacement_rate must be between "
                "zero and one"
            )
        if (
            not isinstance(value["top_k"], int)
            or isinstance(value["top_k"], bool)
            or value["top_k"] <= 0
        ):
            raise PlanValidationError(
                "masked-lm-replacement top_k must be positive"
            )
        if value["candidate_sampling"] != "score-weighted":
            raise PlanValidationError(
                "masked-lm-replacement candidate_sampling must be "
                "score-weighted"
            )
        model = resolve_model(
            value["model"],
            models=context.models,
            repository=context.repository,
        )
        return {
            **value,
            "model": model,
            "implementation_revision": "masked-lm-roundtrip-window-v2",
        }
    if method == "word-deletion":
        unknown = sorted(set(value) - {"method", "rate"})
        if unknown:
            raise PlanValidationError(
                "word-deletion has unknown settings: " + ", ".join(unknown)
            )
        rate = value.get("rate")
        if not isinstance(rate, (int, float)) or not 0 < rate < 1:
            raise PlanValidationError(
                "word-deletion rate must be between zero and one"
            )
        return dict(value)
    if method == "openai-paraphrase":
        unknown = sorted(set(value) - _OPENAI_PARAPHRASE_FIELDS)
        missing = sorted(_OPENAI_PARAPHRASE_FIELDS - set(value))
        if unknown or missing:
            messages = []
            if unknown:
                messages.append("unknown: " + ", ".join(unknown))
            if missing:
                messages.append("missing: " + ", ".join(missing))
            raise PlanValidationError(
                "openai-paraphrase settings are invalid ("
                + "; ".join(messages)
                + ")"
            )
        if (
            not isinstance(value["model"], str)
            or not value["model"].strip()
        ):
            raise PlanValidationError(
                "openai-paraphrase model must be non-empty"
            )
        if (
            not isinstance(value["max_output_tokens"], int)
            or isinstance(value["max_output_tokens"], bool)
            or value["max_output_tokens"] <= 0
        ):
            raise PlanValidationError(
                "openai-paraphrase max_output_tokens must be positive"
            )
        temperature = value["temperature"]
        if temperature is not None and (
            not isinstance(temperature, (int, float))
            or isinstance(temperature, bool)
            or not 0 <= temperature <= 2
        ):
            raise PlanValidationError(
                "openai-paraphrase temperature must be between zero "
                "and two or null"
            )
        reasoning_effort = value["reasoning_effort"]
        if (
            reasoning_effort is not None
            and reasoning_effort not in _REASONING_EFFORTS
        ):
            raise PlanValidationError(
                "openai-paraphrase reasoning_effort must be one of "
                + ", ".join(sorted(_REASONING_EFFORTS))
                + " or null"
            )
        model = value["model"]
        if model.startswith("gpt-3.5") and reasoning_effort is not None:
            raise PlanValidationError(
                f"openai-paraphrase model {model!r} does not support "
                "reasoning_effort"
            )
        if (
            model.startswith("gpt-5.6")
            and temperature is not None
            and reasoning_effort != "none"
        ):
            raise PlanValidationError(
                f"openai-paraphrase model {model!r} does not support "
                "temperature unless reasoning_effort is 'none'"
            )
        if not isinstance(value["instruction"], str) or not value[
            "instruction"
        ].strip():
            raise PlanValidationError(
                "openai-paraphrase instruction must be non-empty"
            )
        return dict(value)
    raise PlanValidationError(
        "transformation.method must be masked-lm-replacement, "
        "word-deletion, or openai-paraphrase"
    )


class RobustnessStageAdapter:
    kind = "robustness"
    revision = "robustness-v3"
    accepted_settings = {
        "transformation",
        "batch_size",
        "openai_concurrency",
        "target_field",
        "seed",
        "device",
        "dtype",
    }
    _required = accepted_settings - {"openai_concurrency"}

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        validate_execution_settings(settings)
        if not isinstance(settings["batch_size"], int) or settings[
            "batch_size"
        ] <= 0:
            raise PlanValidationError("batch_size must be positive")
        openai_concurrency = settings.get(
            "openai_concurrency",
            settings["batch_size"],
        )
        if (
            not isinstance(openai_concurrency, int)
            or isinstance(openai_concurrency, bool)
            or openai_concurrency <= 0
        ):
            raise PlanValidationError("openai_concurrency must be positive")
        if not isinstance(settings["seed"], int):
            raise PlanValidationError("seed must be an integer")
        if (
            not isinstance(settings["target_field"], str)
            or not settings["target_field"]
        ):
            raise PlanValidationError("target_field must be non-empty")
        transformation = _resolve_transformation(
            settings["transformation"],
            context,
        )
        resource_key = "cpu:robustness"
        if transformation["method"] == "openai-paraphrase":
            resource_key = f"remote:openai:{transformation['model']}"
        elif transformation["method"] == "masked-lm-replacement":
            model = transformation["model"]
            resource_key = (
                f"local:fill-mask:{model['checkpoint']}@{model['revision']}:"
                f"{settings['device']}:{settings['dtype']}"
            )
        return ResolvedStageDefinition(
            settings={
                **settings,
                "transformation": transformation,
            },
            semantic_settings={
                "transformation": transformation,
                "batch_size": settings["batch_size"],
                "target_field": settings["target_field"],
                "seed": settings["seed"],
            },
            execution_settings={
                "device": settings["device"],
                "dtype": settings["dtype"],
                "openai_concurrency": openai_concurrency,
            },
            artifact_schema_revision="robustness-text-v2",
            resource_key=resource_key,
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if len(inputs) != 1:
            raise PlanValidationError(
                "robustness requires exactly one source Artifact"
            )
        source_semantic = inputs[0].manifest.get("semantic_settings", {})
        model = source_semantic.get("model")
        watermark = source_semantic.get("watermark")
        if not isinstance(model, dict) or not isinstance(watermark, dict):
            raise PlanValidationError(
                "source Artifact does not declare model/watermark provenance"
            )
        return ResolvedStageDefinition(
            settings=dict(definition.settings),
            semantic_settings={
                **definition.semantic_settings,
                "model": model,
                "watermark": watermark,
            },
            execution_settings=definition.execution_settings,
            artifact_schema_revision=definition.artifact_schema_revision,
            resource_key=definition.resource_key,
        )

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> "_RobustnessExecution":
        return _RobustnessExecution(context)


class _RobustnessExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        self.records = artifact_records(context.inputs[0])
        transformation = context.semantic_settings["transformation"]
        self.paraphraser = (
            OpenAIParaphraser()
            if transformation["method"] == "openai-paraphrase"
            else None
        )
        self.synonym_replacer = None
        if transformation["method"] == "masked-lm-replacement":
            unmasker, tokenizer = context.runtime.fill_mask(
                transformation["model"],
                device=context.execution_settings["device"],
                dtype=context.execution_settings["dtype"],
            )
            self.synonym_replacer = SynonymReplacer(tokenizer, unmasker)

    def work_items(self) -> list[WorkItem]:
        result = []
        for ordinal, batch in enumerate(
            batches(
                self.records,
                self.context.semantic_settings["batch_size"],
            )
        ):
            sample_ids = tuple(record["sample_id"] for record in batch)
            result.append(
                WorkItem(
                    identity=identity_for(
                        {"ordinal": ordinal, "sample_ids": sample_ids},
                        prefix="batch",
                    ),
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=batch,
                )
            )
        return result

    @staticmethod
    def _delete_words(text: str, rate: float, seed: int) -> str:
        pieces = re.findall(r"\S+|\s+", text)
        word_indices = [
            index for index, piece in enumerate(pieces) if not piece.isspace()
        ]
        if len(word_indices) <= 1:
            return text
        rng = random.Random(seed)
        deleted = {
            index for index in word_indices if rng.random() < rate
        }
        if len(deleted) == len(word_indices):
            deleted.remove(word_indices[0])
        transformed = "".join(
            piece for index, piece in enumerate(pieces) if index not in deleted
        )
        return re.sub(r"\s+", " ", transformed).strip()

    def execute(self, item: WorkItem) -> WorkResult:
        semantic = self.context.semantic_settings
        transformation = semantic["transformation"]
        target_field = semantic["target_field"]
        missing = [
            record["sample_id"]
            for record in item.payload
            if target_field not in record
        ]
        if missing:
            raise PlanValidationError(
                f"source records lack target_field={target_field!r}: "
                f"{missing[:5]}"
            )
        texts = [str(record[target_field]) for record in item.payload]
        if transformation["method"] == "word-deletion":
            transformed = [
                self._delete_words(
                    text,
                    transformation["rate"],
                    derived_seed(semantic["seed"], record["sample_id"]),
                )
                for record, text in zip(item.payload, texts)
            ]
            provenance = [{} for _ in transformed]
        elif transformation["method"] == "openai-paraphrase":
            paraphrases = self.paraphraser.paraphrase_many(
                texts,
                concurrency=self.context.execution_settings[
                    "openai_concurrency"
                ],
                model=transformation["model"],
                instruction=transformation["instruction"],
                max_output_tokens=transformation["max_output_tokens"],
                temperature=transformation["temperature"],
                reasoning_effort=transformation["reasoning_effort"],
            )
            transformed = [result.text for result in paraphrases]
            provenance = [result.provenance for result in paraphrases]
        else:
            replacements = self.synonym_replacer.replace_many(
                texts,
                seeds=[
                    derived_seed(semantic["seed"], record["sample_id"])
                    for record in item.payload
                ],
                replacement_rate=transformation["replacement_rate"],
                top_k=transformation["top_k"],
                candidate_sampling=transformation["candidate_sampling"],
                batch_size=semantic["batch_size"],
            )
            transformed = [result.text for result in replacements]
            provenance = [result.provenance() for result in replacements]
        records = []
        for source, original, changed, remote_provenance in zip(
            item.payload,
            texts,
            transformed,
            provenance,
        ):
            records.append(
                {
                    **source,
                    "sample_id": source["sample_id"],
                    "source_artifact": self.context.inputs[0].identity.value,
                    "original_text": original,
                    "transformed_text": changed,
                    "robustness": {
                        "method": transformation["method"],
                        "original_character_num": len(original),
                        "transformed_character_num": len(changed),
                        "character_ratio": (
                            len(changed) / len(original) if original else 0.0
                        ),
                        **remote_provenance,
                    },
                }
            )
        return WorkResult(records=tuple(records))

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        ratios = [
            record["robustness"]["character_ratio"] for record in records
        ]
        summary = {
            "sample_num": len(records),
            "transformation": self.context.semantic_settings[
                "transformation"
            ],
            "mean_character_ratio": (
                statistics.mean(ratios) if ratios else 0.0
            ),
        }
        api_records = [
            record["robustness"]
            for record in records
            if record["robustness"].get("provider") == "openai"
        ]
        if api_records:
            usage_fields = (
                "input_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "output_tokens",
                "reasoning_output_tokens",
                "total_tokens",
            )
            summary["openai"] = {
                "request_num": len(api_records),
                "mean_latency_seconds": statistics.mean(
                    record["latency_seconds"] for record in api_records
                ),
                "usage": {
                    field: sum(
                        record["usage"][field] for record in api_records
                    )
                    for field in usage_fields
                },
            }
        replacement_records = [
            record["robustness"]
            for record in records
            if record["robustness"]["method"]
            == "masked-lm-replacement"
        ]
        if replacement_records:
            eligible_num = sum(
                record["eligible_word_count"]
                for record in replacement_records
            )
            selected_num = sum(
                record["selected_word_count"]
                for record in replacement_records
            )
            replaced_num = sum(
                record["replaced_word_count"]
                for record in replacement_records
            )
            ranked_replacement_num = sum(
                record["replaced_word_count"]
                for record in replacement_records
                if record["mean_selected_candidate_rank"] is not None
            )
            rank_sum = sum(
                record["mean_selected_candidate_rank"]
                * record["replaced_word_count"]
                for record in replacement_records
                if record["mean_selected_candidate_rank"] is not None
            )
            summary["masked_lm_replacement"] = {
                "eligible_word_count": eligible_num,
                "selected_word_count": selected_num,
                "replaced_word_count": replaced_num,
                "failed_replacement_count": selected_num - replaced_num,
                "realized_replacement_rate": (
                    replaced_num / eligible_num if eligible_num else 0.0
                ),
                "mean_selected_candidate_rank": (
                    rank_sum / ranked_replacement_num
                    if ranked_replacement_num
                    else None
                ),
            }
        return summary

    def close(self) -> None:
        pass
