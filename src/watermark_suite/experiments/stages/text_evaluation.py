from __future__ import annotations

import math
import re
import statistics
from collections import defaultdict

import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..openai_embedding import OpenAIEmbedder
from .common import artifact_records, batches, require, resolve_model


def _describe(values: list[float]) -> JsonObject:
    finite = [float(value) for value in values if math.isfinite(value)]
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(finite),
        "mean": statistics.mean(finite),
        "median": statistics.median(finite),
        "std": statistics.pstdev(finite),
        "min": min(finite),
        "max": max(finite),
    }


def _normalized(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.maximum(norms, 1e-12)


def _cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(_normalized(left) * _normalized(right), axis=1)


def _tokens(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def _distinct_n(texts: list[str], n: int) -> float:
    grams = []
    for text in texts:
        tokens = _tokens(text)
        grams.extend(
            tuple(tokens[index : index + n])
            for index in range(len(tokens) - n + 1)
        )
    return len(set(grams)) / len(grams) if grams else 0.0


def _self_bleu(texts: list[str]) -> float:
    tokenized = [_tokens(text) for text in texts]
    if len(tokenized) < 2:
        return 0.0
    smoothing = SmoothingFunction().method1
    scores = []
    for index, hypothesis in enumerate(tokenized):
        if not hypothesis:
            continue
        references = tokenized[:index] + tokenized[index + 1 :]
        scores.append(
            sentence_bleu(
                references,
                hypothesis,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing,
            )
        )
    return statistics.mean(scores) if scores else 0.0


def _vendi_score(embeddings: np.ndarray) -> float:
    if len(embeddings) == 0:
        return 0.0
    similarity = _normalized(embeddings) @ _normalized(embeddings).T
    eigenvalues = np.clip(
        np.linalg.eigvalsh(similarity) / len(embeddings),
        0.0,
        None,
    )
    total = eigenvalues.sum()
    if total <= 0:
        return 0.0
    probabilities = eigenvalues / total
    nonzero = probabilities[probabilities > 0]
    return float(np.exp(-np.sum(nonzero * np.log(nonzero))))


class TextEvaluationStageAdapter:
    kind = "text-evaluation"
    revision = "text-evaluation-v2"
    accepted_settings = {
        "metric_set",
        "embedding_provider",
        "embedding_model",
        "embedding_dimensions",
        "batch_size",
        "reference_field",
        "target_field",
        "group_field",
        "device",
    }
    _required = accepted_settings - {
        "embedding_provider",
        "embedding_dimensions",
    }

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        metrics = settings["metric_set"]
        if (
            not isinstance(metrics, list)
            or not metrics
            or len(metrics) != len(set(metrics))
            or any(
                metric not in {"similarity", "diversity"}
                for metric in metrics
            )
        ):
            raise PlanValidationError(
                "metric_set must contain unique similarity/diversity names"
            )
        if settings["batch_size"] <= 0:
            raise PlanValidationError("batch_size must be positive")
        if settings["device"] not in {"auto", "cpu", "cuda"}:
            raise PlanValidationError("device must be auto, cpu, or cuda")
        if (
            not isinstance(settings["target_field"], str)
            or not settings["target_field"]
        ):
            raise PlanValidationError("target_field must be non-empty")
        if settings["group_field"] is not None and (
            not isinstance(settings["group_field"], str)
            or not settings["group_field"]
        ):
            raise PlanValidationError(
                "group_field must be non-empty or null"
            )
        if (
            "similarity" in metrics
            and (
                not isinstance(settings["reference_field"], str)
                or not settings["reference_field"]
            )
        ):
            raise PlanValidationError(
                "similarity requires a non-empty reference_field"
            )
        provider = settings.get("embedding_provider", "local")
        dimensions = settings.get("embedding_dimensions")
        if provider not in {"local", "openai"}:
            raise PlanValidationError(
                "embedding_provider must be local or openai"
            )
        if dimensions is not None and (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions <= 0
        ):
            raise PlanValidationError(
                "embedding_dimensions must be positive or null"
            )
        if provider == "local":
            if dimensions is not None:
                raise PlanValidationError(
                    "embedding_dimensions is only supported by openai"
                )
            model = resolve_model(
                settings["embedding_model"],
                models=context.models,
                repository=context.repository,
            )
        else:
            requested_model = settings["embedding_model"]
            if (
                not isinstance(requested_model, str)
                or not requested_model.strip()
            ):
                raise PlanValidationError(
                    "OpenAI embedding_model must be non-empty"
                )
            model = {
                "provider": "openai",
                "checkpoint": requested_model,
                "revision": "api",
            }
        semantic = {
            "metric_set": list(metrics),
            "embedding_provider": provider,
            "embedding_model": model,
            "embedding_dimensions": dimensions,
            "batch_size": settings["batch_size"],
            "reference_field": settings["reference_field"],
            "target_field": settings["target_field"],
            "group_field": settings["group_field"],
        }
        return ResolvedStageDefinition(
            settings={
                **settings,
                "embedding_provider": provider,
                "embedding_model": model,
                "embedding_dimensions": dimensions,
            },
            semantic_settings=semantic,
            execution_settings={"device": settings["device"]},
            artifact_schema_revision="text-evaluation-v1",
            resource_key=(
                f"remote:openai-embedding:{model['checkpoint']}"
                if provider == "openai"
                else (
                    f"encoder:{model['checkpoint']}@{model['revision']}:"
                    f"{settings['device']}"
                )
            ),
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if len(inputs) != 1:
            raise PlanValidationError(
                "text-evaluation requires exactly one source Artifact"
            )
        return definition

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> "_TextEvaluationExecution":
        return _TextEvaluationExecution(context)


class _TextEvaluationExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        self.records = artifact_records(context.inputs[0])
        provider = context.semantic_settings["embedding_provider"]
        self.encoder = (
            context.runtime.encoder(
                context.semantic_settings["embedding_model"],
                device=context.execution_settings["device"],
            )
            if provider == "local"
            else None
        )
        self.openai_embedder = (
            OpenAIEmbedder() if provider == "openai" else None
        )

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

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.openai_embedder is not None:
            model = self.context.semantic_settings["embedding_model"]
            return np.asarray(
                self.openai_embedder.embed_many(
                    texts,
                    model=model["checkpoint"],
                    dimensions=self.context.semantic_settings[
                        "embedding_dimensions"
                    ],
                ),
                dtype=float,
            )
        return np.asarray(
            self.encoder.encode(
                texts,
                batch_size=self.context.semantic_settings["batch_size"],
                convert_to_numpy=True,
                show_progress_bar=False,
            ),
            dtype=float,
        )

    def execute(self, item: WorkItem) -> WorkResult:
        semantic = self.context.semantic_settings
        fields = [semantic["target_field"]]
        if "similarity" in semantic["metric_set"]:
            fields.append(semantic["reference_field"])
        if semantic["group_field"] is not None:
            fields.append(semantic["group_field"])
        missing = [
            record["sample_id"]
            for record in item.payload
            if any(field not in record for field in fields)
        ]
        if missing:
            raise PlanValidationError(
                f"source records lack text-evaluation fields: {missing[:5]}"
            )
        targets = [
            str(record[semantic["target_field"]]) for record in item.payload
        ]
        similarities: list[float | None] = [None] * len(targets)
        if "similarity" in semantic["metric_set"]:
            references = [
                str(record[semantic["reference_field"]])
                for record in item.payload
            ]
            similarities = _cosine(
                self._encode(references),
                self._encode(targets),
            ).tolist()
        records = []
        for source, target, similarity in zip(
            item.payload, targets, similarities
        ):
            evaluation: JsonObject = {}
            if similarity is not None:
                evaluation["cosine_similarity"] = similarity
            records.append(
                {
                    "sample_id": source["sample_id"],
                    "source_artifact": self.context.inputs[0].identity.value,
                    "target_text": target,
                    "group": (
                        source[semantic["group_field"]]
                        if semantic["group_field"] is not None
                        else "__all__"
                    ),
                    "text_evaluation": evaluation,
                }
            )
        return WorkResult(records=tuple(records))

    def _diversity(self, records: list[JsonObject]) -> JsonObject:
        grouped: dict[str, list[str]] = defaultdict(list)
        for record in records:
            grouped[str(record["group"])].append(record["target_text"])
        groups = []
        for group, texts in sorted(grouped.items()):
            embeddings = self._encode(texts)
            groups.append(
                {
                    "group": group,
                    "sample_num": len(texts),
                    "distinct_1": _distinct_n(texts, 1),
                    "distinct_2": _distinct_n(texts, 2),
                    "distinct_3": _distinct_n(texts, 3),
                    "self_bleu_4": _self_bleu(texts),
                    "vendi_score": _vendi_score(embeddings),
                }
            )
        metric_names = (
            "distinct_1",
            "distinct_2",
            "distinct_3",
            "self_bleu_4",
            "vendi_score",
        )
        return {
            "group_num": len(groups),
            "aggregate": {
                name: (
                    statistics.mean(group[name] for group in groups)
                    if groups
                    else 0.0
                )
                for name in metric_names
            },
            "groups": groups,
        }

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        metrics = self.context.semantic_settings["metric_set"]
        summary: JsonObject = {
            "sample_num": len(records),
            "metric_set": metrics,
        }
        if "similarity" in metrics:
            summary["similarity"] = _describe(
                [
                    record["text_evaluation"]["cosine_similarity"]
                    for record in records
                ]
            )
        if "diversity" in metrics:
            summary["diversity"] = self._diversity(records)
        return summary

    def close(self) -> None:
        pass
