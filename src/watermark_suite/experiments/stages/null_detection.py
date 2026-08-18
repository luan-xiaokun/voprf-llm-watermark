from __future__ import annotations

import hashlib
import math
import statistics
import threading
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

import pyarrow.parquet as pq
import torch

from ..adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from ..errors import PlanValidationError
from ..identity import identity_for
from ..models import ArtifactRef, JsonObject, WorkItem, WorkResult
from ..scheme_registry import WATERMARK_SCHEMES
from .common import artifact_records, require


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


class NullDetectionStageAdapter:
    kind = "null-detection"
    revision = "null-detection-v1"
    accepted_settings = {
        "detector_watermark",
        "sample_num",
        "detector_batch_size",
        "concurrent_batches",
        "max_in_flight_batches",
        "intraop_threads",
        "significance_levels",
        "device",
    }
    _required = accepted_settings

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition:
        require(settings, self._required, kind=self.kind)
        for field in (
            "sample_num",
            "detector_batch_size",
            "concurrent_batches",
            "max_in_flight_batches",
            "intraop_threads",
        ):
            value = settings[field]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise PlanValidationError(f"{field} must be a positive integer")
        if settings["max_in_flight_batches"] < settings["concurrent_batches"]:
            raise PlanValidationError(
                "max_in_flight_batches must be at least concurrent_batches"
            )
        levels = settings["significance_levels"]
        if (
            not isinstance(levels, list)
            or not levels
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not 0 < float(value) < 1
                for value in levels
            )
            or len({float(value) for value in levels}) != len(levels)
        ):
            raise PlanValidationError(
                "significance_levels must contain unique values between zero and one"
            )
        watermark = WATERMARK_SCHEMES.resolve(
            settings["detector_watermark"], context.repository
        )
        if watermark["method"] not in {"vow", "lefthash", "selfhash", "rdf"}:
            raise PlanValidationError(
                "null-detection supports vow, lefthash, selfhash, and rdf"
            )
        semantic = {
            "detector_watermark": watermark,
            "sample_num": settings["sample_num"],
            "significance_levels": sorted(
                (float(value) for value in levels), reverse=True
            ),
        }
        return ResolvedStageDefinition(
            settings={**settings, "detector_watermark": watermark},
            semantic_settings=semantic,
            execution_settings={
                "device": settings["device"],
                "detector_batch_size": settings["detector_batch_size"],
                "worker_count": settings["concurrent_batches"],
                "max_in_flight": settings["max_in_flight_batches"],
                "intraop_threads": settings["intraop_threads"],
                "resume_compatibility": {
                    "detector_batch_size": settings["detector_batch_size"]
                },
            },
            artifact_schema_revision="null-detection-v1",
            resource_key=f"input-bound:null-detection:{watermark['method']}",
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        if len(inputs) != 1:
            raise PlanValidationError(
                "null-detection requires one token-window-corpus Artifact"
            )
        source = inputs[0]
        if (
            source.manifest.get("kind") != "token-window-corpus"
            or source.schema_revision != "token-window-corpus-v1"
        ):
            raise PlanValidationError(
                "null-detection input must be token-window-corpus-v1"
            )
        source_semantic = source.manifest.get("semantic_settings", {})
        tokenizer = source_semantic.get("tokenizer")
        population = source_semantic.get("population")
        window = source_semantic.get("window")
        if not all(isinstance(value, dict) for value in (tokenizer, population, window)):
            raise PlanValidationError(
                "token-window-corpus lacks tokenizer/population/window provenance"
            )
        available = int(source_semantic.get("sample_num", 0))
        requested = definition.semantic_settings["sample_num"]
        if requested > available:
            raise PlanValidationError(
                f"null-detection requests {requested} samples, source has {available}"
            )
        semantic = {
            **definition.semantic_settings,
            "tokenizer": tokenizer,
            "population": population,
            "window": window,
        }
        return ResolvedStageDefinition(
            settings={
                **definition.settings,
                "derived_tokenizer": tokenizer,
                "derived_population": population,
                "derived_window": window,
            },
            semantic_settings=semantic,
            execution_settings=definition.execution_settings,
            artifact_schema_revision=definition.artifact_schema_revision,
            resource_key=(
                f"tokenizer:{tokenizer['tokenizer_checkpoint']}@"
                f"{tokenizer['tokenizer_revision']}:"
                f"{semantic['detector_watermark']['method']}"
            ),
        )

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> "_NullDetectionExecution":
        return _NullDetectionExecution(context)


class _NullDetectionExecution:
    def __init__(self, context: StageExecutionContext) -> None:
        self.context = context
        tokenizer_model = {
            **context.semantic_settings["tokenizer"],
            "checkpoint": context.semantic_settings["tokenizer"][
                "tokenizer_checkpoint"
            ],
            "revision": context.semantic_settings["tokenizer"][
                "tokenizer_revision"
            ],
            "location": context.semantic_settings["tokenizer"][
                "tokenizer_location"
            ],
            "verification": context.semantic_settings["tokenizer"][
                "tokenizer_verification"
            ],
        }
        self.tokenizer = context.runtime.tokenizer(
            tokenizer_model, padding_side="left"
        )
        self.source = context.inputs[0]
        self.detector_dimensions = WATERMARK_SCHEMES.analysis_dimensions(
            context.semantic_settings["detector_watermark"]
        )
        self._workers = threading.local()
        self._shared_rdf: Any | None = None
        if context.semantic_settings["detector_watermark"]["method"] == "rdf":
            self._shared_rdf, _ = WATERMARK_SCHEMES.detector(
                context.semantic_settings["detector_watermark"],
                self.tokenizer,
                device=context.execution_settings["device"],
            )
            self._shared_rdf.prepare_token_detection()

    def _detector(self) -> Any:
        if self._shared_rdf is not None:
            return self._shared_rdf
        detector = getattr(self._workers, "detector", None)
        if detector is None:
            detector, _ = WATERMARK_SCHEMES.detector(
                self.context.semantic_settings["detector_watermark"],
                self.tokenizer,
                device=self.context.execution_settings["device"],
            )
            if self.context.semantic_settings["detector_watermark"]["method"] in {
                "lefthash",
                "selfhash",
            }:
                detector.cache_ngram_scores = False
            self._workers.detector = detector
        return detector

    def work_items(self) -> Iterable[WorkItem]:
        batch_size = self.context.execution_settings["detector_batch_size"]
        target = self.context.semantic_settings["sample_num"]
        selected = 0
        ordinal = 0
        for part in artifact_records(self.source):
            path = self.source.path / part["part"]
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size):
                rows = batch.to_pylist()
                if selected + len(rows) > target:
                    rows = rows[: target - selected]
                if not rows:
                    return
                sample_ids = tuple(row["sample_id"] for row in rows)
                yield WorkItem(
                    identity=identity_for(
                        {"ordinal": ordinal, "sample_ids": sample_ids},
                        prefix="null-detection-batch",
                    ),
                    ordinal=ordinal,
                    sample_identities=sample_ids,
                    payload=tuple(rows),
                )
                selected += len(rows)
                ordinal += 1
                if selected >= target:
                    return
        if selected != target:
            raise PlanValidationError(
                f"token-window Artifact yielded {selected} samples, expected {target}"
            )

    def _rdf_seed(self, sample_id: str) -> int:
        digest = hashlib.sha256(
            f"{self.detector_dimensions['identity']}\0{sample_id}".encode(
                "utf-8"
            )
        ).digest()
        return int.from_bytes(digest[:8], "big")

    def execute(self, item: WorkItem) -> WorkResult:
        detector = self._detector()
        method = self.context.semantic_settings["detector_watermark"]["method"]
        token_lists = [
            [int(value) for value in row["token_ids"]] for row in item.payload
        ]
        if method == "vow":
            results = detector.local_batch_detect_tokens(token_lists)
        elif method in {"lefthash", "selfhash"}:
            results = [
                detector.detect(
                    tokenized_text=torch.as_tensor(
                        token_ids, dtype=torch.long, device=detector.device
                    ),
                    return_z_at_T=False,
                )
                for token_ids in token_lists
            ]
        else:
            results = [
                detector.detect_token_ids(
                    token_ids,
                    sample_seed=self._rdf_seed(row["sample_id"]),
                    intraop_threads=self.context.execution_settings[
                        "intraop_threads"
                    ],
                )
                for row, token_ids in zip(item.payload, token_lists)
            ]
        if len(results) != len(item.payload):
            raise PlanValidationError("detector returned a misaligned result batch")
        records = []
        for source, result in zip(item.payload, results):
            detection = _json_value(result)
            p_value = detection.get("p_value")
            if (
                not isinstance(p_value, (int, float))
                or not math.isfinite(float(p_value))
                or not 0 <= float(p_value) <= 1
            ):
                raise PlanValidationError(
                    f"detector returned invalid p-value for {source['sample_id']}"
                )
            records.append(
                {
                    "sample_id": source["sample_id"],
                    "source_artifact": self.source.identity.value,
                    "source_token_sha256": source["token_sha256"],
                    "detection": detection,
                }
            )
        return WorkResult(records=tuple(records))

    def summarize(self, records: Iterable[JsonObject]) -> JsonObject:
        records = list(records)
        expected = self.context.semantic_settings["sample_num"]
        if len(records) != expected:
            raise PlanValidationError(
                f"null-detection produced {len(records)} records, expected {expected}"
            )
        p_values = [float(record["detection"]["p_value"]) for record in records]
        levels = self.context.semantic_settings["significance_levels"]
        counts = {
            f"{level:.0e}": {
                "positive_num": sum(value < level for value in p_values),
                "sample_num": len(p_values),
            }
            for level in levels
        }
        rates = {
            key: value["positive_num"] / value["sample_num"]
            for key, value in counts.items()
        }
        watermark = self.context.semantic_settings["detector_watermark"]
        minimum_attainable = None
        if watermark["method"] == "rdf":
            minimum_attainable = 1.0 / (watermark["n_runs"] + 1.0)
        return {
            "sample_num": len(records),
            "population": self.context.semantic_settings["population"],
            "window": self.context.semantic_settings["window"],
            "detector_watermark": watermark,
            "detector_scheme": self.detector_dimensions,
            "significance_levels": levels,
            "false_positive_counts": counts,
            "false_positive_rate": rates,
            "p_value": {
                "min": min(p_values),
                "median": statistics.median(p_values),
                "max": max(p_values),
                "minimum_attainable": minimum_attainable,
            },
            "source_artifact": self.source.identity.value,
            "retention": {
                "raw_text_retained": False,
                "sample_p_values_retained": True,
            },
        }

    def close(self) -> None:
        self._shared_rdf = None
