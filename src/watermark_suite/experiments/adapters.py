from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from .models import ArtifactRef, JsonObject, WorkItem, WorkResult


@dataclass(frozen=True)
class ResolutionContext:
    repository: Path
    models: JsonObject
    datasets: JsonObject
    defaults: JsonObject
    code_revision: str


@dataclass(frozen=True)
class ResolvedStageDefinition:
    settings: JsonObject
    semantic_settings: JsonObject
    execution_settings: JsonObject
    artifact_schema_revision: str
    resource_key: str


@dataclass(frozen=True)
class StageExecutionContext:
    repository: Path
    workspace: Path
    stage_name: str
    run_identity: str
    attempt_identity: str
    settings: JsonObject
    semantic_settings: JsonObject
    execution_settings: JsonObject
    inputs: tuple[ArtifactRef, ...]
    runtime: Any


class PreparedStageExecution(Protocol):
    def work_items(self) -> Iterable[WorkItem]: ...

    def execute(self, item: WorkItem) -> WorkResult: ...

    def summarize(self, records: Iterable[JsonObject]) -> JsonObject: ...

    def close(self) -> None: ...


class StageAdapter(Protocol):
    kind: str
    revision: str

    def resolve(
        self,
        settings: JsonObject,
        context: ResolutionContext,
    ) -> ResolvedStageDefinition: ...

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition: ...

    def prepare(
        self,
        context: StageExecutionContext,
    ) -> PreparedStageExecution: ...
