from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


JsonObject = dict[str, Any]


class AttemptState(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FINALIZING = "finalizing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class RunIdentity:
    value: str


@dataclass(frozen=True)
class AttemptIdentity:
    value: str


@dataclass(frozen=True)
class ArtifactIdentity:
    value: str


@dataclass(frozen=True)
class ArtifactRef:
    identity: ArtifactIdentity
    path: Path
    schema_revision: str
    run_identity: RunIdentity
    attempt_identity: AttemptIdentity
    source_artifacts: tuple[ArtifactIdentity, ...] = ()
    manifest: JsonObject = field(default_factory=dict)


@dataclass(frozen=True)
class PreparedStage:
    stage_name: str
    instance_name: str
    kind: str
    settings: JsonObject
    setting_sources: JsonObject
    semantic_settings: JsonObject
    execution_settings: JsonObject
    adapter_revision: str
    artifact_schema_revision: str
    input_instances: tuple[str, ...]
    ordinal: int
    resource_key: str

    def to_dict(self) -> JsonObject:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: JsonObject) -> "PreparedStage":
        return cls(
            **{
                **value,
                "input_instances": tuple(value["input_instances"]),
            }
        )


@dataclass(frozen=True)
class ResolvedExperimentPlan:
    version: int
    name: str
    source_path: Path
    plan_digest: str
    code_revision: str
    implementation_dirty: tuple[str, ...]
    stages: tuple[PreparedStage, ...]
    source: JsonObject

    def to_dict(self) -> JsonObject:
        return {
            "version": self.version,
            "name": self.name,
            "source_path": str(self.source_path),
            "plan_digest": self.plan_digest,
            "code_revision": self.code_revision,
            "implementation_dirty": list(self.implementation_dirty),
            "stages": [stage.to_dict() for stage in self.stages],
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ResolvedExperimentPlan":
        return cls(
            version=value["version"],
            name=value["name"],
            source_path=Path(value["source_path"]),
            plan_digest=value["plan_digest"],
            code_revision=value["code_revision"],
            implementation_dirty=tuple(value.get("implementation_dirty", [])),
            stages=tuple(
                PreparedStage.from_dict(stage) for stage in value["stages"]
            ),
            source=value["source"],
        )


@dataclass(frozen=True)
class WorkItem:
    identity: str
    ordinal: int
    sample_identities: tuple[str, ...]
    payload: Any = None


@dataclass(frozen=True)
class WorkResult:
    records: tuple[JsonObject, ...]
    metrics: JsonObject = field(default_factory=dict)
    files: tuple["WorkFile", ...] = ()


@dataclass(frozen=True)
class WorkFile:
    """An Attempt-local file transferred into the finalized Artifact."""

    source_path: Path
    artifact_path: str


@dataclass(frozen=True)
class ExecutionReport:
    plan_digest: str
    reused_artifacts: tuple[str, ...]
    resumed_attempts: tuple[str, ...]
    created_attempts: tuple[str, ...]
    finalized_artifacts: tuple[str, ...]
    failed_stages: tuple[str, ...]
    blocked_stages: tuple[str, ...]
    skipped_stages: tuple[str, ...]

    @property
    def succeeded(self) -> bool:
        return not (
            self.failed_stages
            or self.blocked_stages
            or self.skipped_stages
        )

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class PlanStatus:
    plan_digest: str
    runs: tuple[JsonObject, ...]
    attempts: tuple[JsonObject, ...]
    artifacts: tuple[JsonObject, ...]

    def to_dict(self) -> JsonObject:
        return asdict(self)


@dataclass(frozen=True)
class PlanErrors:
    plan_digest: str
    attempts: tuple[JsonObject, ...]

    def to_dict(self) -> JsonObject:
        return asdict(self)
