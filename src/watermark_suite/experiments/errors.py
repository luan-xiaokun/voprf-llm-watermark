class ExperimentRunError(Exception):
    """Base error raised by the Experiment Run module."""


class PlanError(ExperimentRunError):
    """The source or resolved Experiment Plan is invalid."""


class PlanSyntaxError(PlanError):
    """The Plan file cannot be parsed."""


class PlanValidationError(PlanError):
    """The Plan contains invalid or ambiguous settings."""

    def __init__(self, messages: list[str] | str):
        self.messages = [messages] if isinstance(messages, str) else messages
        super().__init__("\n".join(self.messages))


class SweepExpansionError(PlanError):
    """A Sweep is invalid or expands beyond the configured safety limit."""


class DependencyCycleError(PlanError):
    """Stage dependencies in an Experiment Plan contain a cycle."""


class ResolutionError(ExperimentRunError):
    """A revision, input, or setting cannot be resolved exactly."""


class DirtyImplementationError(ResolutionError):
    """Execution code differs from the code revision in the resolved Plan."""


class ArtifactError(ExperimentRunError):
    """An Artifact is unavailable, incompatible, or corrupt."""


class ArtifactIntegrityError(ArtifactError):
    """Artifact contents do not match their manifest."""


class ArtifactFinalizationError(ArtifactError):
    """A draft cannot be published as an immutable Artifact."""


class AttemptError(ExperimentRunError):
    """An Attempt cannot be created, resumed, or finalized."""


class AttemptConflictError(AttemptError):
    """Attempt state conflicts with the requested action."""


class CheckpointCorruptionError(AttemptError):
    """Attempt checkpoints are missing, duplicated, or corrupt."""


class StageExecutionError(ExperimentRunError):
    """A stage failed while executing a work item."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        run_identity: str,
        attempt_identity: str,
        work_identity: str | None = None,
        resumable: bool = True,
    ) -> None:
        self.stage = stage
        self.run_identity = run_identity
        self.attempt_identity = attempt_identity
        self.work_identity = work_identity
        self.resumable = resumable
        super().__init__(message)
