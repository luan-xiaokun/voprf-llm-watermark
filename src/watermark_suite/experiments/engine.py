from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from collections.abc import Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

from .adapters import (
    ResolvedStageDefinition,
    StageAdapter,
    StageExecutionContext,
)
from .errors import (
    ArtifactFinalizationError,
    ArtifactIntegrityError,
    AttemptConflictError,
    CheckpointCorruptionError,
    DirtyImplementationError,
    StageExecutionError,
)
from .identity import canonical_json, identity_for, sha256_file
from .models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    AttemptState,
    ExecutionReport,
    JsonObject,
    PlanErrors,
    PlanStatus,
    PreparedStage,
    ResolvedExperimentPlan,
    RunIdentity,
    WorkResult,
)
from .plan import code_revision, resolve_plan
from .plan_graph import (
    Continuation,
    ResolvedPlanGraph,
    RunOutcome,
)
from .workspace import ExperimentWorkspace


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_chunk(path: Path, result: WorkResult) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in result.records:
            output.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def _safe_artifact_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} for part in path.parts)
    ):
        raise ArtifactFinalizationError(
            f"WorkResult file has unsafe Artifact path {value!r}"
        )
    return path


def _read_jsonlines(path: Path) -> Iterable[JsonObject]:
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise CheckpointCorruptionError(
                    f"{path}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(value, dict):
                raise CheckpointCorruptionError(
                    f"{path}:{line_number}: record must be an object"
                )
            yield value


def _attempt_provenance(stage: PreparedStage) -> JsonObject:
    compatibility: JsonObject = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device": stage.execution_settings.get("device"),
        "dtype": stage.execution_settings.get("dtype"),
        "execution_revision": stage.adapter_revision,
    }
    resume_compatibility = stage.execution_settings.get("resume_compatibility")
    if resume_compatibility:
        compatibility["resume_compatibility"] = resume_compatibility
    try:
        import torch

        compatibility["torch"] = torch.__version__
        compatibility["cuda"] = torch.version.cuda
        configured_device = compatibility["device"]
        compatibility["actual_device"] = (
            "cuda"
            if configured_device == "auto" and torch.cuda.is_available()
            else (
                "cpu"
                if configured_device == "auto"
                else configured_device
            )
        )
        compatibility["gpu"] = (
            torch.cuda.get_device_name(0)
            if compatibility["actual_device"] == "cuda"
            else None
        )
    except ImportError:
        compatibility["torch"] = None
        compatibility["cuda"] = None
        compatibility["actual_device"] = compatibility["device"]
        compatibility["gpu"] = None
    for package in ("transformers", "datasets"):
        try:
            module = __import__(package)
            compatibility[package] = module.__version__
        except ImportError:
            compatibility[package] = None
    return {
        "compatibility": compatibility,
        "host": socket.gethostname(),
        "executable": sys.executable,
    }


class ExperimentRuns:
    """Deep interface for resolving, executing, and inspecting Experiment Runs."""

    def __init__(
        self,
        *,
        repository: Path,
        workspace: Path,
        stage_adapters: Mapping[str, StageAdapter] | None = None,
        enforce_clean: bool = True,
        runtime: Any = None,
    ) -> None:
        self.repository = repository.expanduser().resolve()
        self.workspace = ExperimentWorkspace(workspace)
        if stage_adapters is None:
            from .stages import default_stage_adapters

            stage_adapters = default_stage_adapters()
        self.stage_adapters = dict(stage_adapters)
        self.enforce_clean = enforce_clean
        if runtime is None:
            from .runtime import LocalModelRuntime

            runtime = LocalModelRuntime()
        self.runtime = runtime

    @classmethod
    def local(
        cls,
        *,
        repository: Path | None = None,
        workspace: Path | None = None,
        **kwargs: Any,
    ) -> "ExperimentRuns":
        selected_repository = (repository or Path.cwd()).resolve()
        selected_workspace = (
            workspace
            if workspace is not None
            else selected_repository / "output" / "experiments"
        )
        return cls(
            repository=selected_repository,
            workspace=selected_workspace,
            **kwargs,
        )

    def resolve(
        self, plan: str | Path, *, max_runs: int = 10_000
    ) -> ResolvedExperimentPlan:
        resolved = resolve_plan(
            Path(plan),
            repository=self.repository,
            adapters=self.stage_adapters,
            max_runs=max_runs,
        )
        self.workspace.register_plan(resolved)
        return resolved

    def _load_plan(
        self, plan: str | Path | ResolvedExperimentPlan
    ) -> ResolvedExperimentPlan:
        if isinstance(plan, ResolvedExperimentPlan):
            return plan
        candidate = Path(plan)
        if candidate.exists():
            return self.resolve(candidate)
        return self.workspace.load_plan(str(plan))

    def _check_implementation(self, plan: ResolvedExperimentPlan) -> None:
        revision, dirty = code_revision(self.repository)
        problems = []
        if revision != plan.code_revision:
            problems.append(
                f"resolved at {plan.code_revision}, current revision is {revision}"
            )
        if dirty:
            problems.append(
                "execution implementation is dirty:\n  " + "\n  ".join(dirty)
            )
        if problems and self.enforce_clean:
            raise DirtyImplementationError("\n".join(problems))

    @staticmethod
    def _definition(stage: PreparedStage) -> ResolvedStageDefinition:
        return ResolvedStageDefinition(
            settings=stage.settings,
            semantic_settings=stage.semantic_settings,
            execution_settings=stage.execution_settings,
            artifact_schema_revision=stage.artifact_schema_revision,
            resource_key=stage.resource_key,
        )

    def _run_identity(
        self,
        plan: ResolvedExperimentPlan,
        stage: PreparedStage,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> tuple[RunIdentity, str]:
        document = {
            "identity_schema": 3,
            "kind": stage.kind,
            "adapter_revision": stage.adapter_revision,
            "artifact_schema_revision": definition.artifact_schema_revision,
            "semantic_settings": definition.semantic_settings,
            "runtime_settings": {
                key: definition.execution_settings[key]
                for key in ("device", "dtype")
                if key in definition.execution_settings
            },
            "source_artifacts": [item.identity.value for item in inputs],
        }
        serialized = canonical_json(document)
        return (
            RunIdentity(identity_for(document, prefix="run")),
            serialized,
        )

    def _select_attempt(
        self,
        run: RunIdentity,
        stage: PreparedStage,
        *,
        force_new: bool,
    ) -> tuple[AttemptIdentity, bool]:
        provenance = _attempt_provenance(stage)
        if not force_new:
            existing = self.workspace.find_resumable_attempt(run)
            if existing is not None:
                previous = self.workspace.attempt_provenance(existing)
                if previous.get("compatibility") == provenance["compatibility"]:
                    return existing, True
        return self.workspace.create_attempt(run, provenance), False

    def _validate_checkpoints(
        self, attempt: AttemptIdentity
    ) -> list[JsonObject]:
        chunks = self.workspace.checkpoint_chunks(attempt)
        seen_ordinals: set[int] = set()
        for chunk in chunks:
            path = self.workspace.root / chunk["chunk_path"]
            if chunk["ordinal"] in seen_ordinals:
                raise CheckpointCorruptionError(
                    f"Attempt {attempt.value} contains a duplicate ordinal"
                )
            seen_ordinals.add(chunk["ordinal"])
            if not path.is_file() or sha256_file(path) != chunk["chunk_sha256"]:
                raise CheckpointCorruptionError(
                    f"Attempt {attempt.value} has a corrupt chunk {path}"
                )
            actual_count = sum(1 for _ in _read_jsonlines(path))
            if actual_count != chunk["record_count"]:
                raise CheckpointCorruptionError(
                    f"Attempt {attempt.value} chunk {path} has "
                    f"{actual_count} records, expected {chunk['record_count']}"
                )
            for artifact_path, file_info in chunk.get("files", {}).items():
                _safe_artifact_path(artifact_path)
                if not isinstance(file_info, dict):
                    raise CheckpointCorruptionError(
                        f"Attempt {attempt.value} contains invalid file metadata"
                    )
                relative_path = file_info.get("checkpoint_path")
                expected_digest = file_info.get("sha256")
                if not isinstance(relative_path, str) or not isinstance(
                    expected_digest, str
                ):
                    raise CheckpointCorruptionError(
                        f"Attempt {attempt.value} contains invalid file metadata"
                    )
                file_path = self.workspace.root / relative_path
                if (
                    not file_path.is_file()
                    or sha256_file(file_path) != expected_digest
                ):
                    raise CheckpointCorruptionError(
                        f"Attempt {attempt.value} has a corrupt file {file_path}"
                    )
        return chunks

    def _checkpoint_result(
        self,
        *,
        attempt: AttemptIdentity,
        item: Any,
        result: WorkResult,
    ) -> None:
        chunk_stem = f"{item.ordinal:08d}-{item.identity}"
        chunk_path = (
            self.workspace.attempt_directory(attempt)
            / "chunks"
            / f"{chunk_stem}.jsonl"
        )
        _write_chunk(chunk_path, result)
        checkpoint_files: JsonObject = {}
        file_root = (
            self.workspace.attempt_directory(attempt)
            / "chunk-files"
            / chunk_stem
        )
        seen_paths: set[str] = set()
        attempt_directory = self.workspace.attempt_directory(attempt).resolve()
        for produced in result.files:
            logical = str(_safe_artifact_path(produced.artifact_path))
            if logical in seen_paths:
                raise ArtifactFinalizationError(
                    f"WorkItem {item.identity!r} emitted duplicate file {logical!r}"
                )
            seen_paths.add(logical)
            source = produced.source_path.resolve()
            if not source.is_file():
                raise ArtifactFinalizationError(
                    f"WorkItem {item.identity!r} did not produce {source}"
                )
            try:
                source.relative_to(attempt_directory)
            except ValueError as error:
                raise ArtifactFinalizationError(
                    f"WorkItem {item.identity!r} produced a file outside its "
                    f"Attempt directory: {source}"
                ) from error
            destination = file_root / logical
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            os.replace(source, temporary)
            with temporary.open("rb") as copied:
                os.fsync(copied.fileno())
            os.replace(temporary, destination)
            checkpoint_files[logical] = {
                "checkpoint_path": str(destination.relative_to(self.workspace.root)),
                "sha256": sha256_file(destination),
            }
        self.workspace.commit_checkpoint(
            attempt=attempt,
            work_identity=item.identity,
            ordinal=item.ordinal,
            chunk_path=chunk_path,
            record_count=len(result.records),
            files=checkpoint_files,
        )

    def _finalize(
        self,
        *,
        plan: ResolvedExperimentPlan,
        stage: PreparedStage,
        definition: ResolvedStageDefinition,
        run: RunIdentity,
        attempt: AttemptIdentity,
        inputs: tuple[ArtifactRef, ...],
        execution: Any,
    ) -> ArtifactRef:
        existing = self.workspace.artifact_for_attempt(attempt)
        if existing is not None:
            return existing
        self.workspace.set_attempt_state(attempt, AttemptState.FINALIZING)
        chunks = self._validate_checkpoints(attempt)
        attempt_dir = self.workspace.attempt_directory(attempt)
        draft = attempt_dir / "artifact-draft"
        draft.mkdir(parents=True, exist_ok=True)
        records_path = draft / "records.jsonl"
        temporary_records = records_path.with_suffix(".jsonl.tmp")
        record_count = 0
        seen_samples: set[str] = set()
        with temporary_records.open("w", encoding="utf-8") as output:
            for chunk in chunks:
                chunk_path = self.workspace.root / chunk["chunk_path"]
                for record in _read_jsonlines(chunk_path):
                    sample_id = record.get("sample_id")
                    if not isinstance(sample_id, str) or not sample_id:
                        raise ArtifactFinalizationError(
                            f"{stage.instance_name} emitted a record without "
                            "a stable sample_id"
                        )
                    if sample_id in seen_samples:
                        raise ArtifactFinalizationError(
                            f"{stage.instance_name} emitted duplicate sample_id "
                            f"{sample_id!r}"
                        )
                    seen_samples.add(sample_id)
                    output.write(
                        json.dumps(
                            record, ensure_ascii=False, sort_keys=True
                        )
                        + "\n"
                    )
                    record_count += 1
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_records, records_path)

        attached_digests: dict[str, str] = {}
        for chunk in chunks:
            for artifact_path, file_info in chunk.get("files", {}).items():
                if artifact_path in attached_digests:
                    raise ArtifactFinalizationError(
                        f"{stage.instance_name} emitted duplicate Artifact file "
                        f"{artifact_path!r}"
                    )
                source = self.workspace.root / file_info["checkpoint_path"]
                destination = draft / str(_safe_artifact_path(artifact_path))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                attached_digests[artifact_path] = sha256_file(destination)

        records = tuple(_read_jsonlines(records_path))
        summary = execution.summarize(records)
        summary_path = draft / "summary.json"
        _atomic_json(summary_path, summary)
        content_digests = {
            "records.jsonl": sha256_file(records_path),
            "summary.json": sha256_file(summary_path),
            **attached_digests,
        }
        identity_document = {
            "identity_schema": 1,
            "run_identity": run.value,
            "attempt_identity": attempt.value,
            "artifact_schema_revision": definition.artifact_schema_revision,
            "content_digests": content_digests,
        }
        artifact_identity = ArtifactIdentity(
            identity_for(identity_document, prefix="artifact")
        )
        manifest: JsonObject = {
            **identity_document,
            "artifact_identity": artifact_identity.value,
            "stage_name": stage.stage_name,
            "instance_name": stage.instance_name,
            "kind": stage.kind,
            "source_artifacts": [item.identity.value for item in inputs],
            "code_revision": plan.code_revision,
            "semantic_settings": definition.semantic_settings,
            "execution_settings": definition.execution_settings,
            "record_count": record_count,
            "files": content_digests,
        }
        _atomic_json(draft / "manifest.json", manifest)
        destination = self.workspace.artifacts_dir / artifact_identity.value
        if destination.exists():
            existing_manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            if existing_manifest != manifest:
                raise ArtifactFinalizationError(
                    f"Artifact path collision at {destination}"
                )
        else:
            os.replace(draft, destination)
        return ArtifactRef(
            identity=artifact_identity,
            path=destination,
            schema_revision=definition.artifact_schema_revision,
            run_identity=run,
            attempt_identity=attempt,
            source_artifacts=tuple(item.identity for item in inputs),
            manifest=manifest,
        )

    def _execute_stage(
        self,
        *,
        plan: ResolvedExperimentPlan,
        stage: PreparedStage,
        inputs: tuple[ArtifactRef, ...],
        force_new: bool,
    ) -> tuple[ArtifactRef, str, str]:
        adapter = self.stage_adapters[stage.kind]
        definition = adapter.bind_inputs(self._definition(stage), inputs)
        run, run_document = self._run_identity(
            plan, stage, definition, inputs
        )
        self.workspace.register_run(
            run,
            stage_name=stage.stage_name,
            stage_kind=stage.kind,
            canonical_json=run_document,
        )
        self.workspace.bind_plan_run(
            plan_digest=plan.plan_digest,
            instance_name=stage.instance_name,
            run_identity=run,
        )
        with self.workspace.run_lease(run):
            return self._execute_stage_with_run_lease(
                plan=plan,
                stage=stage,
                definition=definition,
                inputs=inputs,
                force_new=force_new,
                adapter=adapter,
                run=run,
            )

    def _execute_stage_with_run_lease(
        self,
        *,
        plan: ResolvedExperimentPlan,
        stage: PreparedStage,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
        force_new: bool,
        adapter: StageAdapter,
        run: RunIdentity,
    ) -> tuple[ArtifactRef, str, str]:
        if not force_new:
            canonical = self.workspace.canonical_artifact(run)
            if canonical is not None:
                self.workspace.bind_plan_run(
                    plan_digest=plan.plan_digest,
                    instance_name=stage.instance_name,
                    run_identity=run,
                    artifact_identity=canonical.identity,
                )
                return canonical, "reused", ""

        attempt, resumed = self._select_attempt(
            run, stage, force_new=force_new
        )
        with self.workspace.attempt_lease(attempt):
            return self._execute_attempt(
                plan=plan,
                stage=stage,
                definition=definition,
                inputs=inputs,
                run=run,
                attempt=attempt,
                resumed=resumed,
                adapter=adapter,
            )

    def _execute_attempt(
        self,
        *,
        plan: ResolvedExperimentPlan,
        stage: PreparedStage,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
        run: RunIdentity,
        attempt: AttemptIdentity,
        resumed: bool,
        adapter: StageAdapter,
    ) -> tuple[ArtifactRef, str, str]:
        execution_context = StageExecutionContext(
            repository=self.repository,
            workspace=self.workspace.root,
            stage_name=stage.instance_name,
            run_identity=run.value,
            attempt_identity=attempt.value,
            settings=definition.settings,
            semantic_settings=definition.semantic_settings,
            execution_settings=definition.execution_settings,
            inputs=inputs,
            runtime=self.runtime,
        )
        segment_start = time.perf_counter()
        runtime_recorded = False
        execution = None
        try:
            self.workspace.set_attempt_state(attempt, AttemptState.RUNNING)
            execution = adapter.prepare(execution_context)
            completed = self.workspace.completed_work(attempt)
            self._validate_checkpoints(attempt)
            seen_work: set[str] = set()
            seen_ordinals: set[int] = set()

            def pending_items() -> Iterable[Any]:
                for item in execution.work_items():
                    if item.identity in seen_work:
                        raise StageExecutionError(
                            "stage emitted duplicate work identities",
                            stage=stage.instance_name,
                            run_identity=run.value,
                            attempt_identity=attempt.value,
                        )
                    if item.ordinal in seen_ordinals:
                        raise StageExecutionError(
                            "stage emitted duplicate work ordinals",
                            stage=stage.instance_name,
                            run_identity=run.value,
                            attempt_identity=attempt.value,
                        )
                    seen_work.add(item.identity)
                    seen_ordinals.add(item.ordinal)
                    if item.identity not in completed:
                        yield item

            def execute_item(item: Any) -> WorkResult:
                try:
                    return execution.execute(item)
                except Exception as error:
                    raise StageExecutionError(
                        str(error),
                        stage=stage.instance_name,
                        run_identity=run.value,
                        attempt_identity=attempt.value,
                        work_identity=item.identity,
                    ) from error

            worker_count = int(
                definition.execution_settings.get("worker_count", 1)
            )
            max_in_flight = int(
                definition.execution_settings.get(
                    "max_in_flight", max(1, worker_count * 2)
                )
            )
            if worker_count <= 0 or max_in_flight < worker_count:
                raise StageExecutionError(
                    "worker_count must be positive and max_in_flight must "
                    "be at least worker_count",
                    stage=stage.instance_name,
                    run_identity=run.value,
                    attempt_identity=attempt.value,
                )
            items = iter(pending_items())
            if worker_count == 1:
                for item in items:
                    self._checkpoint_result(
                        attempt=attempt,
                        item=item,
                        result=execute_item(item),
                    )
            else:
                executor = ThreadPoolExecutor(max_workers=worker_count)
                futures: dict[Future[WorkResult], Any] = {}
                exhausted = False
                try:
                    while futures or not exhausted:
                        while not exhausted and len(futures) < max_in_flight:
                            try:
                                item = next(items)
                            except StopIteration:
                                exhausted = True
                                break
                            futures[executor.submit(execute_item, item)] = item
                        if not futures:
                            continue
                        finished, _ = wait(
                            tuple(futures), return_when=FIRST_COMPLETED
                        )
                        for future in finished:
                            item = futures.pop(future)
                            self._checkpoint_result(
                                attempt=attempt,
                                item=item,
                                result=future.result(),
                            )
                finally:
                    for future in futures:
                        future.cancel()
                    executor.shutdown(wait=True, cancel_futures=True)
            artifact = self._finalize(
                plan=plan,
                stage=stage,
                definition=definition,
                run=run,
                attempt=attempt,
                inputs=inputs,
                execution=execution,
            )
            self.workspace.add_attempt_runtime(
                attempt, time.perf_counter() - segment_start
            )
            runtime_recorded = True
            self.workspace.register_artifact(
                artifact,
                plan_digest=plan.plan_digest,
                instance_name=stage.instance_name,
            )
            return artifact, "resumed" if resumed else "created", attempt.value
        except Exception as error:
            if not runtime_recorded:
                self.workspace.add_attempt_runtime(
                    attempt, time.perf_counter() - segment_start
                )
            self.workspace.set_attempt_state(
                attempt,
                AttemptState.FAILED,
                error={
                    "type": type(error).__name__,
                    "message": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            if execution is not None:
                execution.close()

    def execute(
        self,
        plan: str | Path | ResolvedExperimentPlan,
        *,
        new_attempts: Iterable[str] = (),
        keep_going: bool = False,
    ) -> ExecutionReport:
        resolved = self._load_plan(plan)
        self.workspace.register_plan(resolved)
        self._check_implementation(resolved)
        self.workspace.reset_plan_stage_states(resolved.plan_digest)
        force_new = set(new_attempts)
        unknown = force_new - {stage.instance_name for stage in resolved.stages}
        if unknown:
            raise KeyError(
                "unknown stage instances for --new-attempt: "
                + ", ".join(sorted(unknown))
            )

        graph = ResolvedPlanGraph(resolved)
        graph_state = graph.start()
        artifacts: dict[str, ArtifactRef] = {}
        failed: set[str] = set()
        blocked: set[str] = set()
        skipped: set[str] = set()
        reused: list[str] = []
        resumed: list[str] = []
        created: list[str] = []
        finalized: list[str] = []
        loaded_resource: str | None = None

        while (selected := graph.next(graph_state)) is not None:
            stage = selected.prepared
            inputs = tuple(
                artifacts[name] for name in stage.input_instances
            )
            try:
                if (
                    loaded_resource is not None
                    and stage.resource_key != loaded_resource
                    and hasattr(self.runtime, "release")
                ):
                    self.runtime.release()
                    loaded_resource = None
                artifact, disposition, attempt = self._execute_stage(
                    plan=resolved,
                    stage=stage,
                    inputs=inputs,
                    force_new=stage.instance_name in force_new,
                )
                artifacts[stage.instance_name] = artifact
                if disposition == "reused":
                    reused.append(artifact.identity.value)
                else:
                    finalized.append(artifact.identity.value)
                    if disposition == "resumed":
                        resumed.append(attempt)
                    else:
                        created.append(attempt)
                    loaded_resource = stage.resource_key
                transition = graph.record(
                    graph_state,
                    selected,
                    RunOutcome.SUCCEEDED,
                )
            except (ArtifactIntegrityError, AttemptConflictError):
                raise
            except Exception:
                if hasattr(self.runtime, "release"):
                    self.runtime.release()
                loaded_resource = None
                transition = graph.record(
                    graph_state,
                    selected,
                    RunOutcome.FAILED,
                    continuation=(
                        Continuation.CONTINUE
                        if keep_going
                        else Continuation.STOP
                    ),
                )

            classified = transition.classified
            failed.update(classified.failed)
            blocked.update(classified.blocked)
            skipped.update(classified.skipped)
            for instance_name in classified.blocked:
                self.workspace.set_plan_stage_state(
                    resolved.plan_digest,
                    instance_name,
                    "blocked",
                )
            for instance_name in classified.skipped:
                self.workspace.set_plan_stage_state(
                    resolved.plan_digest,
                    instance_name,
                    "skipped",
                )
            graph_state = transition.state

        return ExecutionReport(
            plan_digest=resolved.plan_digest,
            reused_artifacts=tuple(reused),
            resumed_attempts=tuple(resumed),
            created_attempts=tuple(created),
            finalized_artifacts=tuple(finalized),
            failed_stages=tuple(sorted(failed)),
            blocked_stages=tuple(sorted(blocked)),
            skipped_stages=tuple(sorted(skipped)),
        )

    def status(
        self, plan: str | Path | ResolvedExperimentPlan
    ) -> PlanStatus:
        resolved = self._load_plan(plan)
        status = self.workspace.status(resolved.plan_digest)
        stage_states = self.workspace.plan_stage_states(
            resolved.plan_digest
        )
        by_instance: dict[str, list[JsonObject]] = {}
        for run in status.runs:
            by_instance.setdefault(run["instance_name"], []).append(run)
        runs: list[JsonObject] = []
        for stage in resolved.stages:
            bound = by_instance.get(stage.instance_name)
            if bound:
                runs.extend(bound)
                continue
            runs.append(
                {
                    "instance_name": stage.instance_name,
                    "stage_name": stage.stage_name,
                    "stage_kind": stage.kind,
                    "run_identity": None,
                    "canonical_artifact_identity": None,
                    "state": stage_states.get(
                        stage.instance_name, "pending"
                    ),
                }
            )
        return PlanStatus(
            plan_digest=status.plan_digest,
            runs=tuple(runs),
            attempts=status.attempts,
            artifacts=status.artifacts,
        )

    def errors(
        self,
        plan: str | Path | ResolvedExperimentPlan,
        *,
        stage: str | None = None,
        attempt: str | None = None,
    ) -> PlanErrors:
        resolved = self._load_plan(plan)
        if stage is not None and stage not in {
            item.instance_name for item in resolved.stages
        }:
            raise KeyError(f"unknown stage instance {stage!r}")
        return PlanErrors(
            plan_digest=resolved.plan_digest,
            attempts=self.workspace.errors(
                resolved.plan_digest,
                instance_name=stage,
                attempt_identity=attempt,
            ),
        )
