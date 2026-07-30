from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from watermark_suite.experiments import ExperimentRuns
from watermark_suite.experiments.errors import (
    ArtifactIntegrityError,
    AttemptConflictError,
)
from watermark_suite.experiments.adapters import (
    ResolutionContext,
    ResolvedStageDefinition,
    StageExecutionContext,
)
from watermark_suite.experiments.identity import identity_for, sha256_file
from watermark_suite.experiments.models import (
    ArtifactRef,
    AttemptIdentity,
    AttemptState,
    JsonObject,
    WorkItem,
    WorkResult,
)


class FakeStageAdapter:
    kind = "fake"
    revision = "fake-v1"
    accepted_settings = {
        "value",
        "batch_size",
        "sample_num",
        "fail_on",
        "crash_on",
        "device",
        "dtype",
        "allow_download",
    }

    def __init__(self) -> None:
        self.crashed: set[tuple[str, int]] = set()
        self.failed: set[tuple[str, int]] = set()

    def resolve(
        self, settings: JsonObject, context: ResolutionContext
    ) -> ResolvedStageDefinition:
        del context
        semantic = {
            key: settings[key]
            for key in (
                "value",
                "batch_size",
                "sample_num",
                "fail_on",
                "crash_on",
            )
        }
        return ResolvedStageDefinition(
            settings=dict(settings),
            semantic_settings=semantic,
            execution_settings={
                "device": settings["device"],
                "dtype": settings["dtype"],
            },
            artifact_schema_revision="fake-records-v1",
            resource_key=f"fake:{settings['value']}",
        )

    def bind_inputs(
        self,
        definition: ResolvedStageDefinition,
        inputs: tuple[ArtifactRef, ...],
    ) -> ResolvedStageDefinition:
        semantic = {
            **definition.semantic_settings,
            "input_count": len(inputs),
        }
        return replace(definition, semantic_settings=semantic)

    def prepare(self, context: StageExecutionContext) -> "FakeExecution":
        return FakeExecution(context, self)


class FakeExecution:
    def __init__(
        self, context: StageExecutionContext, adapter: FakeStageAdapter
    ) -> None:
        self.context = context
        self.adapter = adapter

    def work_items(self) -> list[WorkItem]:
        settings = self.context.semantic_settings
        sample_ids = [
            f"sample:{index}" for index in range(settings["sample_num"])
        ]
        result = []
        batch_size = settings["batch_size"]
        for ordinal, start in enumerate(
            range(0, len(sample_ids), batch_size)
        ):
            selected = tuple(sample_ids[start : start + batch_size])
            result.append(
                WorkItem(
                    identity=identity_for(
                        {"ordinal": ordinal, "samples": selected},
                        prefix="work",
                    ),
                    ordinal=ordinal,
                    sample_identities=selected,
                    payload=selected,
                )
            )
        return result

    def execute(self, item: WorkItem) -> WorkResult:
        settings = self.context.semantic_settings
        failure_key = (self.context.run_identity, item.ordinal)
        if (
            settings["fail_on"] == item.ordinal
            and failure_key not in self.adapter.failed
        ):
            self.adapter.failed.add(failure_key)
            raise RuntimeError(f"failure at {item.ordinal}")
        crash_key = (self.context.run_identity, item.ordinal)
        if (
            settings["crash_on"] == item.ordinal
            and crash_key not in self.adapter.crashed
        ):
            self.adapter.crashed.add(crash_key)
            raise KeyboardInterrupt("simulated process loss")
        records = tuple(
            {
                "sample_id": sample_id,
                "value": settings["value"],
                "input_artifact": (
                    self.context.inputs[0].identity.value
                    if self.context.inputs
                    else None
                ),
            }
            for sample_id in item.sample_identities
        )
        return WorkResult(records=records)

    def summarize(self, records: list[JsonObject]) -> JsonObject:
        return {
            "sample_num": len(records),
            "value": self.context.semantic_settings["value"],
        }

    def close(self) -> None:
        pass


def write_plan(
    path: Path,
    stages: JsonObject,
    *,
    defaults: JsonObject | None = None,
) -> Path:
    value = {
        "version": 1,
        "name": "test-plan",
        "models": {},
        "datasets": {},
        "defaults": defaults
        or {
            "batch_size": 2,
            "sample_num": 4,
            "fail_on": None,
            "crash_on": None,
            "device": "cpu",
            "dtype": "float32",
            "allow_download": False,
        },
        "stages": stages,
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def experiment_runs(
    repository: Path,
    workspace: Path,
    adapter: FakeStageAdapter,
) -> ExperimentRuns:
    return ExperimentRuns.local(
        repository=repository,
        workspace=workspace,
        stage_adapters={"fake": adapter},
        enforce_clean=False,
        runtime=object(),
    )


def artifact_manifests(workspace: Path) -> list[JsonObject]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((workspace / "artifacts").glob("*/manifest.json"))
    ]


def test_resolve_materializes_defaults_sweep_and_setting_sources(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {
                "kind": "fake",
                "value": "base",
                "sweep": {"value": ["left", "right"]},
            },
            "derived": {
                "kind": "fake",
                "input": "root",
                "value": "derived",
            },
        },
    )
    runs = experiment_runs(tmp_path, tmp_path / "workspace", adapter)

    plan = runs.resolve(plan_path)

    assert len(plan.stages) == 4
    roots = [stage for stage in plan.stages if stage.stage_name == "root"]
    derived = [
        stage for stage in plan.stages if stage.stage_name == "derived"
    ]
    assert {stage.settings["value"] for stage in roots} == {"left", "right"}
    assert all(stage.setting_sources["value"] == "sweep" for stage in roots)
    assert all(
        stage.setting_sources["batch_size"] == "plan.defaults"
        for stage in plan.stages
    )
    assert {stage.input_instance for stage in derived} == {
        stage.instance_name for stage in roots
    }


def test_execute_reuses_canonical_and_preserves_lineage(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {"kind": "fake", "value": "root"},
            "derived": {
                "kind": "fake",
                "input": "root",
                "value": "derived",
            },
        },
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)

    first = runs.execute(plan_path)
    second = runs.execute(plan_path)

    assert first.succeeded
    assert len(first.finalized_artifacts) == 2
    assert len(second.reused_artifacts) == 2
    assert not second.finalized_artifacts
    manifests = artifact_manifests(workspace)
    root = next(item for item in manifests if item["instance_name"] == "root")
    derived = next(
        item for item in manifests if item["instance_name"] == "derived"
    )
    assert derived["source_artifacts"] == [root["artifact_identity"]]


def test_process_loss_resumes_same_attempt_without_rewriting_batch(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {
                "kind": "fake",
                "value": "resume",
                "crash_on": 1,
            }
        },
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)

    with pytest.raises(KeyboardInterrupt):
        runs.execute(plan_path)
    attempt_dirs = list((workspace / "attempts").iterdir())
    assert len(attempt_dirs) == 1
    attempt = AttemptIdentity(attempt_dirs[0].name)
    assert runs.workspace.attempt_state(attempt) == AttemptState.RUNNING
    first_chunk = next((attempt_dirs[0] / "chunks").glob("*.jsonl"))
    first_digest = sha256_file(first_chunk)

    resumed_runs = experiment_runs(tmp_path, workspace, adapter)
    assert resumed_runs.workspace.attempt_state(attempt) == AttemptState.RUNNING
    report = resumed_runs.execute(plan_path)

    assert report.succeeded
    assert report.resumed_attempts == (attempt_dirs[0].name,)
    assert sha256_file(first_chunk) == first_digest
    assert len(list((attempt_dirs[0] / "chunks").glob("*.jsonl"))) == 2


def test_caught_failure_resumes_same_attempt_and_keeps_checkpoint(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {
                "kind": "fake",
                "value": "resume-failed",
                "fail_on": 1,
            }
        },
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)

    failed_report = runs.execute(plan_path)
    attempt_dir = next((workspace / "attempts").iterdir())
    attempt = AttemptIdentity(attempt_dir.name)
    first_chunk = next((attempt_dir / "chunks").glob("*.jsonl"))
    first_digest = sha256_file(first_chunk)

    assert failed_report.failed_stages == ("root",)
    assert runs.workspace.attempt_state(attempt) == AttemptState.FAILED

    resumed_report = runs.execute(plan_path)

    assert resumed_report.succeeded
    assert resumed_report.resumed_attempts == (attempt.value,)
    assert sha256_file(first_chunk) == first_digest
    assert len(list((attempt_dir / "chunks").glob("*.jsonl"))) == 2


def test_attempt_lease_rejects_a_second_executor(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {"root": {"kind": "fake", "value": "leased"}},
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    plan = runs.resolve(plan_path)
    stage = plan.stages[0]
    definition = runs._definition(stage)
    run, run_document = runs._run_identity(plan, stage, definition, ())
    runs.workspace.register_run(
        run,
        stage_name=stage.stage_name,
        stage_kind=stage.kind,
        canonical_json=run_document,
    )
    attempt = runs.workspace.create_attempt(run, {"compatibility": {}})
    second = experiment_runs(tmp_path, workspace, adapter)

    with runs.workspace.attempt_lease(attempt):
        with pytest.raises(AttemptConflictError, match="already active"):
            with second.workspace.attempt_lease(attempt):
                pass


def test_run_lease_prevents_concurrent_attempt_selection(tmp_path):
    adapter = FakeStageAdapter()
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    second = experiment_runs(tmp_path, workspace, adapter)
    plan = runs.resolve(
        write_plan(
            tmp_path / "plan.yaml",
            {"root": {"kind": "fake", "value": "leased-run"}},
        )
    )
    stage = plan.stages[0]
    run = runs._run_identity(
        plan,
        stage,
        runs._definition(stage),
        (),
    )[0]

    with runs.workspace.run_lease(run):
        with pytest.raises(AttemptConflictError, match="already active"):
            with second.workspace.run_lease(run):
                pass


def test_status_lists_unstarted_plan_stages(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {"kind": "fake", "value": "root"},
            "derived": {
                "kind": "fake",
                "input": "root",
                "value": "derived",
            },
        },
    )
    runs = experiment_runs(tmp_path, tmp_path / "workspace", adapter)
    plan = runs.resolve(plan_path)

    status = runs.status(plan).to_dict()

    assert [
        (run["instance_name"], run["state"]) for run in status["runs"]
    ] == [("root", "pending"), ("derived", "pending")]
    assert all(run["run_identity"] is None for run in status["runs"])


def test_new_attempt_keeps_first_success_canonical(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {"root": {"kind": "fake", "value": "canonical"}},
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    first = runs.execute(plan_path)
    plan = runs.resolve(plan_path)
    first_manifest = artifact_manifests(workspace)[0]

    second = runs.execute(plan, new_attempts=["root"])

    assert second.succeeded
    assert len(artifact_manifests(workspace)) == 2
    status = runs.status(plan).to_dict()
    assert status["runs"][0]["canonical_artifact_identity"] == (
        first_manifest["artifact_identity"]
    )
    assert first.finalized_artifacts[0] != second.finalized_artifacts[0]


def test_keep_going_runs_independent_stage_after_failure(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "broken": {
                "kind": "fake",
                "value": "broken",
                "fail_on": 0,
            },
            "independent": {"kind": "fake", "value": "okay"},
            "blocked": {
                "kind": "fake",
                "input": "broken",
                "value": "blocked",
            },
        },
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)

    report = runs.execute(plan_path, keep_going=True)

    assert report.failed_stages == ("broken",)
    assert report.blocked_stages == ("blocked",)
    assert len(report.finalized_artifacts) == 1
    states = {
        item["instance_name"]: item["state"]
        for item in runs.status(plan_path).to_dict()["runs"]
    }
    assert states == {
        "broken": "failed",
        "independent": "succeeded",
        "blocked": "blocked",
    }


def test_fail_fast_distinguishes_blocked_from_skipped_stages(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "broken": {
                "kind": "fake",
                "value": "broken",
                "fail_on": 0,
            },
            "independent": {"kind": "fake", "value": "not-run"},
            "blocked": {
                "kind": "fake",
                "input": "broken",
                "value": "blocked",
            },
        },
    )
    runs = experiment_runs(tmp_path, tmp_path / "workspace", adapter)

    report = runs.execute(plan_path)
    states = {
        item["instance_name"]: item["state"]
        for item in runs.status(plan_path).to_dict()["runs"]
    }

    assert report.failed_stages == ("broken",)
    assert report.blocked_stages == ("blocked",)
    assert report.skipped_stages == ("independent",)
    assert states == {
        "broken": "failed",
        "independent": "skipped",
        "blocked": "blocked",
    }


def test_result_setting_changes_run_identity(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {"root": {"kind": "fake", "value": "one"}},
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    first = runs.execute(plan_path)
    write_plan(
        plan_path,
        {"root": {"kind": "fake", "value": "two"}},
    )
    second = runs.execute(plan_path)

    assert first.finalized_artifacts != second.finalized_artifacts
    assert len(artifact_manifests(workspace)) == 2


def test_runtime_setting_changes_run_identity(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {
            "root": {
                "kind": "fake",
                "value": "same",
                "dtype": "float32",
            }
        },
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    first = runs.execute(plan_path)
    write_plan(
        plan_path,
        {
            "root": {
                "kind": "fake",
                "value": "same",
                "dtype": "bfloat16",
            }
        },
    )
    second = runs.execute(plan_path)

    assert first.finalized_artifacts != second.finalized_artifacts
    assert len(artifact_manifests(workspace)) == 2


def test_canonical_reuse_rejects_corrupt_artifact(tmp_path):
    adapter = FakeStageAdapter()
    plan_path = write_plan(
        tmp_path / "plan.yaml",
        {"root": {"kind": "fake", "value": "integrity"}},
    )
    workspace = tmp_path / "workspace"
    runs = experiment_runs(tmp_path, workspace, adapter)
    runs.execute(plan_path)
    records_path = next(
        (workspace / "artifacts").glob("*/records.jsonl")
    )
    records_path.write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError):
        runs.execute(plan_path)
