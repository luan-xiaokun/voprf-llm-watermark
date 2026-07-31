from __future__ import annotations

import itertools
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .adapters import ResolutionContext, StageAdapter
from .errors import (
    DependencyCycleError,
    PlanSyntaxError,
    PlanValidationError,
    SweepExpansionError,
)
from .identity import identity_for
from .models import JsonObject, PreparedStage, ResolvedExperimentPlan
from .plan_graph import ResolvedPlanGraph


_TOP_LEVEL_FIELDS = {
    "version",
    "name",
    "models",
    "datasets",
    "defaults",
    "stages",
}
_STAGE_CONTROL_FIELDS = {"name", "kind", "input", "inputs", "sweep"}
_IMPLEMENTATION_PATHS = (
    "src/watermark_suite",
    "voprf-py",
    "hatch_build.py",
    "pyproject.toml",
)


def _git_output(repository: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def code_revision(repository: Path) -> tuple[str, tuple[str, ...]]:
    try:
        revision = _git_output(repository, "rev-parse", "HEAD")
        dirty_output = _git_output(
            repository,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            *_IMPLEMENTATION_PATHS,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unversioned", ()
    dirty = tuple(line for line in dirty_output.splitlines() if line)
    return revision, dirty


def load_plan_source(path: Path) -> JsonObject:
    try:
        with path.open("r", encoding="utf-8") as source:
            value = yaml.safe_load(source)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = (
            f"{path}:{mark.line + 1}:{mark.column + 1}"
            if mark is not None
            else str(path)
        )
        raise PlanSyntaxError(f"{location}: {error}") from error
    except OSError as error:
        raise PlanSyntaxError(f"cannot read Experiment Plan {path}: {error}") from error
    if not isinstance(value, dict):
        raise PlanValidationError("Experiment Plan root must be a mapping")
    return value


def _validate_source(source: JsonObject, adapters: Mapping[str, StageAdapter]) -> None:
    errors: list[str] = []
    unknown = sorted(set(source) - _TOP_LEVEL_FIELDS)
    if unknown:
        errors.append(f"unknown top-level fields: {', '.join(unknown)}")
    if source.get("version") != 1:
        errors.append("version must be 1")
    if not isinstance(source.get("name"), str) or not source["name"].strip():
        errors.append("name must be a non-empty string")
    for field in ("models", "datasets", "defaults"):
        if field in source and not isinstance(source[field], dict):
            errors.append(f"{field} must be a mapping")
    defaults = source.get("defaults", {})
    accepted_default_fields = {
        field
        for adapter in adapters.values()
        for field in (getattr(adapter, "accepted_settings", None) or ())
    }
    if isinstance(defaults, dict):
        unknown_defaults = sorted(set(defaults) - accepted_default_fields)
        if unknown_defaults:
            errors.append(
                "defaults contains settings accepted by no stage kind: "
                + ", ".join(unknown_defaults)
            )
    stages = source.get("stages")
    if not isinstance(stages, dict) or not stages:
        errors.append("stages must be a non-empty mapping")
    else:
        for stage_name, stage in stages.items():
            if not isinstance(stage_name, str) or not stage_name:
                errors.append("stage names must be non-empty strings")
                continue
            if not isinstance(stage, dict):
                errors.append(f"stage {stage_name!r} must be a mapping")
                continue
            kind = stage.get("kind")
            if kind not in adapters:
                errors.append(f"stage {stage_name!r} has unknown kind {kind!r}")
            if "input" in stage and "inputs" in stage:
                errors.append(
                    f"stage {stage_name!r} cannot declare both input and inputs"
                )
            if "input" in stage and not isinstance(stage["input"], str):
                errors.append(f"stage {stage_name!r} input must be a stage name")
            inputs = stage.get("inputs")
            if inputs is not None and (
                not isinstance(inputs, list)
                or not inputs
                or any(not isinstance(item, str) or not item for item in inputs)
                or len(inputs) != len(set(inputs))
            ):
                errors.append(
                    f"stage {stage_name!r} inputs must be a non-empty list "
                    "of unique stage names"
                )
            sweep = stage.get("sweep", {})
            if not isinstance(sweep, dict):
                errors.append(f"stage {stage_name!r} sweep must be a mapping")
            elif any(not isinstance(values, list) or not values for values in sweep.values()):
                errors.append(
                    f"stage {stage_name!r} sweep values must be non-empty lists"
                )
    if errors:
        raise PlanValidationError(errors)


def _topological_stage_names(stages: JsonObject) -> list[str]:
    order = {name: index for index, name in enumerate(stages)}
    dependencies: dict[str, set[str]] = {name: set() for name in stages}
    for name, stage in stages.items():
        input_names = (
            [stage["input"]]
            if "input" in stage
            else stage.get("inputs", [])
        )
        for input_name in input_names:
            if input_name not in stages:
                raise PlanValidationError(
                    f"stage {name!r} references unknown input stage {input_name!r}"
                )
            dependencies[name].add(input_name)

    ready = sorted(
        (name for name, deps in dependencies.items() if not deps),
        key=order.__getitem__,
    )
    result: list[str] = []
    while ready:
        name = ready.pop(0)
        result.append(name)
        for candidate, deps in dependencies.items():
            if name in deps:
                deps.remove(name)
                if not deps and candidate not in result and candidate not in ready:
                    ready.append(candidate)
                    ready.sort(key=order.__getitem__)
    if len(result) != len(stages):
        cycle = sorted(name for name, deps in dependencies.items() if deps)
        raise DependencyCycleError(
            f"Experiment Plan contains a stage dependency cycle: {cycle}"
        )
    return result


def _accepted_settings(adapter: StageAdapter) -> set[str] | None:
    fields = getattr(adapter, "accepted_settings", None)
    return set(fields) if fields is not None else None


def _merge_settings(
    defaults: JsonObject,
    stage: JsonObject,
    adapter: StageAdapter,
) -> tuple[JsonObject, JsonObject]:
    accepted = _accepted_settings(adapter)
    if accepted is None:
        selected_defaults = defaults
    else:
        selected_defaults = {
            key: value for key, value in defaults.items() if key in accepted
        }
    settings = deepcopy(selected_defaults)
    sources: JsonObject = {
        key: "plan.defaults" for key in selected_defaults
    }
    explicit = {
        key: deepcopy(value)
        for key, value in stage.items()
        if key not in _STAGE_CONTROL_FIELDS
    }
    if accepted is not None:
        unknown = sorted(set(explicit) - accepted)
        if unknown:
            raise PlanValidationError(
                f"stage {stage.get('name', '<unknown>')!r} has unknown settings: "
                f"{', '.join(unknown)}"
            )
    settings.update(explicit)
    sources.update({key: "stage" for key in explicit})
    return settings, sources


def _sweep_assignments(sweep: JsonObject) -> list[JsonObject]:
    if not sweep:
        return [{}]
    keys = list(sweep)
    values = [sweep[key] for key in keys]
    return [
        dict(zip(keys, combination))
        for combination in itertools.product(*values)
    ]


def resolve_plan(
    path: Path,
    *,
    repository: Path,
    adapters: Mapping[str, StageAdapter],
    max_runs: int = 10_000,
) -> ResolvedExperimentPlan:
    path = path.expanduser().resolve()
    repository = repository.resolve()
    source = load_plan_source(path)
    _validate_source(source, adapters)
    revision, dirty = code_revision(repository)
    models = deepcopy(source.get("models", {}))
    datasets = deepcopy(source.get("datasets", {}))
    defaults = deepcopy(source.get("defaults", {}))
    stages_source = source["stages"]
    context = ResolutionContext(
        repository=repository,
        models=models,
        datasets=datasets,
        defaults=defaults,
        code_revision=revision,
    )

    prepared_by_stage: dict[str, list[PreparedStage]] = {}
    prepared: list[PreparedStage] = []
    ordinal = 0
    for stage_name in _topological_stage_names(stages_source):
        stage_source = deepcopy(stages_source[stage_name])
        stage_source["name"] = stage_name
        kind = stage_source["kind"]
        adapter = adapters[kind]
        base_settings, base_sources = _merge_settings(
            defaults, stage_source, adapter
        )
        sweep = stage_source.get("sweep", {})
        accepted = _accepted_settings(adapter)
        if accepted is not None:
            unknown_sweep = sorted(set(sweep) - accepted)
            if unknown_sweep:
                raise PlanValidationError(
                    f"stage {stage_name!r} sweeps unknown settings: "
                    f"{', '.join(unknown_sweep)}"
                )
        assignments = _sweep_assignments(sweep)
        input_stage = stage_source.get("input")
        collected_stages = stage_source.get("inputs")
        if input_stage is not None:
            upstream_groups = [
                (upstream,)
                for upstream in prepared_by_stage[input_stage]
            ]
        elif collected_stages is not None:
            upstream_groups = [
                tuple(
                    upstream
                    for name in collected_stages
                    for upstream in prepared_by_stage[name]
                )
            ]
        else:
            upstream_groups = [()]
        stage_instances: list[PreparedStage] = []
        for upstreams in upstream_groups:
            for assignment in assignments:
                settings = deepcopy(base_settings)
                settings.update(deepcopy(assignment))
                setting_sources = deepcopy(base_sources)
                setting_sources.update(
                    {key: "sweep" for key in assignment}
                )
                definition = adapter.resolve(settings, context)
                expansion_identity = identity_for(
                    {
                        "stage": stage_name,
                        "settings": definition.settings,
                        "inputs": [
                            upstream.instance_name for upstream in upstreams
                        ],
                    },
                    prefix="instance",
                )
                expanded = len(assignments) * len(upstream_groups) > 1
                instance_name = (
                    f"{stage_name}[{expansion_identity[-10:]}]"
                    if expanded
                    else stage_name
                )
                if any(
                    existing.instance_name == instance_name
                    for existing in stage_instances
                ):
                    raise SweepExpansionError(
                        f"stage {stage_name!r} expands to duplicate Run "
                        f"instance {instance_name!r}"
                    )
                instance = PreparedStage(
                    stage_name=stage_name,
                    instance_name=instance_name,
                    kind=kind,
                    settings=definition.settings,
                    setting_sources=setting_sources,
                    semantic_settings=definition.semantic_settings,
                    execution_settings=definition.execution_settings,
                    adapter_revision=adapter.revision,
                    artifact_schema_revision=(
                        definition.artifact_schema_revision
                    ),
                    input_instances=tuple(
                        upstream.instance_name for upstream in upstreams
                    ),
                    ordinal=ordinal,
                    resource_key=definition.resource_key,
                )
                ordinal += 1
                prepared.append(instance)
                stage_instances.append(instance)
                if len(prepared) > max_runs:
                    raise SweepExpansionError(
                        f"Experiment Plan expands beyond max_runs={max_runs}"
                    )
        prepared_by_stage[stage_name] = stage_instances

    resolved_document = {
        "identity_schema": 1,
        "version": source["version"],
        "name": source["name"],
        "code_revision": revision,
        "stages": [stage.to_dict() for stage in prepared],
    }
    plan_digest = identity_for(resolved_document, prefix="plan")
    resolved = ResolvedExperimentPlan(
        version=source["version"],
        name=source["name"],
        source_path=path,
        plan_digest=plan_digest,
        code_revision=revision,
        implementation_dirty=dirty,
        stages=tuple(prepared),
        source=source,
    )
    ResolvedPlanGraph(resolved)
    return resolved
