from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from heapq import heappop, heappush

from .errors import (
    DependencyCycleError,
    PlanProtocolError,
    PlanValidationError,
)
from .identity import identity_for
from .models import PreparedStage, ResolvedExperimentPlan


class RunOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class Continuation(StrEnum):
    CONTINUE = "continue"
    STOP = "stop"


class _RunPhase(StrEnum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class PlanExecutionState:
    """Opaque immutable state for one Resolved Experiment Plan traversal."""

    _graph_key: str
    _generation: int
    _phases: tuple[_RunPhase, ...]


@dataclass(frozen=True)
class NextRun:
    prepared: PreparedStage
    _graph_key: str
    _generation: int
    _position: int


@dataclass(frozen=True)
class OutcomeDelta:
    succeeded: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    blocked: tuple[str, ...] = ()
    skipped: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlanTransition:
    state: PlanExecutionState
    classified: OutcomeDelta


class ResolvedPlanGraph:
    """Deterministic ordering and failure semantics for resolved Runs."""

    _ORDER_POLICY = "resource-affine-ordinal-v1"

    def __init__(self, plan: ResolvedExperimentPlan) -> None:
        self._plan_digest = plan.plan_digest
        stages = tuple(plan.stages)
        self._validate_stages(stages)
        ordered = self._ordered_stages(stages)
        self._execution_order = tuple(ordered)
        self._positions = {
            stage.instance_name: position
            for position, stage in enumerate(self._execution_order)
        }
        self._inputs = tuple(
            tuple(self._positions[name] for name in stage.input_instances)
            for stage in self._execution_order
        )
        dependents: list[list[int]] = [
            [] for _ in self._execution_order
        ]
        for position, inputs in enumerate(self._inputs):
            for input_position in inputs:
                dependents[input_position].append(position)
        self._dependents = tuple(
            tuple(items) for items in dependents
        )
        self._graph_key = identity_for(
            {
                "order_policy": self._ORDER_POLICY,
                "plan_digest": plan.plan_digest,
                "runs": [
                    {
                        "instance_name": stage.instance_name,
                        "inputs": list(stage.input_instances),
                        "ordinal": stage.ordinal,
                        "resource_key": stage.resource_key,
                    }
                    for stage in stages
                ],
            },
            prefix="plan-graph",
        )

    @property
    def execution_order(self) -> tuple[PreparedStage, ...]:
        return self._execution_order

    def start(self) -> PlanExecutionState:
        return PlanExecutionState(
            _graph_key=self._graph_key,
            _generation=0,
            _phases=tuple(
                _RunPhase.PENDING for _ in self._execution_order
            ),
        )

    def next(self, state: PlanExecutionState) -> NextRun | None:
        self._validate_state(state)
        for position, phase in enumerate(state._phases):
            if phase is not _RunPhase.PENDING:
                continue
            if all(
                state._phases[input_position] is _RunPhase.SUCCEEDED
                for input_position in self._inputs[position]
            ):
                return NextRun(
                    prepared=self._execution_order[position],
                    _graph_key=self._graph_key,
                    _generation=state._generation,
                    _position=position,
                )
        if _RunPhase.PENDING in state._phases:
            raise PlanProtocolError(
                "Plan execution state contains pending Runs but none is runnable"
            )
        return None

    def record(
        self,
        state: PlanExecutionState,
        selected: NextRun,
        outcome: RunOutcome,
        *,
        continuation: Continuation = Continuation.CONTINUE,
    ) -> PlanTransition:
        self._validate_state(state)
        expected = self.next(state)
        if expected is None:
            raise PlanProtocolError(
                "cannot record a Run outcome after Plan execution finished"
            )
        if (
            selected._graph_key != self._graph_key
            or selected._generation != state._generation
            or selected._position != expected._position
            or selected.prepared.instance_name
            != expected.prepared.instance_name
        ):
            raise PlanProtocolError(
                "selected Run is foreign, stale, or not the deterministic next Run"
            )
        if not isinstance(outcome, RunOutcome):
            raise PlanProtocolError(f"unsupported Run outcome {outcome!r}")
        if not isinstance(continuation, Continuation):
            raise PlanProtocolError(
                f"unsupported continuation policy {continuation!r}"
            )

        phases = list(state._phases)
        succeeded: list[str] = []
        failed: list[str] = []
        blocked: list[str] = []
        skipped: list[str] = []
        selected_name = expected.prepared.instance_name

        if outcome is RunOutcome.SUCCEEDED:
            phases[expected._position] = _RunPhase.SUCCEEDED
            succeeded.append(selected_name)
        else:
            phases[expected._position] = _RunPhase.FAILED
            failed.append(selected_name)
            descendants = self._descendants(expected._position)
            for position, stage in enumerate(self._execution_order):
                if (
                    position in descendants
                    and phases[position] is _RunPhase.PENDING
                ):
                    phases[position] = _RunPhase.BLOCKED
                    blocked.append(stage.instance_name)

        if continuation is Continuation.STOP:
            for position, stage in enumerate(self._execution_order):
                if phases[position] is _RunPhase.PENDING:
                    phases[position] = _RunPhase.SKIPPED
                    skipped.append(stage.instance_name)

        return PlanTransition(
            state=PlanExecutionState(
                _graph_key=self._graph_key,
                _generation=state._generation + 1,
                _phases=tuple(phases),
            ),
            classified=OutcomeDelta(
                succeeded=tuple(succeeded),
                failed=tuple(failed),
                blocked=tuple(blocked),
                skipped=tuple(skipped),
            ),
        )

    @staticmethod
    def _validate_stages(stages: tuple[PreparedStage, ...]) -> None:
        errors: list[str] = []
        if not stages:
            errors.append(
                "Resolved Experiment Plan must contain at least one Run"
            )
        names = [stage.instance_name for stage in stages]
        duplicate_names = sorted(
            name for name, count in Counter(names).items() if count > 1
        )
        if duplicate_names:
            errors.append(
                "Resolved Experiment Plan contains duplicate Run instances: "
                + ", ".join(duplicate_names)
            )
        ordinals = [stage.ordinal for stage in stages]
        invalid_ordinals = [
            stage.instance_name
            for stage in stages
            if (
                not isinstance(stage.ordinal, int)
                or isinstance(stage.ordinal, bool)
                or stage.ordinal < 0
            )
        ]
        if invalid_ordinals:
            errors.append(
                "Resolved Runs have invalid ordinals: "
                + ", ".join(invalid_ordinals)
            )
        duplicate_ordinals = sorted(
            ordinal
            for ordinal, count in Counter(ordinals).items()
            if count > 1
        )
        if duplicate_ordinals:
            errors.append(
                "Resolved Experiment Plan contains duplicate ordinals: "
                + ", ".join(str(item) for item in duplicate_ordinals)
            )
        name_set = set(names)
        for stage in stages:
            if (
                not isinstance(stage.resource_key, str)
                or not stage.resource_key
            ):
                errors.append(
                    f"Run {stage.instance_name!r} has an empty resource_key"
                )
            if len(stage.input_instances) != len(
                set(stage.input_instances)
            ):
                errors.append(
                    f"Run {stage.instance_name!r} has duplicate inputs"
                )
            unknown = sorted(set(stage.input_instances) - name_set)
            if unknown:
                errors.append(
                    f"Run {stage.instance_name!r} references unknown inputs: "
                    + ", ".join(unknown)
                )
        if errors:
            raise PlanValidationError(errors)

    @staticmethod
    def _ordered_stages(
        stages: tuple[PreparedStage, ...],
    ) -> list[PreparedStage]:
        by_name = {
            stage.instance_name: position
            for position, stage in enumerate(stages)
        }
        indegree = [
            len(stage.input_instances) for stage in stages
        ]
        dependents: list[list[int]] = [[] for _ in stages]
        for position, stage in enumerate(stages):
            for input_name in stage.input_instances:
                dependents[by_name[input_name]].append(position)

        global_ready: list[tuple[int, int]] = []
        ready_by_resource: dict[str, list[tuple[int, int]]] = {}
        available: set[int] = set()
        for position, stage in enumerate(stages):
            if indegree[position] == 0:
                item = (stage.ordinal, position)
                heappush(global_ready, item)
                heappush(
                    ready_by_resource.setdefault(stage.resource_key, []),
                    item,
                )
                available.add(position)

        result: list[PreparedStage] = []
        current_resource: str | None = None
        while available:
            preferred = (
                ready_by_resource.get(current_resource, [])
                if current_resource is not None
                else []
            )
            selected = ResolvedPlanGraph._pop_available(
                preferred, available
            )
            if selected is None:
                selected = ResolvedPlanGraph._pop_available(
                    global_ready, available
                )
            if selected is None:
                raise RuntimeError("ready Run index is inconsistent")
            available.remove(selected)
            stage = stages[selected]
            result.append(stage)
            current_resource = stage.resource_key
            for dependent in dependents[selected]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    item = (stages[dependent].ordinal, dependent)
                    heappush(global_ready, item)
                    heappush(
                        ready_by_resource.setdefault(
                            stages[dependent].resource_key, []
                        ),
                        item,
                    )
                    available.add(dependent)

        if len(result) != len(stages):
            cycle = sorted(
                stages[position].instance_name
                for position, remaining in enumerate(indegree)
                if remaining > 0
            )
            raise DependencyCycleError(
                "Resolved Experiment Plan contains a Run dependency cycle: "
                f"{cycle}"
            )
        return result

    @staticmethod
    def _pop_available(
        heap: list[tuple[int, int]],
        available: set[int],
    ) -> int | None:
        while heap:
            _, position = heappop(heap)
            if position in available:
                return position
        return None

    def _descendants(self, position: int) -> set[int]:
        descendants: set[int] = set()
        pending = list(self._dependents[position])
        while pending:
            candidate = pending.pop()
            if candidate in descendants:
                continue
            descendants.add(candidate)
            pending.extend(self._dependents[candidate])
        return descendants

    def _validate_state(self, state: PlanExecutionState) -> None:
        if not isinstance(state, PlanExecutionState):
            raise PlanProtocolError("Plan execution state has an invalid type")
        if state._graph_key != self._graph_key:
            raise PlanProtocolError(
                "Plan execution state belongs to another Resolved Plan graph"
            )
        if (
            not isinstance(state._generation, int)
            or isinstance(state._generation, bool)
            or state._generation < 0
        ):
            raise PlanProtocolError(
                "Plan execution state has an invalid generation"
            )
        if len(state._phases) != len(self._execution_order) or any(
            not isinstance(phase, _RunPhase)
            for phase in state._phases
        ):
            raise PlanProtocolError(
                "Plan execution state has invalid Run phases"
            )
