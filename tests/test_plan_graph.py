from __future__ import annotations

from pathlib import Path

import pytest

from watermark_suite.experiments.errors import (
    DependencyCycleError,
    PlanProtocolError,
    PlanValidationError,
)
from watermark_suite.experiments.models import (
    PreparedStage,
    ResolvedExperimentPlan,
)
from watermark_suite.experiments.plan_graph import (
    Continuation,
    ResolvedPlanGraph,
    RunOutcome,
)


def prepared(
    name: str,
    ordinal: int,
    resource_key: str,
    *,
    inputs: tuple[str, ...] = (),
) -> PreparedStage:
    return PreparedStage(
        stage_name=name,
        instance_name=name,
        kind="fake",
        settings={},
        setting_sources={},
        semantic_settings={},
        execution_settings={},
        adapter_revision="fake-v1",
        artifact_schema_revision="fake-records-v1",
        input_instances=inputs,
        ordinal=ordinal,
        resource_key=resource_key,
    )


def plan(*stages: PreparedStage) -> ResolvedExperimentPlan:
    return ResolvedExperimentPlan(
        version=1,
        name="graph-test",
        source_path=Path("graph-test.yaml"),
        plan_digest="plan_graph_test",
        code_revision="test",
        implementation_dirty=(),
        stages=stages,
        source={},
    )


def test_execution_order_prefers_ready_run_with_same_resource():
    graph = ResolvedPlanGraph(
        plan(
            prepared("root-a", 0, "model:a"),
            prepared("root-b", 1, "model:b"),
            prepared(
                "follow-a",
                2,
                "model:a",
                inputs=("root-a",),
            ),
        )
    )

    assert [
        stage.instance_name for stage in graph.execution_order
    ] == ["root-a", "follow-a", "root-b"]


def test_all_success_execution_matches_plan_execution_order():
    graph = ResolvedPlanGraph(
        plan(
            prepared("root-a", 0, "model:a"),
            prepared("root-b", 1, "model:b"),
            prepared(
                "follow-a",
                2,
                "model:a",
                inputs=("root-a",),
            ),
        )
    )
    state = graph.start()
    observed: list[str] = []

    while (selected := graph.next(state)) is not None:
        assert graph.next(state) == selected
        observed.append(selected.prepared.instance_name)
        state = graph.record(
            state,
            selected,
            RunOutcome.SUCCEEDED,
        ).state

    assert observed == [
        stage.instance_name for stage in graph.execution_order
    ]


def test_failure_blocks_descendants_and_keeps_canonical_subsequence():
    graph = ResolvedPlanGraph(
        plan(
            prepared("broken", 0, "model:a"),
            prepared("independent", 1, "model:b"),
            prepared(
                "blocked",
                2,
                "model:a",
                inputs=("broken",),
            ),
            prepared(
                "blocked-transitively",
                3,
                "model:a",
                inputs=("blocked",),
            ),
            prepared(
                "independent-child",
                4,
                "model:b",
                inputs=("independent",),
            ),
        )
    )
    state = graph.start()
    failed = graph.next(state)
    assert failed is not None

    transition = graph.record(
        state,
        failed,
        RunOutcome.FAILED,
        continuation=Continuation.CONTINUE,
    )

    assert transition.classified.failed == ("broken",)
    assert transition.classified.blocked == (
        "blocked",
        "blocked-transitively",
    )
    remaining: list[str] = []
    state = transition.state
    while (selected := graph.next(state)) is not None:
        remaining.append(selected.prepared.instance_name)
        state = graph.record(
            state,
            selected,
            RunOutcome.SUCCEEDED,
        ).state
    assert remaining == ["independent", "independent-child"]


def test_stop_distinguishes_blocked_descendants_from_skipped_runs():
    graph = ResolvedPlanGraph(
        plan(
            prepared("broken", 0, "model:a"),
            prepared("independent", 1, "model:b"),
            prepared(
                "blocked",
                2,
                "model:a",
                inputs=("broken",),
            ),
            prepared(
                "blocked-transitively",
                3,
                "model:a",
                inputs=("blocked",),
            ),
            prepared(
                "independent-child",
                4,
                "model:b",
                inputs=("independent",),
            ),
        )
    )
    state = graph.start()
    selected = graph.next(state)
    assert selected is not None

    transition = graph.record(
        state,
        selected,
        RunOutcome.FAILED,
        continuation=Continuation.STOP,
    )

    assert transition.classified.failed == ("broken",)
    assert transition.classified.blocked == (
        "blocked",
        "blocked-transitively",
    )
    assert transition.classified.skipped == (
        "independent",
        "independent-child",
    )
    assert graph.next(transition.state) is None


def test_record_rejects_a_stale_selection():
    graph = ResolvedPlanGraph(
        plan(
            prepared("first", 0, "model:a"),
            prepared("second", 1, "model:b"),
        )
    )
    state = graph.start()
    selected = graph.next(state)
    assert selected is not None
    state = graph.record(
        state,
        selected,
        RunOutcome.SUCCEEDED,
    ).state

    with pytest.raises(PlanProtocolError, match="foreign, stale"):
        graph.record(
            state,
            selected,
            RunOutcome.SUCCEEDED,
        )


def test_graph_rejects_unknown_inputs_and_cycles():
    with pytest.raises(PlanValidationError, match="unknown inputs"):
        ResolvedPlanGraph(
            plan(
                prepared(
                    "orphan",
                    0,
                    "model:a",
                    inputs=("missing",),
                )
            )
        )

    with pytest.raises(DependencyCycleError, match="dependency cycle"):
        ResolvedPlanGraph(
            plan(
                prepared("left", 0, "model:a", inputs=("right",)),
                prepared("right", 1, "model:b", inputs=("left",)),
            )
        )
