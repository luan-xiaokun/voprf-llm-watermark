from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .engine import ExperimentRuns
from .errors import ExperimentRunError
from .models import ResolvedExperimentPlan
from .plan_graph import ResolvedPlanGraph


def _add_location_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current directory).",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help=(
            "Experiment Workspace (default: "
            "<repository>/output/experiments)."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wmexp",
        description="Resolve and execute local text-watermark experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check", help="Resolve and validate an Experiment Plan."
    )
    check.add_argument("plan", type=Path)
    check.add_argument("--max-runs", type=int, default=10_000)
    check.add_argument(
        "--json", action="store_true", help="Print resolved Plan as JSON."
    )
    _add_location_options(check)

    run = subparsers.add_parser(
        "run", help="Execute or resume an Experiment Plan."
    )
    run.add_argument("plan", type=Path)
    run.add_argument(
        "--new-attempt",
        action="append",
        default=[],
        metavar="STAGE_INSTANCE",
        help=(
            "Execute a new Attempt even if a canonical Artifact exists; "
            "repeat for multiple stage instances."
        ),
    )
    run.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue independent Runs after a failure.",
    )
    _add_location_options(run)

    status = subparsers.add_parser(
        "status", help="Show Runs, Attempts, and Artifacts for a Plan."
    )
    status.add_argument(
        "plan",
        help="Experiment Plan path or a registered Plan digest.",
    )
    status.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    _add_location_options(status)

    errors = subparsers.add_parser(
        "errors", help="Show failed Attempt errors for a Plan."
    )
    errors.add_argument(
        "plan",
        help="Experiment Plan path or a registered Plan digest.",
    )
    errors.add_argument(
        "--stage",
        help="Limit errors to one resolved stage instance.",
    )
    errors.add_argument(
        "--attempt",
        help="Limit errors to one Attempt identity.",
    )
    errors.add_argument(
        "--traceback",
        action="store_true",
        help="Include stored tracebacks in human-readable output.",
    )
    errors.add_argument(
        "--json", action="store_true", help="Print machine-readable JSON."
    )
    _add_location_options(errors)
    return parser


def _runs(args: argparse.Namespace, *, enforce_clean: bool) -> ExperimentRuns:
    repository = args.repository.expanduser().resolve()
    workspace = (
        args.workspace
        if args.workspace is not None
        else repository / "output" / "experiments"
    )
    return ExperimentRuns.local(
        repository=repository,
        workspace=workspace,
        enforce_clean=enforce_clean,
    )


def _print_check(plan: ResolvedExperimentPlan) -> None:
    print(f"Plan: {plan.name}")
    print(f"Digest: {plan.plan_digest}")
    print(f"Code revision: {plan.code_revision}")
    if plan.implementation_dirty:
        print("Implementation: DIRTY (wmexp run will reject it)")
        for path in plan.implementation_dirty:
            print(f"  {path}")
    else:
        print("Implementation: clean")
    print(f"Resolved Runs: {len(plan.stages)}")
    print("Execution order:")
    graph = ResolvedPlanGraph(plan)
    for ordinal, stage in enumerate(graph.execution_order, start=1):
        source = (
            " <- " + ", ".join(stage.input_instances)
            if stage.input_instances
            else ""
        )
        print(
            f"  {ordinal:>3}. {stage.instance_name} "
            f"({stage.kind}){source}"
        )
        print(f"       resource: {stage.resource_key}")
        for key in sorted(stage.settings):
            origin = stage.setting_sources.get(key, "adapter-derived")
            rendered = json.dumps(
                stage.settings[key], ensure_ascii=False, sort_keys=True
            )
            print(f"       {key} = {rendered} [{origin}]")


def _print_status(value: dict[str, Any]) -> None:
    print(f"Plan: {value['plan_digest']}")
    print(
        f"Runs: {len(value['runs'])}; "
        f"Attempts: {len(value['attempts'])}; "
        f"Artifacts: {len(value['artifacts'])}"
    )
    for run in value["runs"]:
        canonical = run.get("canonical_artifact_identity") or "-"
        run_identity = run.get("run_identity") or "-"
        state = run.get("state", "pending")
        print(
            f"  {run['instance_name']}: {state} run={run_identity} "
            f"canonical={canonical}"
        )
        attempts = [
            attempt
            for attempt in value["attempts"]
            if attempt["run_identity"] == run.get("run_identity")
        ]
        for attempt in attempts:
            print(
                f"    {attempt['attempt_identity']}: {attempt['state']}"
            )


def _print_errors(
    value: dict[str, Any], *, include_traceback: bool
) -> None:
    attempts = value["attempts"]
    print(f"Plan: {value['plan_digest']}")
    print(f"Errors: {len(attempts)}")
    for attempt in attempts:
        error = attempt["error"]
        error_type = error.get("type") or "UnknownError"
        message = error.get("message") or ""
        print(
            f"  {attempt['instance_name']}: "
            f"{attempt['attempt_identity']} ({attempt['state']})"
        )
        print(f"    {error_type}: {message}")
        traceback = error.get("traceback")
        if include_traceback and traceback:
            print("    Traceback:")
            for line in str(traceback).splitlines():
                print(f"      {line}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check":
            runs = _runs(args, enforce_clean=False)
            plan = runs.resolve(args.plan, max_runs=args.max_runs)
            if args.json:
                print(
                    json.dumps(
                        plan.to_dict(),
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                )
            else:
                _print_check(plan)
            return 0

        if args.command == "run":
            report = _runs(args, enforce_clean=True).execute(
                args.plan,
                new_attempts=args.new_attempt,
                keep_going=args.keep_going,
            )
            print(
                json.dumps(
                    report.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0 if report.succeeded else 1

        runs = _runs(args, enforce_clean=False)
        if args.command == "errors":
            plan_or_digest: str | Path = args.plan
            if Path(args.plan).exists():
                plan_or_digest = Path(args.plan)
            errors = runs.errors(
                plan_or_digest,
                stage=args.stage,
                attempt=args.attempt,
            ).to_dict()
            if args.json:
                print(
                    json.dumps(
                        errors,
                        ensure_ascii=False,
                        sort_keys=True,
                        indent=2,
                    )
                )
            else:
                _print_errors(
                    errors, include_traceback=args.traceback
                )
            return 0

        plan_or_digest: str | Path = args.plan
        if Path(args.plan).exists():
            plan_or_digest = Path(args.plan)
        status = runs.status(plan_or_digest).to_dict()
        if args.json:
            print(
                json.dumps(
                    status, ensure_ascii=False, sort_keys=True, indent=2
                )
            )
        else:
            _print_status(status)
        return 0
    except (ExperimentRunError, KeyError, ValueError) as error:
        print(f"wmexp: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
