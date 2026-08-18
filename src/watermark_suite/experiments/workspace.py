from __future__ import annotations

import errno
import fcntl
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .errors import ArtifactIntegrityError, AttemptConflictError
from .identity import sha256_file
from .models import (
    ArtifactIdentity,
    ArtifactRef,
    AttemptIdentity,
    AttemptState,
    JsonObject,
    PlanStatus,
    ResolvedExperimentPlan,
    RunIdentity,
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


class ExperimentWorkspace:
    """Transactional ledger and immutable Artifact home for local experiments."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()
        self.artifacts_dir = self.root / "artifacts"
        self.attempts_dir = self.root / "attempts"
        self.plans_dir = self.root / "resolved-plans"
        self.logs_dir = self.root / "logs"
        self.locks_dir = self.root / "locks"
        for directory in (
            self.root,
            self.artifacts_dir,
            self.attempts_dir,
            self.plans_dir,
            self.logs_dir,
            self.locks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "ledger.sqlite"
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_digest TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    resolved_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS runs (
                    run_identity TEXT PRIMARY KEY,
                    stage_name TEXT NOT NULL,
                    stage_kind TEXT NOT NULL,
                    canonical_json TEXT NOT NULL,
                    canonical_artifact_identity TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS plan_runs (
                    plan_digest TEXT NOT NULL,
                    instance_name TEXT NOT NULL,
                    run_identity TEXT NOT NULL,
                    artifact_identity TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (plan_digest, instance_name, run_identity),
                    FOREIGN KEY (plan_digest) REFERENCES plans(plan_digest),
                    FOREIGN KEY (run_identity) REFERENCES runs(run_identity)
                );
                CREATE TABLE IF NOT EXISTS plan_stage_states (
                    plan_digest TEXT NOT NULL,
                    instance_name TEXT NOT NULL,
                    state TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (plan_digest, instance_name),
                    FOREIGN KEY (plan_digest) REFERENCES plans(plan_digest)
                );
                CREATE TABLE IF NOT EXISTS attempts (
                    attempt_identity TEXT PRIMARY KEY,
                    run_identity TEXT NOT NULL,
                    state TEXT NOT NULL,
                    provenance_json TEXT NOT NULL,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (run_identity) REFERENCES runs(run_identity)
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    attempt_identity TEXT NOT NULL,
                    work_identity TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    chunk_path TEXT NOT NULL,
                    chunk_sha256 TEXT NOT NULL,
                    record_count INTEGER NOT NULL,
                    files_json TEXT NOT NULL DEFAULT '{}',
                    committed_at TEXT NOT NULL,
                    PRIMARY KEY (attempt_identity, work_identity),
                    FOREIGN KEY (attempt_identity)
                        REFERENCES attempts(attempt_identity)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    artifact_identity TEXT PRIMARY KEY,
                    run_identity TEXT NOT NULL,
                    attempt_identity TEXT NOT NULL UNIQUE,
                    schema_revision TEXT NOT NULL,
                    path TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (run_identity) REFERENCES runs(run_identity),
                    FOREIGN KEY (attempt_identity)
                        REFERENCES attempts(attempt_identity)
                );
                CREATE INDEX IF NOT EXISTS attempts_by_run
                    ON attempts(run_identity, created_at);
                CREATE INDEX IF NOT EXISTS checkpoints_by_attempt
                    ON checkpoints(attempt_identity, ordinal);
                CREATE INDEX IF NOT EXISTS plan_runs_by_plan
                    ON plan_runs(plan_digest, instance_name);
                """
            )
            checkpoint_columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(checkpoints)"
                ).fetchall()
            }
            if "files_json" not in checkpoint_columns:
                connection.execute(
                    "ALTER TABLE checkpoints ADD COLUMN "
                    "files_json TEXT NOT NULL DEFAULT '{}'"
                )

    def register_plan(self, plan: ResolvedExperimentPlan) -> Path:
        serialized = json.dumps(plan.to_dict(), ensure_ascii=False, sort_keys=True)
        path = self.plans_dir / f"{plan.plan_digest}.json"
        if not path.exists():
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(serialized + "\n", encoding="utf-8")
            os.replace(temporary, path)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO plans
                    (plan_digest, name, resolved_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (plan.plan_digest, plan.name, serialized, _now()),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO plan_stage_states
                    (plan_digest, instance_name, state, updated_at)
                VALUES (?, ?, 'pending', ?)
                """,
                (
                    (plan.plan_digest, stage.instance_name, _now())
                    for stage in plan.stages
                ),
            )
        return path

    def reset_plan_stage_states(self, plan_digest: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE plan_stage_states
                SET state = 'pending', updated_at = ?
                WHERE plan_digest = ?
                """,
                (_now(), plan_digest),
            )

    def set_plan_stage_state(
        self,
        plan_digest: str,
        instance_name: str,
        state: str,
    ) -> None:
        if state not in {"pending", "blocked", "skipped"}:
            raise ValueError(f"unsupported Plan stage state {state!r}")
        with self.connect() as connection:
            changed = connection.execute(
                """
                UPDATE plan_stage_states
                SET state = ?, updated_at = ?
                WHERE plan_digest = ? AND instance_name = ?
                """,
                (state, _now(), plan_digest, instance_name),
            )
            if changed.rowcount != 1:
                raise KeyError(
                    f"unknown Plan stage {plan_digest}:{instance_name}"
                )

    def plan_stage_states(self, plan_digest: str) -> dict[str, str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT instance_name, state
                FROM plan_stage_states
                WHERE plan_digest = ?
                """,
                (plan_digest,),
            ).fetchall()
        return {row["instance_name"]: row["state"] for row in rows}

    def load_plan(self, plan_digest: str) -> ResolvedExperimentPlan:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT resolved_json FROM plans WHERE plan_digest = ?",
                (plan_digest,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Experiment Plan {plan_digest}")
        return ResolvedExperimentPlan.from_dict(json.loads(row["resolved_json"]))

    def register_run(
        self,
        identity: RunIdentity,
        *,
        stage_name: str,
        stage_kind: str,
        canonical_json: str,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO runs
                    (run_identity, stage_name, stage_kind, canonical_json,
                     canonical_artifact_identity, created_at)
                VALUES (?, ?, ?, ?, NULL, ?)
                """,
                (
                    identity.value,
                    stage_name,
                    stage_kind,
                    canonical_json,
                    _now(),
                ),
            )
            row = connection.execute(
                "SELECT canonical_json FROM runs WHERE run_identity = ?",
                (identity.value,),
            ).fetchone()
            if row["canonical_json"] != canonical_json:
                raise ArtifactIntegrityError(
                    f"Run Identity collision for {identity.value}"
                )

    def bind_plan_run(
        self,
        *,
        plan_digest: str,
        instance_name: str,
        run_identity: RunIdentity,
        artifact_identity: ArtifactIdentity | None = None,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO plan_runs
                    (plan_digest, instance_name, run_identity,
                     artifact_identity, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_digest, instance_name, run_identity)
                DO UPDATE SET artifact_identity = COALESCE(
                    excluded.artifact_identity,
                    plan_runs.artifact_identity
                )
                """,
                (
                    plan_digest,
                    instance_name,
                    run_identity.value,
                    artifact_identity.value if artifact_identity else None,
                    _now(),
                ),
            )

    def _artifact_from_row(self, row: sqlite3.Row) -> ArtifactRef:
        manifest = json.loads(row["manifest_json"])
        path = self.root / row["path"]
        return ArtifactRef(
            identity=ArtifactIdentity(row["artifact_identity"]),
            path=path,
            schema_revision=row["schema_revision"],
            run_identity=RunIdentity(row["run_identity"]),
            attempt_identity=AttemptIdentity(row["attempt_identity"]),
            source_artifacts=tuple(
                ArtifactIdentity(value)
                for value in manifest.get("source_artifacts", [])
            ),
            manifest=manifest,
        )

    def _verify_artifact(self, artifact: ArtifactRef) -> ArtifactRef:
        manifest_path = artifact.path / "manifest.json"
        if not manifest_path.is_file():
            raise ArtifactIntegrityError(
                f"Artifact {artifact.identity.value} has no manifest"
            )
        try:
            disk_manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactIntegrityError(
                f"Artifact {artifact.identity.value} has an invalid manifest"
            ) from error
        if disk_manifest != artifact.manifest:
            raise ArtifactIntegrityError(
                f"Artifact {artifact.identity.value} manifest differs from "
                "the ledger"
            )
        for relative_path, expected_digest in artifact.manifest.get(
            "files", {}
        ).items():
            path = artifact.path / relative_path
            if not path.is_file() or sha256_file(path) != expected_digest:
                raise ArtifactIntegrityError(
                    f"Artifact {artifact.identity.value} has a corrupt file "
                    f"{relative_path}"
                )
        return artifact

    def canonical_artifact(self, run_identity: RunIdentity) -> ArtifactRef | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT artifacts.*
                FROM runs
                JOIN artifacts
                  ON artifacts.artifact_identity =
                     runs.canonical_artifact_identity
                WHERE runs.run_identity = ?
                """,
                (run_identity.value,),
            ).fetchone()
        return (
            self._verify_artifact(self._artifact_from_row(row))
            if row
            else None
        )

    def artifact(self, identity: ArtifactIdentity) -> ArtifactRef:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM artifacts WHERE artifact_identity = ?",
                (identity.value,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Artifact {identity.value}")
        return self._verify_artifact(self._artifact_from_row(row))

    def find_resumable_attempt(
        self, run_identity: RunIdentity
    ) -> AttemptIdentity | None:
        states = (
            AttemptState.CREATED,
            AttemptState.RUNNING,
            AttemptState.INTERRUPTED,
            AttemptState.FINALIZING,
            AttemptState.FAILED,
        )
        placeholders = ",".join("?" for _ in states)
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT attempt_identity
                FROM attempts
                WHERE run_identity = ?
                  AND state IN ({placeholders})
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (run_identity.value, *states),
            ).fetchone()
        return AttemptIdentity(row["attempt_identity"]) if row else None

    def create_attempt(
        self,
        run_identity: RunIdentity,
        provenance: JsonObject,
    ) -> AttemptIdentity:
        identity = AttemptIdentity(f"attempt_{uuid.uuid4().hex}")
        now = _now()
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO attempts
                    (attempt_identity, run_identity, state, provenance_json,
                     error_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, NULL, ?, ?)
                """,
                (
                    identity.value,
                    run_identity.value,
                    AttemptState.CREATED,
                    json.dumps(provenance, ensure_ascii=False, sort_keys=True),
                    now,
                    now,
                ),
            )
        (self.attempts_dir / identity.value / "chunks").mkdir(
            parents=True, exist_ok=True
        )
        return identity

    def attempt_state(self, identity: AttemptIdentity) -> AttemptState:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT state FROM attempts WHERE attempt_identity = ?",
                (identity.value,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Attempt {identity.value}")
        return AttemptState(row["state"])

    def attempt_provenance(self, identity: AttemptIdentity) -> JsonObject:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT provenance_json
                FROM attempts
                WHERE attempt_identity = ?
                """,
                (identity.value,),
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown Attempt {identity.value}")
        return json.loads(row["provenance_json"])

    def add_attempt_runtime(
        self, identity: AttemptIdentity, elapsed_seconds: float
    ) -> None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT provenance_json
                FROM attempts
                WHERE attempt_identity = ?
                """,
                (identity.value,),
            ).fetchone()
            if row is None:
                raise AttemptConflictError(
                    f"unknown Attempt {identity.value}"
                )
            provenance = json.loads(row["provenance_json"])
            segments = provenance.setdefault("runtime_segments_seconds", [])
            segments.append(elapsed_seconds)
            provenance["total_runtime_seconds"] = sum(segments)
            connection.execute(
                """
                UPDATE attempts
                SET provenance_json = ?, updated_at = ?
                WHERE attempt_identity = ?
                """,
                (
                    json.dumps(
                        provenance, ensure_ascii=False, sort_keys=True
                    ),
                    _now(),
                    identity.value,
                ),
            )

    def set_attempt_state(
        self,
        identity: AttemptIdentity,
        state: AttemptState,
        *,
        error: JsonObject | None = None,
    ) -> None:
        with self.connect() as connection:
            changed = connection.execute(
                """
                UPDATE attempts
                SET state = ?, error_json = ?, updated_at = ?
                WHERE attempt_identity = ?
                """,
                (
                    state,
                    (
                        json.dumps(error, ensure_ascii=False, sort_keys=True)
                        if error
                        else None
                    ),
                    _now(),
                    identity.value,
                ),
            ).rowcount
        if changed != 1:
            raise AttemptConflictError(f"unknown Attempt {identity.value}")

    def attempt_directory(self, identity: AttemptIdentity) -> Path:
        return self.attempts_dir / identity.value

    @contextmanager
    def run_lease(self, identity: RunIdentity) -> Iterator[None]:
        """Prevent concurrent attempt selection/execution for one Run."""

        lease_path = self.locks_dir / f"{identity.value}.lock"
        lease = lease_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                raise AttemptConflictError(
                    f"Run {identity.value} is already active in another "
                    "executor"
                ) from error
            yield
        finally:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()

    @contextmanager
    def attempt_lease(
        self, identity: AttemptIdentity
    ) -> Iterator[None]:
        """Hold the single-machine execution lease for one Attempt."""

        directory = self.attempt_directory(identity)
        if not directory.is_dir():
            raise AttemptConflictError(f"unknown Attempt {identity.value}")
        lease_path = directory / "execution.lock"
        lease = lease_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise
                raise AttemptConflictError(
                    f"Attempt {identity.value} is already active in another "
                    "executor"
                ) from error
            if self.attempt_state(identity) in {
                AttemptState.RUNNING,
                AttemptState.FINALIZING,
            }:
                self.set_attempt_state(identity, AttemptState.INTERRUPTED)
            yield
        finally:
            try:
                fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            finally:
                lease.close()

    def completed_work(self, identity: AttemptIdentity) -> set[str]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT work_identity
                FROM checkpoints
                WHERE attempt_identity = ?
                """,
                (identity.value,),
            ).fetchall()
        return {row["work_identity"] for row in rows}

    def commit_checkpoint(
        self,
        *,
        attempt: AttemptIdentity,
        work_identity: str,
        ordinal: int,
        chunk_path: Path,
        record_count: int,
        files: JsonObject | None = None,
    ) -> None:
        relative_path = chunk_path.relative_to(self.root)
        digest = sha256_file(chunk_path)
        with self.connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO checkpoints
                        (attempt_identity, work_identity, ordinal, chunk_path,
                         chunk_sha256, record_count, files_json, committed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        attempt.value,
                        work_identity,
                        ordinal,
                        str(relative_path),
                        digest,
                        record_count,
                        json.dumps(
                            files or {}, ensure_ascii=False, sort_keys=True
                        ),
                        _now(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                raise AttemptConflictError(
                    f"work item {work_identity!r} is already committed"
                ) from error

    def checkpoint_chunks(self, attempt: AttemptIdentity) -> list[JsonObject]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM checkpoints
                WHERE attempt_identity = ?
                ORDER BY ordinal
                """,
                (attempt.value,),
            ).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["files"] = json.loads(value.pop("files_json", "{}"))
            result.append(value)
        return result

    def artifact_for_attempt(
        self, attempt: AttemptIdentity
    ) -> ArtifactRef | None:
        with self.connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM artifacts
                WHERE attempt_identity = ?
                """,
                (attempt.value,),
            ).fetchone()
        return self._artifact_from_row(row) if row else None

    def register_artifact(
        self,
        artifact: ArtifactRef,
        *,
        plan_digest: str,
        instance_name: str,
    ) -> bool:
        relative_path = artifact.path.relative_to(self.root)
        manifest_json = json.dumps(
            artifact.manifest, ensure_ascii=False, sort_keys=True
        )
        with self.connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts
                    (artifact_identity, run_identity, attempt_identity,
                     schema_revision, path, manifest_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    artifact.identity.value,
                    artifact.run_identity.value,
                    artifact.attempt_identity.value,
                    artifact.schema_revision,
                    str(relative_path),
                    manifest_json,
                    _now(),
                ),
            )
            existing = connection.execute(
                """
                SELECT artifact_identity, run_identity, manifest_json
                FROM artifacts
                WHERE attempt_identity = ?
                """,
                (artifact.attempt_identity.value,),
            ).fetchone()
            if existing is None:
                raise ArtifactIntegrityError(
                    f"Artifact Identity collision for "
                    f"{artifact.identity.value}"
                )
            if existing["artifact_identity"] != artifact.identity.value:
                raise ArtifactIntegrityError(
                    "an Attempt cannot finalize more than one Artifact"
                )
            if (
                existing["run_identity"] != artifact.run_identity.value
                or existing["manifest_json"] != manifest_json
            ):
                raise ArtifactIntegrityError(
                    f"Artifact ledger entry differs for "
                    f"{artifact.identity.value}"
                )
            connection.execute(
                """
                UPDATE attempts
                SET state = ?, updated_at = ?
                WHERE attempt_identity = ?
                """,
                (
                    AttemptState.SUCCEEDED,
                    _now(),
                    artifact.attempt_identity.value,
                ),
            )
            row = connection.execute(
                """
                SELECT canonical_artifact_identity
                FROM runs
                WHERE run_identity = ?
                """,
                (artifact.run_identity.value,),
            ).fetchone()
            became_canonical = row["canonical_artifact_identity"] is None
            if became_canonical:
                connection.execute(
                    """
                    UPDATE runs
                    SET canonical_artifact_identity = ?
                    WHERE run_identity = ?
                      AND canonical_artifact_identity IS NULL
                    """,
                    (
                        artifact.identity.value,
                        artifact.run_identity.value,
                    ),
                )
            connection.execute(
                """
                INSERT INTO plan_runs
                    (plan_digest, instance_name, run_identity,
                     artifact_identity, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(plan_digest, instance_name, run_identity)
                DO UPDATE SET artifact_identity = excluded.artifact_identity
                """,
                (
                    plan_digest,
                    instance_name,
                    artifact.run_identity.value,
                    artifact.identity.value,
                    _now(),
                ),
            )
        return became_canonical

    def status(self, plan_digest: str) -> PlanStatus:
        with self.connect() as connection:
            runs = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT plan_runs.instance_name, runs.*
                    FROM plan_runs
                    JOIN runs USING (run_identity)
                    WHERE plan_runs.plan_digest = ?
                    ORDER BY plan_runs.created_at
                    """,
                    (plan_digest,),
                ).fetchall()
            ]
            run_ids = [run["run_identity"] for run in runs]
            attempts: list[JsonObject] = []
            artifacts: list[JsonObject] = []
            if run_ids:
                placeholders = ",".join("?" for _ in run_ids)
                attempts = [
                    dict(row)
                    for row in connection.execute(
                        f"""
                        SELECT * FROM attempts
                        WHERE run_identity IN ({placeholders})
                        ORDER BY created_at
                        """,
                        run_ids,
                    ).fetchall()
                ]
                artifacts = [
                    dict(row)
                    for row in connection.execute(
                        f"""
                        SELECT * FROM artifacts
                        WHERE run_identity IN ({placeholders})
                        ORDER BY created_at
                        """,
                        run_ids,
                    ).fetchall()
                ]
        for run in runs:
            if run["canonical_artifact_identity"]:
                run["state"] = AttemptState.SUCCEEDED
                continue
            matching = [
                attempt
                for attempt in attempts
                if attempt["run_identity"] == run["run_identity"]
            ]
            run["state"] = (
                matching[-1]["state"]
                if matching
                else "pending"
            )
        return PlanStatus(
            plan_digest=plan_digest,
            runs=tuple(runs),
            attempts=tuple(attempts),
            artifacts=tuple(artifacts),
        )

    def errors(
        self,
        plan_digest: str,
        *,
        instance_name: str | None = None,
        attempt_identity: str | None = None,
    ) -> tuple[JsonObject, ...]:
        conditions = [
            "plan_runs.plan_digest = ?",
            "attempts.error_json IS NOT NULL",
        ]
        parameters: list[str] = [plan_digest]
        if instance_name is not None:
            conditions.append("plan_runs.instance_name = ?")
            parameters.append(instance_name)
        if attempt_identity is not None:
            conditions.append("attempts.attempt_identity = ?")
            parameters.append(attempt_identity)

        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT
                    plan_runs.instance_name,
                    runs.stage_name,
                    runs.stage_kind,
                    attempts.attempt_identity,
                    attempts.run_identity,
                    attempts.state,
                    attempts.error_json,
                    attempts.created_at,
                    attempts.updated_at
                FROM plan_runs
                JOIN runs USING (run_identity)
                JOIN attempts USING (run_identity)
                WHERE {' AND '.join(conditions)}
                ORDER BY attempts.created_at DESC, plan_runs.instance_name
                """,
                parameters,
            ).fetchall()

        errors = []
        for row in rows:
            value = dict(row)
            value["error"] = json.loads(value.pop("error_json"))
            errors.append(value)
        return tuple(errors)
