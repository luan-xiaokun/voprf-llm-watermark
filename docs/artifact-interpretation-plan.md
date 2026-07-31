# Artifact Interpretation Deepening Plan

Status: implemented on `artifact-interpretation-deepening`; retained as the
design and verification record.

Implementation result:

- one public `artifact_interpretation` seam with exact support for the eight
  current kind/schema pairs;
- full transitive preflight with deterministic aggregated diagnostics;
- eager aggregate and filtered streaming Sample Metric Facts;
- atomic cutover of all six Result recipes and removal of the old catalog
  seam;
- `artifact-interpretation-v1` included in aggregation Run revision and
  report summaries;
- all recipe paths covered through the real `result-aggregation` stage.

## Outcome

Replace the current partially normalized `ArtifactCatalog`/`ArtifactView`
path with one deep, read-only Artifact Interpretation module. The module
turns each supported Artifact and its complete transitive lineage into
validated scientific facts. Result recipes consume only those facts and never
read raw manifests, summaries, records, or ancestor Artifacts.

The external interface remains deliberately small:

```text
Experiment Workspace + ordered root Artifacts
    -> validated Interpretation Set

Interpretation Set + Artifact Identity + requested metrics
    -> streamed Sample Metric Facts
```

Artifact Interpretation is an in-process derivation. It does not create a
Run, Attempt, or Artifact.

## Agreed Domain Model

The following terms are recorded in `CONTEXT.md`:

- **Artifact Interpretation** — the validated scientific meaning of one
  Artifact and its transitive lineage.
- **Lineage Invariant** — an inherited scientific fact that may be repeated
  but never overridden.
- **Metric Fact** — a normalized aggregate- or Sample-scoped scientific
  observation.
- **Artifact Relation** — a normalized direct or transitive lineage
  relationship that does not itself choose a comparative join.

Responsibility is split as follows:

| Module | Owns | Does not own |
| --- | --- | --- |
| Artifact Interpretation | Schema validation, lineage validation, normalized scientific provenance, Metric Facts, Artifact Relations, numeric consistency | Cross-Artifact pairing, comparison policy, threshold selection, report rows |
| Result recipe | Exact joins, comparability policy, effective FPR, recipe-specific rows and source accounting | Raw Artifact schema, manifest traversal, summary keys, records parsing |
| Renderer | Inclusion and presentation policy | Scientific aggregation or Artifact interpretation |

## Locked Decisions

1. Support is explicit by `(Artifact kind, schema revision)`.
2. Unknown revisions and kind/schema mismatches fail closed.
3. There is no implicit field-based fallback and no legacy compatibility
   path.
4. Repeated Lineage Invariants must agree exactly.
5. Aggregate Metric Facts are eager; Sample Metric Facts are streamed only
   when requested.
6. Artifact Interpretation is not persisted.
7. Recipes cannot access raw manifests, summaries, records, or ancestors.
8. Preflight covers every direct and transitive Artifact, aggregates all
   issues, and produces no partial result.
9. Interpretation validates arithmetic and population consistency but never
   repeats model, detector, embedding, generation, or PPL computation.
10. Metric Facts use one small, constrained shape rather than a class per
    metric.
11. `ArtifactCatalog` is removed from the caller interface.
12. Version 1 uses one global `artifact-interpretation-v1` revision.
13. The cutover is atomic across all six Result recipes.
14. Old tests tied to raw JSON shapes are replaced rather than retained.

## Deep Module Shape

Create one package with a single public seam:

```text
src/watermark_suite/experiments/artifact_interpretation/
  __init__.py       public interface and exported domain values
  _core.py          lineage traversal, preflight, invariants, streaming
  _schemas.py       private exact kind/schema adapter registry
```

`__init__.py` exports only:

- `ARTIFACT_INTERPRETATION_REVISION`;
- the interpretation entry point;
- immutable `ArtifactInterpretation`, `ArtifactRelation`, `MetricFact`,
  `ExactCount`, `Distribution`, and `InterpretationSet` values;
- one aggregated interpretation error and its structured issues.

The implementation may use additional private helpers, but callers and tests
must cross this one seam.

### Normalized values

`ArtifactInterpretation` contains:

- Artifact Identity, kind, schema revision, and root/non-root role;
- normalized scientific dimensions;
- direct and transitive Artifact Relations;
- eager aggregate Metric Facts;
- record and population counts needed for validation.

`MetricFact` has one constrained envelope:

```text
metric name
scope: aggregate | sample
value
scientific dimensions
optional ExactCount
optional Distribution
source Artifact Identity
optional Sample identity
```

`Distribution` normalizes the currently inconsistent producer summaries into
one fixed value with count, mean, median, standard deviation, min/max, and
available percentiles. Missing optional statistics remain explicit; unknown
distribution fields are rejected by the relevant schema adapter.

`ArtifactRelation` states identities, direct/transitive status, lineage path,
and the kind/schema roles along that path. Recipes use these relations to
choose joins without reading manifests.

## Supported Schema Adapters

Version 1 supports exactly the current schema pairs:

| Artifact kind | Schema revision | Interpretation responsibilities |
| --- | --- | --- |
| `generation` | `generated-text-v3` | Population, generation model, decoding, prompt, seed, repetitions, Scheme identity |
| `adaptive-forgery` | `adaptive-forgery-v2` | Generation/forgery provenance, aggregate cost and green-selection facts, streamed per-Sample forge facts |
| `robustness` | `robustness-text-v2` | Transformation identity, inherited generation/Scheme facts, length and remote-call aggregate facts, streamed transformed-Sample facts |
| `detection` | `watermark-detection-v3` | Exact detection counts, token milestones, p-value/green distributions, adaptive-forgery curve facts, streamed per-Sample detection facts |
| `perplexity` | `conditional-perplexity-v2` | Evaluation model, conditional PPL, token/quality distributions, cost correlations, streamed per-Sample quality facts |
| `text-evaluation` | `text-evaluation-v1` | Embedding model, similarity and diversity facts, streamed per-Sample similarity facts where present |
| `downstream` | `gsm8k-evaluation-v2` | Task/model/prompt policy, exact accuracy count, streamed per-Sample correctness facts |
| `downstream` | `humaneval-evaluation-v2` | Task/model/prompt policy, exact pass@1 count, streamed per-Sample pass facts |

No adapter accepts another revision by structural resemblance.

## Lineage Rules

Preflight first resolves and verifies the complete transitive graph through the
Experiment Workspace. It then checks:

- every identity exists and passes Artifact file integrity verification;
- the graph is acyclic;
- every `(kind, schema revision)` has an exact adapter;
- each manifest kind agrees with its schema adapter;
- every direct source identity agrees between the ledger and manifest;
- repeated Lineage Invariants agree exactly;
- each Artifact receives unambiguous generation, population, Scheme,
  transformation, and task roles where its schema requires them;
- direct and transitive Artifact Relations have deterministic ordering.

Lineage Invariants include:

- Sample population identity;
- generation model, decoding policy, prompt revision, seed, and repetitions;
- Watermark Scheme identity;
- transformation identity after transformation;
- task identity for downstream evaluation.

Evaluation model, threshold, token milestone, and metric values are
Artifact-local facts and do not override those invariants.

## Numeric Validation Rules

Always validate:

- finite values where the metric requires finiteness;
- probability/rate values in `[0, 1]`;
- p-values in `[0, 1]` and target FPR values in `(0, 1)`;
- non-negative token lengths and query counts;
- `0 <= positive_num <= sample_num`;
- stored rates equal their exact count ratio within one documented tolerance;
- eligible counts do not exceed the interpreted population;
- distribution counts agree with their represented population;
- manifest record count, summary Sample count, and aggregate-fact population
  are compatible.

When Sample Metric Facts are requested, also validate Sample identity
uniqueness, Sample count, and cheap aggregate consistency. Do not recompute
model-derived values.

## Aggregated Diagnostics

Preflight visits all reachable Artifacts before returning. Each issue records:

- root Artifact Identity;
- offending Artifact Identity;
- lineage path;
- issue code;
- expected and observed values;
- concise human-readable message.

An invalid input set raises one aggregated error containing every issue.
`result-aggregation` writes no checkpoint and no partial report records for a
failed preflight.

## Implementation Phases

### Phase 0 — Freeze the new scientific contract

Files:

- `CONTEXT.md`
- this plan
- new `tests/test_artifact_interpretation.py`

Tasks:

1. Add test builders for current immutable Artifact directories and ledger
   registrations.
2. Write failing contract tests for the public interpretation seam and all
   locked decisions.
3. Cover current successful schema shapes before deleting old recipe fixtures.

Exit criteria:

- Tests express interpretation results without importing
  `ArtifactCatalog`, `ArtifactView`, or raw recipe helpers.

### Phase 1 — Implement values, registry, and full preflight

Files:

- new
  `src/watermark_suite/experiments/artifact_interpretation/__init__.py`
- new
  `src/watermark_suite/experiments/artifact_interpretation/_core.py`
- new
  `src/watermark_suite/experiments/artifact_interpretation/_schemas.py`
- `src/watermark_suite/experiments/errors.py`

Tasks:

1. Add the immutable interpretation values and global revision.
2. Add the exact kind/schema registry.
3. Resolve the complete lineage graph through `ExperimentWorkspace`.
4. Validate cycles, missing Artifacts, kind/schema pairs, relation agreement,
   and Lineage Invariants.
5. Aggregate issues in deterministic root/path/code order.
6. Return interpretations only after the whole graph passes.

Tests:

- every supported kind/schema pair;
- unknown direct and transitive revisions;
- kind/schema mismatch;
- missing ancestor;
- cycle;
- ledger/manifest source mismatch;
- one and multiple Lineage Invariant conflicts;
- deterministic multi-error diagnostics.

Exit criteria:

- A malformed graph cannot yield any interpretation values.

### Phase 2 — Normalize aggregate and Sample Metric Facts

Files:

- Artifact Interpretation package
- stage schema contract fixtures in tests

Tasks:

1. Implement aggregate extraction for all supported schemas.
2. Normalize count and distribution shapes.
3. Implement numeric and population consistency checks.
4. Implement filtered, streaming Sample Metric Facts without calling
   `artifact_records` or loading the complete JSONL file.
5. Keep generation and robustness text payloads out of Metric Facts unless a
   requested scientific fact needs them.

Tests:

- golden aggregate facts for all schemas;
- exact count/rate checks;
- inconsistent record/summary counts;
- invalid probability, p-value, token, and query values;
- requested Sample metrics only;
- Sample identity uniqueness;
- streaming behavior on a multi-record fixture;
- no model/runtime dependency is constructed.

Exit criteria:

- Every currently reported scientific value can be obtained through the new
  seam.

### Phase 3 — Atomically cut over all Result recipes

Files:

- `src/watermark_suite/experiments/result_assembly.py`
- `src/watermark_suite/experiments/stages/result_aggregation.py`
- `tests/test_result_assembly.py`
- `tests/test_stage_adapter_contracts.py`

Tasks:

1. Have the aggregation stage construct one Interpretation Set from its
   ordered source Artifacts.
2. Include `artifact-interpretation-v1` in the aggregation adapter revision
   used by Run Identity.
3. Rewrite all six recipes to consume interpretations, Metric Facts, and
   Artifact Relations only.
4. Preserve exact recipe behavior:
   - TPR/token effective thresholds and Wilson intervals;
   - TPR/PPL exact detection/PPL pairing and unwatermarked baseline;
   - downstream accuracy/pass@1;
   - clean/transformed robustness and similarity pairing;
   - forge cost, p-value, green ratio, ASR, and PPL;
   - diversity metrics.
5. Preserve deterministic report rows and complete source accounting.
6. Remove `ArtifactCatalog`, `ArtifactView`, raw summary parsing, raw manifest
   parsing, and the global known-schema whitelist from `result_assembly.py`.
7. Replace internal-shape recipe tests with interpretation fixtures and six
   stage-level integration tests.

Non-goals:

- Do not redesign the private Result recipe registry in this phase.
- Do not change renderer presentation.
- Do not change producer Artifact schemas.

Exit criteria:

- `result_assembly.py` has no access to Artifact paths, manifests, summaries,
  records, ancestors, or `ExperimentWorkspace`.
- All six recipes produce the same scientific report facts from valid current
  Artifacts.

### Phase 4 — Delete the old seam and document the new one

Files:

- `docs/experiment-runs.md`
- `docs/evaluation-aggregation-plan.md`
- `README.md`
- affected tests

Tasks:

1. Remove all imports and construction of `ArtifactCatalog` and
   `ArtifactView`.
2. Remove old raw JSON fixture helpers used only by recipes.
3. Document exact schema support, fail-closed behavior, global interpretation
   revision, aggregate preflight diagnostics, and Sample streaming.
4. Update the aggregation design record to identify Artifact Interpretation
   as the internal scientific seam.

Exit criteria:

- Repository search finds no public old interpretation path or compatibility
  fallback.

## Verification Gates

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src/watermark_suite tests analysis
git diff --check
```

Add explicit repository checks:

```bash
! rg "ArtifactCatalog|ArtifactView" \
  src/watermark_suite/experiments tests

! rg "\\.manifest|\\.summary|records\\.jsonl|summary\\.json" \
  src/watermark_suite/experiments/result_assembly.py
```

The implementation is complete only when:

1. All current direct and transitive schemas pass through explicit adapters.
2. Unknown or inconsistent inputs produce one deterministic aggregated
   diagnostic and no report records.
3. Every recipe reads only normalized interpretations.
4. Aggregate reports do not scan Sample records.
5. Sample facts stream without loading full Artifact records.
6. Artifact Interpretation revision changes invalidate only aggregation Runs,
   not expensive upstream Runs.
7. All six report recipes pass CPU-only integration tests.
8. The full existing suite and static verification gates pass.
