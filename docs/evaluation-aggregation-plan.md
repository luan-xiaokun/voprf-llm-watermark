# Evaluation Aggregation and Result Assembly Plan

Status: implemented; retained as the design and verification record.

## Outcome

Extend the Experiment Run architecture with one deep Result Assembly module
that turns compatible evaluation Artifacts into a versioned, tidy report
Artifact. The first implementation must cover the paper-facing TPR,
perplexity, downstream, robustness, adaptive-forgery, and diversity results
without hard-coded result paths or manually copied numbers.

The module's external interface is deliberately small:

```text
ordered source Artifacts + recipe + recipe parameters
    -> experiment-report-v2 Artifact
```

The implementation behind that interface owns schema validation, transitive
lineage inspection, scientific comparability checks, joins, metric
aggregation, uncertainty metadata, and deterministic output ordering.

## Current State and Evidence

The current architecture already provides most of the low-level mechanics:

- `StageExecutionContext.inputs`, `bind_inputs`, Run identity construction,
  and Artifact manifests represent sources as tuples/lists.
- Artifact manifests retain direct source identities and resolved semantic
  settings.
- Detection, conditional perplexity, text evaluation, downstream evaluation,
  robustness, and adaptive forgery already emit immutable records and
  summaries.
- The ledger can resolve and verify an Artifact by identity.

Four gaps prevent result assembly:

1. `plan.py`, `PreparedStage`, and `engine.py` reduce the tuple to one
   `input_instance`, so a Plan cannot express fan-in.
2. Each evaluation schema uses a different summary shape. No module presents
   them as comparable metric observations.
3. Scientific dimensions such as dataset population, decoding policy,
   watermark parameters, transformation, task, token milestone, and FPR are
   distributed across an Artifact and its ancestors.
4. Current plotting scripts hard-code legacy paths and, in some cases,
   hard-code the plotted metric values themselves.

There is also a reuse blocker that should be fixed before the large server
rerun: Run identity currently includes the repository-wide Git revision.
Adding a report stage in a later commit would therefore invalidate otherwise
unchanged generation and evaluation Runs. Git revision is valuable
provenance, but it is too broad to be a result identity input.

## Scope

### In scope

- Unrelated implementation commits must not invalidate unchanged stage Runs.
- A Plan can collect all resolved instances of several upstream stages into
  one downstream Run.
- Result assembly validates and joins multiple Artifact schemas.
- Report Artifacts use one stable long-form metric-row schema.
- Existing paper Plans end with report stages.
- New render/export entry points consume report Artifacts without hard-coded
  experiment paths.

### Out of scope

- A general SQL, dataframe, JSONPath, or expression language in Plan YAML.
- A distributed scheduler or multi-machine Artifact store.
- Importing legacy result directories into the main CLI.
- Rewriting sample-level detection, PPL, downstream, or transformation
  execution.
- Matplotlib styling inside the Experiment Run module.
- Silently coercing or dropping incompatible Artifacts.

## Design Decisions

### 1. Scope implementation identity to a stage

Run identity will use:

- stage kind;
- adapter revision;
- Artifact schema revision;
- result-affecting semantic/runtime settings;
- ordered source Artifact identities.

The repository Git revision remains in the resolved Plan, Attempt provenance,
and Artifact manifest, but is removed from Run identity. A result-affecting
implementation change must bump the affected adapter revision. This is a
one-time Run identity schema change and must land before the main rerun.
Version 1 deliberately keeps the existing explicit adapter-revision policy;
it does not introduce a source-file dependency graph or infer revisions from
Python imports. Shared result-affecting changes, such as a scheme-registry
change, must bump every consuming adapter revision.

Required invariant:

> Adding or changing only a renderer, report recipe, documentation, or an
> unrelated stage adapter does not change an existing generation/detection
> Run identity.

### 2. Add collection fan-in, not Cartesian fan-in

Keep the existing singular `input` behavior: one downstream instance is
created for each resolved upstream instance.

Add mutually exclusive plural `inputs` behavior: one downstream instance
collects every resolved instance of the named stages, in the declared stage
order and then Plan ordinal order.

```yaml
assemble_tpr_ppl:
  kind: result-aggregation
  inputs:
    - detect_watermarked
    - evaluate_watermarked_ppl
    - evaluate_unwatermarked_ppl
  recipe: tpr-vs-ppl
  confidence_level: 0.95
  threshold_policy:
    default: 0.00001
    rdf: 0.01
```

`input` and `inputs` cannot appear together. Empty lists, duplicate stage
names, self-reference, unknown stages, and cycles are Plan errors. The
executor starts a collection Run only when every collected input has
succeeded. Failure of any source blocks it.

This avoids an implicit cross product and keeps the Plan interface
predictable.

### 3. Normalize sources behind Artifact Interpretation

The initial internal catalog design was deepened into the read-only
`artifact-interpretation-v2` module described in
`docs/artifact-interpretation-plan.md`. It loads verified Artifacts, follows
complete transitive lineage, validates exact kind/schema support and Lineage
Invariants, and produces an `ArtifactInterpretation` containing:

```text
identity and schema
direct and transitive lineage
watermark identity and scientific parameters
dataset and selected-population identity
generation model and decoding policy
evaluation model
transformation
task
normalized aggregate Metric Facts and streamed Sample Metric Facts
```

Artifact Interpretation is the sole internal scientific seam. Result recipes
cannot access raw paths, manifests, summaries, records, or ancestors, and
Plan authors do not describe JSON paths. Watermark dimension normalization
belongs in the existing scheme registry so Result Assembly does not recreate
scheme-specific branch forests.

Detection Metric Facts carry an explicit operating point. Calibrated detectors
use p-value thresholds and a `target_fpr` axis. UPV instead uses its native
`classifier_confidence > 0.5` decision; its report rows have no target-FPR
claim and include the independently measured empirical FPR, exact calibration
count, and calibration provenance.

Joins prefer exact lineage over equality of labels:

- PPL is paired with the detection Artifact in its direct lineage.
- Robustness detection and similarity are paired through their common
  transformed-text Artifact.
- Clean and transformed curves are distinguished by their generation or
  transformation ancestor.
- Independent downstream Runs are grouped by task and canonical watermark
  identity.

### 4. Use recipe adapters behind one result-aggregation stage

The Plan-facing stage kind is `result-aggregation`. A private recipe registry
contains the scientific logic. Version 1 provides:

- `tpr-vs-token-length`
- `tpr-vs-ppl`
- `downstream-performance`
- `robustness`
- `adaptive-forgery`
- `diversity`

Recipes are preferable to a generic metric DSL because each recipe can
enforce its own scientific invariants while callers learn only a recipe name
and a small number of parameters.

### 5. Emit a long-form, versioned report Artifact

`experiment-report-v2/records.jsonl` contains one metric observation per row:

```json
{
  "sample_id": "metric_<deterministic identity>",
  "recipe": "tpr-vs-ppl",
  "metric": "true_positive_rate",
  "value": 0.998,
  "dimensions": {
    "scheme": "vow",
    "watermark_parameters": {
      "window_size": 4,
      "gamma": 0.5,
      "delta": 2.5
    },
    "target_fpr": 0.00001,
    "token_num": null,
    "task": null,
    "transformation": null
  },
  "population": {
    "sample_num": 1000,
    "positive_num": 998
  },
  "uncertainty": {
    "method": "wilson",
    "confidence_level": 0.95,
    "lower": 0.9917,
    "upper": 0.9995
  },
  "source_artifacts": ["artifact_..."]
}
```

Rows are sorted deterministically by recipe-defined dimensions and metric.
`summary.json` records the recipe revision, all consumed source Artifacts,
compatibility checks, row count, and threshold policy. No source Artifact may
be silently omitted. PDW, for example, remains in the TPR/PPL report even if a
renderer omits it from the primary figure.

Proportion metrics use exact numerator/denominator counts and Wilson
intervals. Non-proportion metrics retain their source distribution metadata;
bootstrap intervals are deferred until a concrete statistical requirement is
agreed.

### 6. Keep rendering downstream of the report Artifact

Core result assembly produces scientific data, not publication styling.
Renderer/export commands take a report Artifact path or identity and produce
CSV or figures. They never discover experiments through filenames.

The report Artifact is the source of truth; CSV/PDF files are reproducible
views of it.

## Implementation Phases

### Phase 0 — Correct Run identity scope

Files:

- `src/watermark_suite/experiments/adapters.py`
- `src/watermark_suite/experiments/engine.py`
- `src/watermark_suite/experiments/models.py`
- `src/watermark_suite/experiments/plan.py`
- `docs/experiment-runs.md`
- `tests/test_experiment_runs.py`

Tasks:

1. Remove repository `code_revision` from the Run identity document.
2. Bump the Run identity schema.
3. Keep Git revision in Plan, Attempt, and Artifact provenance.
4. Document adapter revision as the result-affecting implementation version.
   Document that changes in shared execution modules require revision bumps
   in every consuming adapter.
5. Add a regression test proving an unrelated code revision change preserves
   an unchanged stage's Run identity.
6. Add a test proving an adapter revision change does change Run identity.

Exit criteria:

- A later report-only commit can reuse raw canonical Artifacts.
- Result-affecting stage changes remain explicit identity changes.

### Phase 1 — Add deterministic collection fan-in

Files:

- `src/watermark_suite/experiments/plan.py`
- `src/watermark_suite/experiments/models.py`
- `src/watermark_suite/experiments/engine.py`
- `src/watermark_suite/experiments/cli.py`
- `tests/test_experiment_runs.py`

Tasks:

1. Add plural `inputs` to the Plan control schema.
2. Replace `PreparedStage.input_instance` with an ordered
   `input_instances` tuple and migrate every internal caller.
3. Preserve singular `input` fan-out and implement plural `inputs` collect.
4. Update readiness, failure propagation, status, planned-order display, Run
   identity, and Artifact lineage to use all inputs.
5. Reject ambiguous or cyclic input definitions during Plan resolution.

Tests:

- collect two swept stages into exactly one downstream Run;
- deterministic source order;
- no Cartesian expansion;
- all-source readiness;
- blocking when one collected source fails;
- Run identity changes when any source Artifact changes;
- all existing singular-input Plans retain the same instance set and
  dependency semantics.

Exit criteria:

- A fake aggregation adapter can consume all expected source Artifacts through
  the normal `StageExecutionContext.inputs` interface.

### Phase 2 — Build Artifact Interpretation and strengthen counts

Files:

- new `src/watermark_suite/experiments/result_assembly.py`
- `src/watermark_suite/experiments/scheme_registry.py`
- `src/watermark_suite/experiments/stages/detection.py`
- `tests/test_experiment_evaluation.py`
- new `tests/test_result_assembly.py`

Tasks:

1. Implement verified, transitive lineage loading from the local Workspace.
2. Normalize watermark, dataset, model, task, and transformation dimensions.
3. Centralize canonical watermark analysis identity in the scheme registry.
4. Add explicit positive/eligible counts to detection-rate and milestone
   summaries; add success counts to adaptive-forgery curves.
5. Bump affected detection adapter/Artifact schema revisions.
6. Implement compatibility errors that name the conflicting Artifact
   identities and dimensions.

Compatibility gates:

- same dataset selection for comparative text-generation recipes;
- same generation model and decoding settings except declared experimental
  axes;
- requested FPR exists in the detection Artifact;
- unique lineage match for every joined metric;
- exactly one unwatermarked PPL baseline where required;
- known Artifact schema revisions only.

Exit criteria:

- Fixture Artifacts from all current evaluation schemas normalize into stable
  `ArtifactInterpretation`, `ArtifactRelation`, and `MetricFact` values.
- Incompatible inputs fail before report records are written.

### Phase 3 — Implement Result Assembly recipes

Files:

- new
  `src/watermark_suite/experiments/stages/result_aggregation.py`
- `src/watermark_suite/experiments/stages/__init__.py`
- `src/watermark_suite/experiments/result_assembly.py`
- `tests/test_result_assembly.py`
- `tests/test_stage_adapter_contracts.py`

Recipe contracts:

1. `tpr-vs-token-length`
   - one row per scheme, token milestone, and effective FPR;
   - uses RDF's configured FPR rather than silently substituting `1e-5`;
   - records eligible and detected counts.
2. `tpr-vs-ppl`
   - joins each watermarked detection with its PPL descendant;
   - emits TPR and conditional PPL using the same scheme dimensions;
   - emits the unwatermarked PPL baseline;
   - retains PDW in the report.
3. `downstream-performance`
   - one accuracy or pass@1 row per task and watermark configuration;
   - retains exact correct/passed and total counts.
4. `robustness`
   - joins clean detection, transformed detection, transformation metadata,
     and cosine similarity;
   - one row per scheme, transformation, token milestone, FPR, and metric.
5. `adaptive-forgery`
   - emits query cost, queries/token, median p-value, selected-green ratio,
     and ASR per token milestone.
6. `diversity`
   - emits distinct-1/2/3, self-BLEU-4, and Vendi score per watermark
     configuration.

Tests:

- golden long-form rows for every recipe;
- deterministic ordering and metric identities;
- exact lineage joins;
- duplicate/missing pair rejection;
- FPR policy validation;
- PDW retention;
- no silent source omission;
- report summary accounts for every direct input.

Exit criteria:

- All six recipes produce `experiment-report-v2` Artifacts using CPU-only
  tests.

### Phase 4 — Add report stages to the experiment Plans

Files:

- `experiments/plans/figure-tpr-vs-token-length.yaml`
- `experiments/plans/figure-tpr-vs-ppl.yaml`
- `experiments/plans/table-downstream-performance.yaml`
- `experiments/plans/robustness-qwen25-7b-instruct.yaml`
- `experiments/plans/adaptive-forgery-qwen25-7b.yaml`
- `experiments/plans/diversity-qwen25-7b.yaml`
- `tests/test_experiment_plans.py`

Tasks:

1. Add one aggregation stage to each Plan.
2. Encode threshold policies explicitly:
   default calibrated detection at `1e-5`, RDF at `1e-2`.
3. Assert source-role coverage and final report schema in Plan tests.
4. Verify shared raw Runs keep the same Run identity across Plans.

Expected stage counts after this phase:

- TPR/token: 11
- TPR/PPL: 30
- downstream: 17
- adaptive forgery: 4
- robustness: 78
- diversity: 15

Exit criteria:

- Running a completed Plan creates only its report Artifact on a report-only
  retry; generation/evaluation Artifacts are canonical reuses.

### Phase 5 — Replace hard-coded result assembly in renderers

Files:

- new `analysis/report_io.py`
- update or replace:
  - `analysis/plot/1_tpr_against_token_num.py`
  - `analysis/plot/2_tpr_against_ppl.py`
  - `analysis/plot/3_new_robustness.py`
- add downstream, adaptive-forgery, and diversity render/export entry points
- `docs/experiment-runs.md`

Tasks:

1. Load report Artifacts by explicit path or Workspace Artifact identity.
2. Export the long-form records to CSV without changing metric semantics.
3. Render the first two figures, downstream table, robustness figure,
   adaptive-forgery curves, and diversity table from report rows.
4. Keep plot inclusion/styling choices outside the report. In particular,
   primary-figure omission of PDW is a renderer option.
5. Remove hard-coded result values and model-specific result filenames from
   the migrated renderers.

Exit criteria:

- Renderer smoke tests consume fixture `experiment-report-v2` Artifacts.
- Changing an input path requires a CLI argument, not source editing.
- The report rows used for every plotted point can be traced to source
  Artifact identities.

## End-to-End Verification

The implementation is complete when all of the following hold:

1. Resolve a small Plan with two swept producers and one collection stage;
   the collection stage receives all producer Artifacts exactly once.
2. Execute raw stages, commit an unrelated report implementation change, and
   confirm raw Run identities remain reusable.
3. Produce all six report recipes from fixture Artifacts without loading a
   GPU model.
4. Reject mismatched datasets, decoding policies, missing baselines, missing
   thresholds, duplicate joins, and unknown schemas with actionable errors.
5. Run the full existing test suite plus the new fan-in and report tests.
6. Run `compileall`, `git diff --check`, Plan resolution tests, and CLI smoke
   tests.
7. Render fixture versions of the target figures/tables solely from report
   Artifacts.

## Delivery Order

Phases 0 and 1 are architectural prerequisites and should be reviewed
together before the server rerun. Phase 2 establishes the scientific
contract. Phase 3 can then implement recipes incrementally, starting with
TPR/token and TPR/PPL. Phase 4 should land before launching the corresponding
large Plans. Phase 5 can proceed after raw experiments start because it does
not affect Run identities.
