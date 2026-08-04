# Running watermark experiments

The `wmexp` interface resolves generation, adaptive forgery, robustness,
detection, conditional-perplexity, text-evaluation, and downstream-evaluation
stages into immutable, reproducible Runs. The example Plan at
`experiments/plans/qwen25-main-and-forgery.yaml` uses Qwen2.5-7B for generation
and forgery and Qwen2.5-14B for perplexity.

## Workflow

Install the project, cache the model revisions and benchmark datasets locally,
and check the Plan:

```bash
uv sync
wmexp check experiments/plans/qwen25-main-and-forgery.yaml
```

`check` does not load a model or use a GPU. It resolves cached Hugging Face
revisions, hashes local model snapshots, selected datasets, seeds, and
external detector/key material, expands Sweeps, shows the materialized setting
value and source for every Run, and displays the deterministic execution
order. It reports dirty execution code but does not reject it.

Execute or resume the Plan:

```bash
wmexp run experiments/plans/qwen25-main-and-forgery.yaml
```

Formal execution requires committed code under `src/watermark_suite`,
`voprf-py`, and the build files. Plan, data, and documentation changes may be
uncommitted because their relevant contents are resolved into Run Identity.
An interrupted process or caught stage failure resumes the same Attempt at its
last complete batch on the next invocation. The executor holds single-machine
Run and Attempt leases while it works, so a second process cannot select or
resume the same work concurrently. By default execution stops at the first
failure; `--keep-going` continues independent Runs and still exits nonzero.
Stages skipped because of a failed dependency are persisted as `blocked` in
SQLite; independent stages not run because of fail-fast are persisted as
`skipped`. Both are reported by `status`.

The Resolved Experiment Plan owns one canonical Plan Execution Order. Both
`check` and `run` consume that order from the same in-process graph module.
Canonical Artifact reuse and resumed or new Attempts do not change it. With
`--keep-going`, failure removes blocked descendants but does not reorder
independent Runs: actual execution remains a subsequence of the canonical
order. Graph traversal state is immutable and in-memory; Workspace Run,
Attempt, Artifact, and checkpoint records remain the restart authority.

The first successful Artifact for a Run is canonical. A repeat invocation
reuses it. To deliberately execute the same Run again:

```bash
wmexp run PLAN.yaml --new-attempt forge_vow
```

The new Artifact is preserved, but it does not silently replace the canonical
Artifact. Inspect the ledger with either a Plan path or the digest printed by
`check`:

```bash
wmexp status PLAN.yaml
wmexp status plan_SHA256_DIGEST --json
```

`status` does not change Attempt state and lists every stage in the resolved
Plan, including stages that have not started.

Inspect stored failures without querying SQLite directly:

```bash
wmexp errors plan_SHA256_DIGEST
wmexp errors plan_SHA256_DIGEST --stage generate_pdw --traceback
wmexp errors plan_SHA256_DIGEST --attempt attempt_ID --json
```

Prefer the registered Plan digest when the checkout has changed since the Run
started. A Plan path is resolved against the current code revision.

Use `--workspace PATH` on any command to replace the default
`output/experiments` Experiment Workspace.

## Experiment Workspace

```text
output/experiments/
  ledger.sqlite
  artifacts/
    artifact_<identity>/
      manifest.json
      records.jsonl
      summary.json
  attempts/
    attempt_<identity>/
      chunks/
      execution.lock
  locks/
  resolved-plans/
  logs/
```

SQLite stores only transactional Run, Attempt, checkpoint, Artifact, and
lineage metadata. Large records and traces remain in immutable file Artifacts.
Artifact Identity is assigned during finalization from the Run, Attempt,
schema revision, and record/summary digests.

## Plan rules

Plan defaults apply only to stage kinds that explicitly accept the setting.
Resolution order is:

```text
Plan defaults -> stage settings -> Sweep assignment
```

Unknown settings, unused defaults, dependency cycles, duplicate Sample
identities, and excessive Sweep expansion are errors. CLI flags cannot
override result-affecting settings. Device and dtype are part of Run Identity.
The singular `input` field maps a downstream stage over every resolved
upstream instance. The plural `inputs` field collects every resolved instance
of the named stages into one downstream Run; it never creates an implicit
Cartesian product.

Run Identity uses stage semantic/runtime settings, ordered source Artifacts,
adapter revision, and Artifact schema revision. The repository Git revision is
retained in Plan, Attempt, and Artifact provenance but does not invalidate an
otherwise unchanged Run. Result-affecting implementation changes must bump
the affected adapter revision; changes to shared execution modules must bump
every consuming adapter.

Dataset and prompt handling form one `dataset-prompt-v1` module. Plan
resolution turns a dataset alias or task into a Prompt Population containing
the verified dataset snapshot, ordered Sample identities, a content-identified
Prompt Policy, and prompt-related generation controls. Execution materializes
that exact population and re-verifies its snapshot and Sample manifest before
rendering prompts. Generation, adaptive forgery, and downstream callers receive
only typed Prompt Samples; they do not load benchmark data, apply chat
templates, construct task demonstrations, or assign repetition identities.
The four adapters are C4, ELI5, GSM8K, and HumanEval. GSM8K and HumanEval always
resolve their complete official test populations.

Local model directories use a verified content snapshot; cached Hugging Face
models retain their resolved commit, and model/tokenizer plus PDW/UPV material
is checked again before use. Detection derives its tokenizer and watermark
settings from its source Artifact. Perplexity always uses the evaluation
model's tokenizer. The downstream stage fixes GSM8K to its full official
4-shot test set and HumanEval to its full official 0-shot set; its Artifacts
retain each answer or code completion and correctness result.

Watermark validation, material verification, and paired generator/detector
construction live in one scheme registry. Experiment stages select a scheme
through that registry instead of reproducing method-specific branches.

Detection operating points are explicit Artifact semantics. Detectors with
calibrated p-values retain their requested target FPR. UPV does not expose a
p-value: its private classifier uses the fixed rule
`classifier_confidence > 0.5`. The checked-in legacy calibration records an
empirical FPR of `7920 / 1,000,000 = 0.00792` on 255-token C4 chunks. UPV
report rows therefore leave `target_fpr` empty and carry the classifier rule,
empirical FPR, exact calibration count, calibration token length, material
digest, and calibration provenance. Generic `significance_levels` are removed
from the semantic settings of a detection Run once it is bound to UPV.

The robustness stage never edits its source Artifact. It emits
`original_text`, `transformed_text`, transformation provenance, and length
statistics in a new Artifact. Detection and text evaluation can independently
consume that Artifact, so either evaluation can be retried without repeating
the transformation. The robustness Plan also detects each clean generation
Artifact, so clean and transformed milestone-detection curves are both
available without regeneration. Supported transformations are deterministic
word deletion and OpenAI Responses API paraphrasing.

Set `OPENAI_API_KEY` before running the robustness Plan. Its paraphrase sweep
pins GPT-3.5 to the formal snapshot ID `gpt-3.5-turbo-0125` and requests
`gpt-5.6-sol`. GPT-3.5 receives no reasoning field because it is not a
reasoning model; Sol receives `reasoning.effort: low`. The provider currently
exposes no dated Sol snapshot, so Sol transformations are reproducible at the
request and recorded-provenance level, not at the model-weight level. Both
preserve the prior paraphrase sampling temperature of `0.7`. Each of the seven
watermark configurations generates 1,000 source texts, so the two remote
paraphrase treatments issue 14,000 API requests in a complete run. Each record
retains the requested and returned model, response ID and status, latency,
service tier, and
input/output/cache/reasoning token usage. The stage summary aggregates those
values. Calls within one stage batch run concurrently, and the batch is
checkpointed only after all its calls succeed.

The text-evaluation stage has one versioned result shape for cosine similarity
and diversity. Diversity generation uses `repetitions` to retain a stable
source-prompt group while assigning every sampled continuation its own Sample
identity. Aggregate diversity includes distinct-1/2/3, self-BLEU-4, and Vendi
score; similarity retains per-sample cosine scores and distribution summaries.

Each paper Plan ends in a `result-aggregation` stage. It validates all source
schemas and transitive lineage, rejects incomparable sample populations or
ambiguous joins, and emits an `experiment-report-v2` Artifact. Its
`records.jsonl` is a deterministic long-form metric table with scientific
dimensions, exact numerator/denominator counts, uncertainty metadata, and
source Artifact identities. Supported recipes are TPR/token, TPR/PPL,
downstream performance, robustness, adaptive forgery, and diversity.

Before a recipe runs, the in-process `artifact-interpretation-v2` module
preflights every root Artifact and transitive ancestor. Support is exact by
Artifact kind and schema revision; unknown revisions, kind/schema mismatches,
missing lineage, conflicting inherited scientific provenance, and cheap
count/rate inconsistencies fail closed in one aggregated diagnostic. The
module normalizes lineage into scientific dimensions, Artifact Relations,
and aggregate Metric Facts. Recipes cannot read raw manifests, summaries, or
Sample records. Per-Sample Metric Facts are available through a filtered
streaming interface and are validated only when requested; aggregate report
assembly therefore does not scan `records.jsonl`.

Producer schemas `generated-text-v3`, `adaptive-forgery-v2`,
`gsm8k-evaluation-v2`, and `humaneval-evaluation-v2` expose the resolved Prompt
Population as the single population/prompt provenance seam. Artifact
Interpretation derives both population identity and generation dimensions from
that value; the superseded producer revisions are intentionally unsupported.
Detection uses `watermark-detection-v4`, which distinguishes calibrated
p-value thresholds from empirical classifier operating points.

Report rendering is downstream of the immutable metric Artifact:

```bash
python analysis/render_report.py \
  output/experiments/artifacts/artifact_REPORT \
  --csv output/reports/tpr-token.csv \
  --figure output/reports/tpr-token.pdf

python analysis/render_report.py \
  artifact_REPORT \
  --workspace output/experiments \
  --markdown output/reports/downstream.md
```

TPR/PPL report exports always retain PDW. Its point is omitted from the
primary figure by default because of the plot scale; pass `--include-pdw` to
render it as well.

The migrated plotting entry points under `analysis/plot/` accept the same
Artifact, Workspace, and output options. They do not discover results from
model-specific filenames or contain hard-coded metric values.

The paper-result Plans currently live at:

```text
experiments/plans/figure-tpr-vs-token-length.yaml
experiments/plans/figure-tpr-vs-ppl.yaml
experiments/plans/table-downstream-performance.yaml
experiments/plans/adaptive-forgery-qwen25-7b.yaml
experiments/plans/robustness-qwen25-7b-instruct.yaml
experiments/plans/diversity-qwen25-7b.yaml
```

The first two Plans intentionally share identical generation semantics for
their common schemes, so the second reuses canonical generation Artifacts
created by the first.

The adaptive-forgery Plan creates 1,000 texts and evaluates them every ten
tokens. Its detection summary contains overall p-value and green-token
distributions plus an `adaptive_forgery_curve` array. Each curve row records
eligible sample count, mean/median cumulative oracle queries, queries per
token, selected-green count/ratio, median p-value, and attack success rate at
each configured detector threshold. This is the direct plotting input for
token length versus attack cost and token length versus ASR.

GPU stages run sequentially on one machine. Ready independent Runs may be
reordered by model resource key to avoid unnecessary reloads; `wmexp check`
shows that order. There is no scheduler integration in this interface.

## Legacy result archive

Old output layouts are intentionally not imported into `wmexp`. Archive them
with the separate, non-destructive script:

```bash
python scripts/archive_legacy_outputs.py \
  /home/lxk/projects/voprf-legacy-archive
```

It copies `output` and `data/legacy` while preserving repository-relative
paths and timestamps, verifies SHA-256 and size, and publishes
`archive-manifest.json` atomically. Re-running verifies existing copies. It
never deletes a source file and refuses to overwrite a differing archive
file.
