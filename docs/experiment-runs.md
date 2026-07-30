# Running watermark experiments

The `wmexp` interface resolves generation, adaptive forgery, detection,
conditional-perplexity, and downstream-evaluation stages into immutable,
reproducible Runs. The example Plan at
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
revisions, hashes the selected dataset and seed, expands Sweeps, shows the
materialized setting value and source for every Run, and displays the
deterministic execution order. It reports dirty execution code but does not
reject it.

Execute or resume the Plan:

```bash
wmexp run experiments/plans/qwen25-main-and-forgery.yaml
```

Formal execution requires committed code under `src/watermark_suite`,
`voprf-py`, and the build files. Plan, data, and documentation changes may be
uncommitted because their relevant contents are resolved into Run Identity.
An interrupted process resumes the same Attempt at its last complete batch.
A caught stage failure is recorded as failed. By default execution stops at
the first failure; `--keep-going` continues independent Runs and still exits
nonzero.

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
override result-affecting settings. Detection derives its tokenizer and
watermark settings from its source Artifact. Perplexity always uses the
evaluation model's tokenizer. The downstream stage fixes GSM8K to its full
official 4-shot test set and HumanEval to its full official 0-shot set; its
Artifacts retain each answer or code completion and correctness result.

The paper-result Plans currently live at:

```text
experiments/plans/figure-tpr-vs-token-length.yaml
experiments/plans/figure-tpr-vs-ppl.yaml
experiments/plans/table-downstream-performance.yaml
```

The first two Plans intentionally share identical generation semantics for
their common schemes, so the second reuses canonical generation Artifacts
created by the first.

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
