# Watermark Experiments

This context describes the experimental language used to generate, transform,
detect, and evaluate watermarked text.

## Language

**Run**:
A single, independently executable experiment stage whose inputs and every
result-affecting setting are fully resolved. Changing any result-affecting
setting for that stage defines a different Run.
_Avoid_: Job, configuration, experiment

**Run Identity**:
The stable identity determined by a Run's exact settings and semantic inputs,
including source Artifacts, dataset Sample selection, and model, tokenizer,
prompt, and code revisions.
_Avoid_: Filename, configuration name

**Attempt**:
One actual execution of a Run. An interrupted Attempt may resume, while starting
the same Run again from the beginning creates a new Attempt.
_Avoid_: Run, retry

**Attempt Provenance**:
The execution environment and observed resource record for an Attempt. Attempt
Provenance does not change the Run Identity.
_Avoid_: Run settings

**Artifact**:
A finalized, immutable result bundle produced by one successful Attempt. A
derived Artifact retains the identity of its source Artifact rather than
modifying it.
_Avoid_: Output file, result file

**Artifact Identity**:
The identity assigned when an Artifact is finalized, derived from its producing
Run, Attempt, schema revision, and content digests.
_Avoid_: Filename, expected path

**Canonical Artifact**:
The first finalized Artifact selected for a Run's default reuse. Artifacts from
later successful Attempts remain available but require explicit selection.
_Avoid_: Latest Artifact

**Sample**:
A stable dataset item selected for a Run, identified independently of its row
position or processing order.
_Avoid_: Row, example

**Sweep**:
A collection of Runs formed by varying settings for comparison. A Sweep groups
Runs but does not change the identity or meaning of any member Run.
_Avoid_: Batch

**Experiment Plan**:
A description of stage-level Runs, Sweeps, and the Artifact dependencies between
them. Repeated references to the same Run retain one Run Identity.
_Avoid_: Run, shell script

**Experiment Workspace**:
The managed local home of Run and Attempt history, Artifacts, resolved Experiment
Plans, and execution logs.
_Avoid_: Output directory
