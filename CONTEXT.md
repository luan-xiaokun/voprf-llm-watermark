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

**Artifact Interpretation**:
The validated scientific meaning of one Artifact and its transitive lineage,
expressed as normalized provenance, dimensions, metric facts, and distributions.
It does not join independent Artifacts or choose comparative reporting policy.
_Avoid_: Raw summary, report recipe

**Lineage Invariant**:
A scientific fact inherited through an Artifact lineage that may be repeated
but never overridden; repeated declarations must agree exactly.
_Avoid_: Child override, first matching value

**Metric Fact**:
A normalized scientific observation from one Artifact Interpretation, scoped
either to an aggregate population or to one Sample.
_Avoid_: Raw record field, raw summary field, report row

**Artifact Relation**:
A normalized direct or transitive lineage relationship between identified
Artifacts. It states provenance structure without choosing a comparative join.
_Avoid_: Raw manifest link, report pairing

**Canonical Artifact**:
The first finalized Artifact selected for a Run's default reuse. Artifacts from
later successful Attempts remain available but require explicit selection.
_Avoid_: Latest Artifact

**Sample**:
A stable dataset item selected for a Run, identified independently of its row
position or processing order.
_Avoid_: Row, example

**Prompt Policy**:
A versioned, content-identified rule that renders one Sample into the source
prompt retained for analysis and the model prompt submitted for generation.
It owns task demonstrations, instructions, chat-template use, and stop strings.
_Avoid_: Prompt string, caller formatting

**Prompt Population**:
An immutable selected Sample population paired with its Prompt Policy. It owns
dataset snapshot verification, Sample identities, prompt rendering, and
repetition identities as one resolved scientific input to a Run.
_Avoid_: Dataset path, sample manifest

**Sweep**:
A collection of Runs formed by varying settings for comparison. A Sweep groups
Runs but does not change the identity or meaning of any member Run.
_Avoid_: Batch

**Experiment Plan**:
A description of stage-level Runs, Sweeps, and the Artifact dependencies between
them. Repeated references to the same Run retain one Run Identity.
_Avoid_: Run, shell script

**Resolved Experiment Plan**:
An Experiment Plan whose defaults and Sweeps have been expanded into uniquely
named, independently executable Runs with explicit Artifact dependencies.
_Avoid_: Raw plan, YAML plan

**Plan Execution Order**:
The deterministic order promised for the Runs of an Experiment Plan when their
dependencies succeed. Plan inspection and execution must report the same order.
_Avoid_: Preview order, incidental order

**Experiment Workspace**:
The managed local home of Run and Attempt history, Artifacts, resolved Experiment
Plans, and execution logs.
_Avoid_: Output directory
