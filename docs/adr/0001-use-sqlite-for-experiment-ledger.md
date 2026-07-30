# Use SQLite for the experiment ledger

The Experiment Run module needs crash-safe Run, Attempt, checkpoint, and Artifact
lineage updates on a single machine. We will use SQLite for this transactional
ledger because it provides atomic state changes and indexed status queries without
an external database; large records, summaries, and traces remain immutable file
Artifacts, so SQLite does not become the experiment data store.
