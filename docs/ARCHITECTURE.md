# Autoresearch Architecture

## Purpose
Autoresearch runs bounded, repeatable ML experiments where code edits are proposed by an LLM and validated by short training runs. The system is designed for overnight operation with persistent state and explicit stop/restart behavior.

## Highest-change modules

### train.py
- Role: Experiment surface for model architecture and training-loop edits.
- Why high-change: This is intentionally the primary mutable file during autonomous experimentation.
- Guardrail: Keep each proposal narrow and measurable so result deltas remain attributable.

### orchestrator.py
- Role: Core execution engine that applies proposals, runs training, parses metrics, and records keep or discard outcomes.
- Inputs: LLM proposal, previous results, runtime profile overrides, GPU reservation state.
- Outputs: Updated results.tsv, run.log, state.json entries.
- Guardrail: Fail fast and preserve state on errors; never silently drop failed runs.

### supervisor.py
- Role: Nightly coordinator that starts batch runs, checks progress, detects plateaus, and generates reports.
- Inputs: API status, results history, Research Director directives.
- Outputs: Batch lifecycle decisions and markdown report artifacts.
- Guardrail: If API is unavailable or plateau threshold is reached, stop safely and report.

### report.py
- Role: Converts experiment history into operator-readable summaries and trend analysis.
- Inputs: results.tsv rows and round directives.
- Outputs: Timestamped report markdown in reports/.
- Guardrail: Report generation must never crash on partial or malformed rows.

## Runtime flow
1. supervisor.py checks API health and current orchestrator status.
2. supervisor.py requests a directive and starts a bounded batch.
3. api.py starts orchestrator.py in a background thread.
4. orchestrator.py applies one proposal at a time and executes training.
5. Results are persisted and the best kept metric is updated.
6. supervisor.py detects completion or plateau and writes a report.

## Reliability boundaries
- State persistence: state.json is the source of continuity across nightly runs.
- Result ledger: results.tsv is append-only history for keep or discard decisions.
- Safety stop: stop event allows graceful interruption without data loss.
- Infrastructure decoupling: API control plane is separate from model-training execution.
