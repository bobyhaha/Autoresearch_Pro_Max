# Architecture

Simplify Autoresearch v2 combines two loops: a provenance-first scientific library that
turns literature into claims, mechanisms, and hypotheses, and a resident experiment engine
that tests those hypotheses without placing reasoning in the GPU launch critical path.

## Authority flow

```text
agenda -> search -> sources -> claims -> mechanisms -> hypotheses
                                                   |
                                                   v
structured scope + hypothesis-bound proposal
            |
            v
     ExperimentSpec  -- immutable scientific intent and executable seeds
            |
            v
    ExecutionManifest -- sealed code, data, resources, runtime, exact argv/env
            |
            +------> deterministic operational queue job
                              |
                              v
                        ExecutionService
                              |
                              v
                         ResultBundle
                              |
                              v
                       EvidenceDecision
                              |
                 +------------+-------------+
                 |                          |
                 v                          v
          BANK/PROMOTION views       Knowledge/SOTA views
                                             |
                                             v
                                experiment-origin claims
                                             |
                                             v
                       SCIENTIFIC_BELIEFS / MECHANISMS / HYPOTHESES
```

Arrows point from an authority to a record or projection derived from it. Queue and view
files can be rebuilt or reconciled; they cannot rewrite a scientific record.

## Immutable records

The execution authority remains:

- `ExperimentSpec` preregisters the question, metric, plan, arms, executable replicate
  seeds, requirements, comparison group, structured scope, and search lane.
- `ExecutionManifest` digest-binds the spec, snapshots code/data into content-addressed
  blobs, and authorizes resources and runtime without changing scientific fields.
- `ResultBundle` records what actually ran: sealed identity, lifecycle, resource and launch
  telemetry, stdout/stderr artifacts, metrics, and failure state.
- `EvidenceDecision` links to a result digest and determines measurement validity from
  provenance, lifecycle, seed, required metrics, steps, and resource policy.
- `Paper` is an optional immutable publication record retained for compatibility and manual
  reporting. It is not part of the v2 execution hot path.

The scientific-library authority adds:

- `research_agenda`: scoped questions, repeatable queries, refresh intervals, and minimum
  source/claim coverage;
- `literature_search`: provider, exact query, time, filters, and returned source IDs;
- `literature_source`: stable work identity, bibliographic snapshot, abstract, topics, and
  an optional content-addressed full-text snapshot;
- `scientific_claim`: one attributed and scoped supporting/opposing assertion, locators,
  reported metrics, provenance, and evidence assessment;
- `scientific_mechanism`: claim-bound causal nodes and edges, assumptions, alternative
  explanations, predictions, and falsifiers; and
- `scientific_hypothesis`: a topic-bound, mechanism-derived prediction with a minimal
  intervention, diagnostics, and explicit falsifiers.

Records are written once. Reusing an ID with different content is a conflict. Code and data
blobs are addressed by SHA-256. `validate` checks schemas, digests, references, sealed-field
equality, blobs, claims, and v2 protocol invariants.

A protocol-v2 ExperimentSpec may exist unsealed, but once sealed it may bind at most one
ExecutionManifest. Sealing holds a per-spec lock and rejects a different second manifest;
validation also rejects an imported/adversarial registry containing multiples. Promotion
requires the one sole manifest. This is stricter than ordinary per-record-ID immutability.

## Scientific-library state

Retrieval never creates a belief. OpenAlex discovery writes a `literature_search` and
immutable metadata/abstract source snapshots. The agent separately reads relevant full
texts and writes atomic claims with exact locations. Mechanism edges may cite only
supporting claims; opposition remains visible in the belief ledger and alternative models.

`ScientificLibrary.synthesize()` is a cheap, deterministic projection. It performs no web,
model, embedding, or GPU call. For every independent work it computes visible component
scores for venue tier, peer review, study design, content depth, artifacts, reproduction,
directness, scope match, and risk of bias. Multiple snapshots/extractions from the same
work cannot multiply confidence. Independent supporting and opposing units contribute in
opposite directions; substantial mass on both sides produces `contested`, not a forced
binary answer. The published aggregate is explicitly a heuristic evidence confidence, not
a calibrated Bayesian posterior.

The exact component mappings, weights, aggregation formula, experiment contribution,
thresholds, and known limitations are versioned in `docs/SCIENTIFIC_METHOD.md`.

Mechanism confidence is limited by its weakest claim-backed edge. Hypothesis foundation
confidence combines its cited belief and mechanism support. Normal production model search
requires agenda coverage complete for every hypothesis topic, at least two independent
evidence units, and foundation confidence of at least 0.60. An explicit
`--allow-weak-science` override supports high-risk exploration without weakening or hiding
the stored provenance.

The derived views are:

```text
SCIENCE.json
SCIENTIFIC_BELIEFS.json
MECHANISMS.json
HYPOTHESES.json
RESEARCH_GAPS.json
IDEA_QUEUE.json
RESEARCH_TASKS.json
SCIENCE.md
```

`RESEARCH_GAPS` separates retrieval due from analysis due. Sparse, stale, or sufficiently
old contested topics trigger search; missing full text or extracted claims triggers agent
analysis rather than repeated queries. `conclude` accepts only a decisive completed
confirmation and creates an experiment-origin claim tied to exact EvidenceDecision IDs.
Subsequent synthesis combines it with the literature ledger instead of replacing history.

Scientific work and GPU work are deliberately concurrent. `run --science-agenda` launches
due network retrieval in one CPU thread while the resident worker pool drains manifests.
An external research agent may register immutable sources/claims/mechanisms/hypotheses and
enqueue candidates from another process while `run --follow` waits for work. No literature
or language-model operation is allowed between a completed GPU job and the next launch.

## Scientific state machine

V2 has three explicit search lanes.

### Bank calibration

`calibrate` creates a pilot control at search seed 42 for each resource/GPU slot declared in
the execution file. Each spec and manifest is pinned to one slot and enqueued.

A bank control enters the derived index only after a valid EvidenceDecision verifies the
result, including its emitted seed and GPU identity. A GPU slot key is the resource ID plus
the physical UUID observed at launch; the CPU demo uses an explicit `resource:cpu` fallback.

An eligible control must match:

- bank revision;
- baseline fingerprint;
- stable context fingerprint;
- structured scope and comparison group;
- executable seed; and
- declared resource/GPU availability.

Bank-view eligibility requires it to be no more than one hour old and have fewer than eight
landed uses. Atomic candidate staging additionally counts active pending/running/waiting
reservations; those reservations are intentionally not subtracted from `BANK.json`.

### Banked candidate

`search` builds the candidate context before writing the spec, selects one fresh exact
control, pins execution to that slot, and freezes the reconstructed control identity into
`search.reference_controls`. Dynamic fallback to another baseline or GPU is forbidden.

The context fingerprint includes comparison group, scope, metric, requirements, all data,
and all stable code. Only execution paths declared in `mutable_code_paths` may differ.
Every v2 arm must use the direct sealed-entry grammar. A launcher is accepted only when its
exact path is trusted by every possible execution resource: the CLI's `@python` expands to
the current interpreter, resources can declare an explicit runtime path, and fixed system
shell paths can enter a sealed launcher script. Bare PATH-resolved names are not identities.
`-c`/`-m` indirection, unsafe option prefixes, unsealed positional programs, absolute
payload arguments, and arm-level PATH/module-loader injection are rejected. Candidate argv
also rejects repeated long option names. The same loader variables are removed from the
inherited v2 child environment locally and with `env -u` in the remote payload. Identical
argv is valid when mutable bytes differ; identical argv and unchanged mutable bytes is a
rejected no-op.

One store lock covers control selection, immutable spec creation, sealing, and queue
admission. Pending reservations are derived only from active pending/running/waiting jobs,
so concurrent search processes cannot oversubscribe the last control use and failed staging
does not permanently consume it.

After execution, `BankIndex` scores the valid candidate against only its frozen control. For
the required minimize `val_bpb` metric, a delta below `-0.000426` produces
`promotion_due: true`. Candidate specs remain `sota_eligible: false`.

### Promotion confirmation

`promote` accepts only one landed, valid, scored candidate that cleared the bank gate. It
reconstructs control and candidate bindings from their immutable manifests, creates a
paired confirmation plan, seals it, and enqueues it.

Defaults are seeds 43–47 with four required valid replicates. Planned seeds must be distinct
non-negative integers held out from the bank/search pilot manifests. Every arm receives
matching `AUTORESEARCH_SEED`, and evidence checks the emitted `seed`. The hard minimum is
three valid preregistered replicates with distinct verified seeds. Every planned replicate
must first have a terminal ResultBundle and EvidenceDecision. Arm order may be counterbalanced
but the set of arms is fixed.

Control and candidate code snapshots are materialized under canonical sibling roots in a
namespace derived from the two immutable source manifest IDs, and their argv entry tokens
are rewritten to those roots. Knowledge recomputes the namespace and rejects aliases or
collisions. This preserves different bytes at the same source path inside one paired
manifest. Data bindings must have identical paths and digests across both source manifests.

Knowledge aggregates confirmation replicates and applies the preregistered paired success
rule. It never promotes a pilot value, a bank delta, or the best individual confirmation
seed. Before SOTA selection it independently reconstructs promotion provenance from
immutable records: unique candidate result and sole source/promotion manifests, frozen
control and recomputed bank score, fingerprints and source delta, scope/metric/requirements,
isolated code and identical data bindings, rewritten argv/env, held-out seeds, and AB/BA
order. A mismatch becomes a `sota_blockers` reason rather than a claim.

## Structured scope and SOTA partitioning

Protocol v2 requires a scope with:

```text
id, hardware_class, dataset_split, tokenizer, evaluator, precision,
metric={name: val_bpb, direction: minimize},
budget={kind: wall_seconds | training_seconds, value: positive number}
```

The workflow sets `comparison_group` to the scope ID. Any change that affects comparability
must therefore receive a new scope ID and a new SOTA group. This prevents a changed shard
range, tokenizer, evaluator, hardware class, precision, or budget from silently replacing
an old benchmark record. Calibration and search require every execution resource's declared
`hardware_class` to equal the scope. If an arm argv declares `--time-budget`, sealing
requires it to equal `scope.budget.value`. Evidence then applies two independent frame
checks. For `wall_seconds`, emitted `total_seconds` must be within
`max(0.05 seconds, 5%)`, while the harness-measured runner wall clock must be within
`max(0.05 * frame, min(0.25 seconds, 2 * frame))`. For `training_seconds`, emitted
`training_seconds` must be within `max(0.05 seconds, 5%)`, runner wall time must be at least
the sealed training frame, and emitted total time must agree with runner time within a
bounded allowance of `max(0.25 seconds, min(15 seconds, 5% of frame))`.
Missing/non-finite runner time or inconsistent `training_seconds`/`total_seconds` makes the
measurement invalid.

V1 records can remain readable in the same registry, but a v1 leaderboard/SOTA is not
automatically a v2 baseline. Migration means creating a structured v2 scope and new
comparison group, recalibrating, and reporting legacy results separately unless every
comparison-defining artifact and policy is demonstrably identical.

Evidence policy is versioned independently of the record schema. Protocol-v2 bank, belief,
and SOTA authority currently requires `evidence-v3`. Older `evidence-v2` decisions remain
immutable history but are excluded until `judge --all` derives v3 decisions from the same
ResultBundles.

## Profile-first gate

`doctor` is a derived diagnostic over current controls, ResultBundles, bank eligibility,
and queue health. It does not mutate or bless evidence.

`profile_health` compares `training_seconds` with `total_seconds`:

- `healthy`: training owns at least 75% of elapsed time;
- `overhead_dominated`: fixed overhead exceeds 25%;
- `unprofiled`: required structured timings are absent or non-numeric; and
- `invalid`: time values are negative, zero where forbidden, or internally inconsistent.

`doctor` reports model search ready only when every eligible control is healthy. `search`
rejects every control with missing or inconsistent timing. An overhead-dominated control is
allowed without override only for subsystem names beginning with calibration, compile,
data, evaluation/evaluator, input, instrumentation, or tokenizer. Other subsystems require
explicit `--allow-overhead-dominated`, so bottleneck work can proceed while the ordinary
search gate is red.

## Durable resident queue

`CampaignQueue` maps each existing immutable manifest ID to one deterministic job file:

```text
<root>/operational/queue/job_<manifest-id-hash>.json
```

All updates are atomic replacements under the campaign queue lock. The normal worker path
is:

```text
pending -> running -> complete
                   -> waiting   (no resource; bounded exponential backoff)
                   -> blocked   (uncertain execution or evidence failure)
waiting -> running
blocked -> complete             (only when immutable records prove completion)
```

Reconciliation may also move pending/waiting work directly to complete or blocked, recover
an unstarted running job to pending, or move a blocked job to pending when the absence of a
claim and artifacts proves that no execution crossed the durable boundary.

`run --workers N` owns a thread pool. `--follow` keeps it resident for newly staged work;
`--poll-seconds` controls polling and `--idle-timeout-seconds` bounds idle residence.
Workers judge every ResultBundle immediately. A job becomes complete only when every
planned replicate has both a ResultBundle and EvidenceDecision.

Queue idempotence means repeated enqueue calls return the same job. It does not authorize a
guess about an interrupted scientific launch. Failure handling is deliberately asymmetric:

- `NoResourceAvailable` proves no arm launched, so the job may wait and allocate later.
- An executed invalid/unknown result completes that replicate and is never automatically
  retried.
- Unexpected execution/evidence errors block the job and propagate to the caller.
- Fully landed results reconcile to complete and are judged if necessary.
- A live local queue owner or runner is left alone.
- An inflight claim or artifacts without a ResultBundle make execution uncertain and block
  the job.
- Only a dead owner with no claim and no artifacts safely returns to pending.

`queue --reconcile` exposes this recovery logic directly. Operational queue state never
deletes, replaces, or downgrades immutable scientific evidence.

## Rolling health circuit

Health is derived from the newest decision for each of the last 12 pilot ResultBundles.
Unknown measurements count with invalid measurements because neither is usable evidence.
The queue pauses new claims when:

- the last three or more pilot measurements are consecutively invalid/unknown; or
- invalid/unknown exceeds 25% once at least eight pilot results are in the window.

Running work is not killed. `--ignore-health` bypasses only queue claiming; it does not
alter EvidenceDecision verdicts or claim eligibility.

## Resource coordination boundary

Execution coordinates cooperating workers with two local mechanisms:

- host-slot locks grouped by `host_id` and limited by `max_concurrent_jobs`;
- per-resource/GPU locks with inspectable lease JSON breadcrumbs.

The `flock` is authoritative only for workers that share the same controller state root.
Lease JSON is operational visibility, including crash breadcrumbs. Neither prevents an
unrelated process, another root, another user, or a cluster scheduler from using the same
physical device.

`reservation.mode: externally_reserved` is provenance, not an allocation API. It requires
an ID but does not obtain exclusivity. The operator must acquire a real reservation first.

Execution holds a workdir lock across staging and the full replicate, so resources sharing
one directory serialize instead of overwriting one another. Separate per-slot workdirs are
therefore required for actual concurrency, but do not isolate shared CPU, filesystem,
network, compiler caches, or evaluation load. The shared-host template caps execution at
one; the eight-resource fleet template is appropriate only with real external reservation
and measured host capacity.

## Reviews and papers

Legacy v1 confirmations require the complete independent review declaration and obey the
legacy paper gate. Protocol-v2 confirmations are reviewless by default; `promote --reviews`
can still attach an optional digest-bound declaration. The sealing paper gate returns early
for protocol v2.

Paper registration and paper-status views remain available, but v2 calibration, search,
promotion, queueing, judging, and SOTA synthesis do not wait for a review council or paper
cadence. This preserves audit features without putting them between successive GPU jobs.

## On-disk layout

```text
<root>/
  records/
    experiment_spec/*.json
    execution_manifest/*.json
    result_bundle/*.json
    evidence_decision/*.json
    paper/*.json
    research_agenda/*.json
    literature_search/*.json
    literature_source/*.json
    scientific_claim/*.json
    scientific_mechanism/*.json
    scientific_hypothesis/*.json
  blobs/sha256/*
  artifacts/<result-id>/*
  operational/
    queue/*.json
    inflight/*.json
    locks/*.lock
    leases/*.json
  views/
    BANK.json
    PROMOTION_QUEUE.json
    KNOWLEDGE.json
    SOTA.json
    SCIENCE.json
    SCIENTIFIC_BELIEFS.json
    MECHANISMS.json
    HYPOTHESES.json
    RESEARCH_GAPS.json
    IDEA_QUEUE.json
    RESEARCH_TASKS.json
    ...
```

Scientific records and blobs are authoritative. Artifacts support result audit. Queue,
claims, locks, and leases are operational. Views are replaceable projections.

## Module map

```text
records.py    record schemas and protocol invariants
store.py      immutable persistence, blobs, atomic operational writes, locks, views
research.py   ExperimentSpec and optional Paper creation
protocol.py   v2 constants, scope normalization, seeds, profile classification
workflow.py   high-level calibrate/search/promote staging
sealing.py    digest-bound ExecutionManifest creation
campaign.py   resident durable queue, reconciliation, health circuit
execution.py  claims, resource coordination, local/SSH runners, telemetry
evidence.py   provenance and measurement validity
bank.py       exact controls, usage/TTL, candidate scoring, promotion projection
doctor.py     calibration and profile diagnostics
knowledge.py  evidence aggregation, beliefs, scope-bounded SOTA
science.py    retrieval, confidence ledger, mechanisms, hypotheses, research gaps
loop.py       legacy synchronous screen path
chart.py      legacy leaderboard projection
cli.py        command parsing and engine composition
```

## Core invariants

1. Literature retrieval, interpretation, and belief projection are separate stages.
2. Every model experiment names an immutable hypothesis and claim-backed mechanism.
3. Scientific intent is immutable before execution.
4. Execution uses only a sealed manifest and snapshotted bindings.
5. Planned and emitted seeds must agree.
6. Physical GPU identity is observed, validated, and used for bank matching.
7. A candidate reference is frozen before launch and never dynamically substituted.
8. Queue recovery never retries work whose execution is uncertain or known to have occurred.
9. Invalid evidence remains immutable and visible.
10. Bank and queue decisions cannot create SOTA.
11. SOTA is an aggregate promotion claim inside one structured scope.
12. OPHIS coordination is not proof of external resource exclusivity.
