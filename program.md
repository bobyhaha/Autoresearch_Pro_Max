# V2 operating program

The v2 program is a scientific loop: continuously search and read relevant literature,
maintain an attributed claim and mechanism library, derive falsifiable hypotheses, and use
a durable resident queue to test them without putting reasoning in the GPU launch path.

This file is an operating protocol, not a campaign log. The immutable registry and derived
views are the record of what happened.

## Non-negotiable boundaries

1. A model idea begins with an immutable scientific hypothesis, not a free-form edit. Its
   agenda coverage, claims, mechanism edges, prediction, falsifiers, and provenance must be
   inspectable before ordinary GPU search.
2. Confidence is a rebuildable evidence score, never a mutable opinion. Venue prestige is
   one small component; methods, artifacts, independent reproduction, applicability,
   contradiction, and our own valid experiments remain separate and visible.
3. Scope is frozen before search. Hardware class, dataset split, tokenizer, evaluator,
   precision, metric, and budget define comparability.
4. Any change to those fields receives a new scope ID. V2 uses that ID as the comparison
   group, so it also creates a new SOTA group.
5. Only a sealed ExecutionManifest can launch work. Code and data bindings are immutable
   snapshots, not paths whose contents may drift after the run. A v2 spec may be unsealed,
   but once sealed it may have only one manifest. A different second seal is rejected
   atomically; validation also rejects externally constructed duplicates.
6. The seed is executable state: every arm receives the preregistered
   `AUTORESEARCH_SEED`, emits `seed`, and evidence verifies the two agree.
7. Every v2 arm emits structured timing. Sealing rejects a conflicting `--time-budget`.
   For a `wall_seconds` scope, evidence checks emitted total time and the independent runner
   clock against the frame. For Karpathy-compatible `training_seconds`, it checks the
   charged training clock against the frame and independently reconciles emitted total with
   complete runner wall time.
8. Queue status is not scientific evidence. ResultBundle says what ran; EvidenceDecision
   says whether it is usable.
9. A pilot is exploratory. Only a supported v2 promotion confirmation can update SOTA.
10. Never automatically retry an executed invalid, unknown, or uncertain scientific
   replicate.

## 0. Maintain the scientific library

Register an agenda before proposing model changes:

```bash
uv run autoresearch --root .autoresearch agenda templates/research_agenda.json
uv run autoresearch --root .autoresearch literature-refresh \
  agenda_karpathy_short_budget --limit 50
```

Retrieval and interpretation are different records. `literature_search` preserves the
provider query and returned identities. `literature_source` preserves bibliographic data,
an abstract, and—when available—a content-addressed full text. Search metadata is never
silently promoted into a claim.

The research agent reads methods, comparator, data, budget, seeds, uncertainty, ablations,
and limitations, then registers atomic supporting or opposing claims with exact locators:

```bash
uv run autoresearch --root .autoresearch literature-source source.json
uv run autoresearch --root .autoresearch scientific-claim claim.json
uv run autoresearch --root .autoresearch mechanism mechanism.json
uv run autoresearch --root .autoresearch hypothesis hypothesis.json
uv run autoresearch --root .autoresearch science
```

`confidence-v1` separately scores venue, peer review, study design, content depth,
artifacts, reproduction, directness, scope match, and bias. It deduplicates snapshots of one
work and preserves independent opposition as a contested belief. The aggregate is a
heuristic decision aid, not a calibrated posterior.

Use `RESEARCH_GAPS.json` to decide when to search again or perform full-text claim
extraction; `SCIENTIFIC_BELIEFS.json` to inspect support and opposition;
`MECHANISMS.json` to find weak causal edges; `HYPOTHESES.json` to update predictions after
results; `IDEA_QUEUE.json` to choose research-ready untested hypotheses; and
`RESEARCH_TASKS.json` as the agent's prioritized action queue.

A normal model hypothesis needs completed agenda coverage, at least two independent
evidence units, foundation confidence of at least 0.60, and a valid claim-backed mechanism.
`--allow-weak-science` is a deliberate exploration exception, never the default.

## 1. Calibrate the bank

The default baseline is `runs/code/train.py`, derived from
`karpathy/autoresearch@228791fb499afffb54b46200aca536f79142f117`. Do not seed a new
campaign from an OPHIS SOTA snapshot. Keep `prepare.py`, `pyproject.toml`, `uv.lock`, and
baseline provenance sealed and stable; only `train.py` is the normal mutable path. Run the
unedited baseline first and do not begin candidate edits until its calibration is valid.

Create the store and stage one seed-42 control for every resource/GPU slot in the execution
file:

```bash
uv run autoresearch --root .autoresearch init

uv run autoresearch --root .autoresearch calibrate baseline_revision calibration_label \
  --scope path/to/filled_karpathy_scope.json \
  --execution templates/execution_remote_h200.json \
  --mutable-code-path train.py \
  --argv .venv/bin/python train.py

uv run autoresearch --root .autoresearch run --workers 8
uv run autoresearch --root .autoresearch doctor --bank-id baseline_revision
```

Calibration creates one pinned manifest and queue job per declared slot. A valid GPU
control is keyed by resource ID plus its observed physical GPU UUID. It matches candidates
only when bank revision, baseline fingerprint, stable context, scope, and seed agree. Every
resource must declare a `hardware_class` exactly equal to the scope's value.

Do not begin ordinary search until `doctor` reports `model_search_ready: true`.
The command exits with status 4 when it is not ready. A healthy profile has
`training_seconds / total_seconds >= 0.75`. A control with missing or inconsistent timing
cannot be selected. An overhead-dominated control may be used for subsystem names beginning with
calibration, compile, data, evaluation/evaluator, input, instrumentation, or tokenizer;
other candidates require the explicit `--allow-overhead-dominated` override.

Controls expire one hour after completion and are capped at eight landed candidate uses or
pending reservations. Recalibrate when they are stale, exhausted, the physical resource
changes, or any exact-context field changes.

## 2. Stage banked candidates

Use `search`, not legacy `screen`, for new work:

```bash
uv run autoresearch --root .autoresearch search candidate_label \
  --bank-id baseline_revision \
  --summary "one precise change and expected mechanism" \
  --scope path/to/scope.json \
  --execution path/to/execution.json \
  --mutable-code-path train.py \
  --hypothesis-id hyp_exact_registered_hypothesis \
  --direction optimization \
  --subsystem optimizer \
  --argv .venv/bin/python train.py
```

Keep `--argv` last because it consumes the remaining tokens. `@python` expands to the exact
current interpreter for local work. Every selectable remote resource must explicitly trust
the same workdir-relative interpreter path (for example `.venv/bin/python`) and argv must
use that exact token. Complex launch logic belongs in a sealed script entered through an
explicitly trusted system shell such as `/bin/bash sealed_launcher.sh`. Bare PATH-resolved launchers,
`-c`/`-m`, unsafe option prefixes, unsealed positional programs, absolute payload arguments,
and arm overrides of `PATH` or module/loader injection variables are rejected; the v2 child
also receives those variables scrubbed from the worker environment. Do not repeat
long option flags in candidate argv. Declare intentionally changed bindings with repeated
`--mutable-code-path`; all other code and all data remain in the stable context fingerprint.
The normal edit loop may keep argv identical while mutable bytes change; unchanged argv and
bytes is a rejected no-op.

For production model subsystems, `--hypothesis-id` is required. The ExperimentSpec copies
the registered statement, prediction, falsifier, and claim-backed mechanism chain and
records the immutable hypothesis link. Promotion preserves that link, so confirmation
evidence can refine the same scientific object instead of becoming an isolated score.

At staging time, one lock protects control selection, immutable registration, sealing, and
queue admission. The workflow chooses a fresh exact control, pins the candidate to that
resource/GPU, and stores the complete reference in the immutable ExperimentSpec. Only active
pending/running/waiting candidate jobs reserve uses, so a failed staging attempt does not
consume the bank. If no matching control exists, if all matching controls are
reserved/exhausted, or if several baseline fingerprints make the bank ambiguous, staging
stops before execution.

Run queued work with the resident worker pool:

```bash
uv run autoresearch --root .autoresearch run \
  --workers 8 \
  --follow \
  --poll-seconds 1 \
  --idle-timeout-seconds 600
```

Use `--follow` when another process will continue staging work. Without it, workers drain
currently eligible jobs and exit.

## 3. Read the bank gate

Rebuild the projections after candidate work:

```bash
uv run autoresearch --root .autoresearch bank
uv run autoresearch --root .autoresearch status
```

`views/BANK.json` describes valid controls, TTL, and landed-use eligibility. It does not
subtract active queue reservations; atomic `search` staging adds those before selecting a
control, so a row marked `eligible_now` can still be temporarily unavailable.
`views/PROMOTION_QUEUE.json` scores valid candidates only against the exact same-slot
control frozen before launch. A minimize-metric candidate is promotion-due only when its
delta is strictly below `-0.000426`.

This is a work-allocation decision, not a claim. A valid candidate remains exploratory and
cannot update SOTA.

## 4. Promote on distinct seeds

Stage confirmation only for a candidate listed as promotion-due:

```bash
uv run autoresearch --root .autoresearch promote exp_candidate_label
uv run autoresearch --root .autoresearch run --workers 8
uv run autoresearch --root .autoresearch synthesize
uv run autoresearch --root .autoresearch validate
```

The default promotion preregisters seeds 43–47, requires four valid replicates, runs both
the exact control and candidate at every seed, and counterbalances arm order. Callers may
choose a different distinct non-negative seed list and minimum, but promotion seeds must be
held out from bank/search pilots and the protocol requires a minimum of three. SOTA
eligibility requires every planned replicate to land and be judged, at least three unique
verified seeds, the preregistered success rule, and the required valid replicate count.

Promotion bindings are reconstructed from the immutable bank and candidate manifests.
Control and candidate code trees are staged under separate immutable roots so the same
relative path may contain different bytes. `promote --execution` may override only
resources/runtime; it cannot replace code or data. Digest-bound reviews are optional for v2
promotion.

On `synthesize`, Knowledge recomputes promotion lineage instead of trusting its labels: the
unique source result and sole manifests, frozen bank score/control, fingerprints and delta,
scope/requirements, exact code/data snapshots, argv/env, held-out seeds, and AB/BA order
must all reconstruct. Inspect `sota_blockers` when valid promotion evidence does not publish.
The roots are canonical siblings beneath a namespace derived from the immutable control and
candidate manifest IDs; a low-level promotion cannot choose its own aliases.

After a decisive confirmation, materialize the agent's own experiment-origin claim and
rescore the library:

```bash
uv run autoresearch --root .autoresearch conclude \
  hyp_exact_registered_hypothesis exp_promote_candidate_label
uv run autoresearch --root .autoresearch science
```

`conclude` binds the claim to the exact confirmation EvidenceDecision IDs. A supported or
refuted result changes the confidence ledger; it never erases the literature claims that
led to the experiment.

Current v2 views require `evidence-v3`. Older `evidence-v2` records remain immutable history
but cannot seed the bank or a claim; run `judge --all` to rejudge their original ResultBundles
before `synthesize`.

## Queue operations and failure policy

The staging commands enqueue automatically. `enqueue MANIFEST_ID ...` exists for manifests
created with the low-level interface.

```bash
uv run autoresearch --root .autoresearch queue --jobs
uv run autoresearch --root .autoresearch queue --reconcile
uv run autoresearch --root .autoresearch claims
```

Each manifest has one deterministic atomic job file and transitions through `pending`,
`running`, `waiting`, `complete`, or `blocked`.

- `NoResourceAvailable` proves no arm launched. The job waits with bounded exponential
  backoff and may claim a resource later.
- A landed ResultBundle is judged immediately. Invalid or unknown evidence still completes
  that executed replicate; it is not retried.
- An unexpected execution or evidence exception blocks the job and is re-raised.
- Reconciliation completes fully landed work, leaves live local owners/runners alone,
  blocks orphaned claims or artifacts without a ResultBundle, and returns a dead unstarted
  owner to pending only when no claim or artifacts exist.
- Release an inflight claim only after independently proving its local and remote processes
  are dead; the CLI requires `claims --release TOKEN --confirm-dead`.

The health circuit derives from the latest EvidenceDecision for each of the last 12 pilot
results. It pauses new claims after three consecutive invalid/unknown pilots, or above a 25%
invalid/unknown rate once eight pilots exist. Inspect `doctor`, `status`, `queue --jobs`,
resource leases, bindings, and instrumentation before using `run --ignore-health`.

## Shared-host policy

OPHIS host slots and resource/GPU leases coordinate only workers using the same state root.
They are not an external reservation.

- Set a stable `host_id` for resources on the same physical host.
- Treat `max_concurrent_jobs` as an OPHIS cooperative cap, not a performance guarantee.
- Begin shared-host work with `templates/execution_remote_shared_single.json` and one job.
- Use distinct workdirs for concurrent slots. A workdir lock prevents cooperating OPHIS
  staging races by serializing a shared directory, which also removes its parallelism.
- Use the eight-slot `templates/execution_remote_fleet.json` only after profiling host CPU,
  storage, network, compilation, and evaluation at that width.
- `reservation.mode: externally_reserved` records a real scheduler/cloud allocation ID; it
  does not acquire exclusivity. Obtain that allocation first.

## Compatibility commands are outside the hot path

`screen` is the legacy synchronous one-seed, one-arm v1 loop and legacy leaderboard. It is
useful for existing scripts, not for v2 banked research. The low-level `design`,
`review-template`, `seal`, `execute`, `judge`, `cycle`, and `paper` commands also remain for
compatibility and manual workflows.

V2 calibrations, candidates, and promotions do not wait for a five-review council or paper
cadence. Optional reviews and Paper records remain auditable features, and `synthesize`
still reports their status, but neither is a v2 search or SOTA gate.

## Parallel scientific and GPU work

Keep GPU workers resident while a separate research process reads papers, extracts claims,
constructs mechanisms, edits `train.py`, and stages candidates:

```bash
uv run autoresearch --root .autoresearch run \
  --workers 8 --follow --idle-timeout-seconds 3600 \
  --science-agenda agenda_karpathy_short_budget --literature-limit 50
```

The built-in due-agenda retrieval runs in a CPU thread parallel to queue draining, and
confidence synthesis is deterministic local work. Deeper agent reasoning should run in a
second process; immutable registration and queue admission are safe while `run --follow`
launches work event-first. The GPU queue must never wait for a literature or model call.

## End-of-run checklist

- `queue` has no unexpected `running`, `waiting`, or `blocked` jobs.
- `doctor` shows fresh controls and healthy profiles for model work.
- `bank` has no unexplained unscored candidates.
- `validate` returns `valid: true` with no registry or inflight errors.
- Any reported best value names its scope ID, aggregate promotion spec, result IDs, and
  distinct verified seeds.
- Results from another scope or legacy comparison group are reported separately, never as
  a head-to-head improvement.
