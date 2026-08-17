# Simplify Autoresearch v2 design

## Purpose

V2 separates fast candidate selection from scientific promotion while keeping both paths
inside the same provenance and execution boundaries. Its target is useful learning per unit
of wall time, not the largest number of records and not the lowest unconfirmed seed.

Before candidate selection, the scientific-library layer continuously maintains a scoped
research agenda, immutable literature searches and source snapshots, atomic supporting and
opposing claims, claim-backed causal mechanisms, and falsifiable hypotheses. Production
model search is hypothesis-bound; a free-form summary is not the scientific authority.

Confidence synthesis is cheap deterministic local arithmetic over venue and peer review,
study design, content depth, artifacts, independent reproduction, directness, scope match,
bias, contradiction, and experiment evidence. It does not call an LLM or the network.
OpenAlex retrieval and deep agent reading run outside the GPU launch path and may proceed
concurrently with resident workers. See `docs/ARCHITECTURE.md` for the record contracts.

The design has three stages:

1. calibrate the benchmark and create fresh controls on each physical GPU;
2. screen a candidate against a control frozen before launch on that same GPU;
3. promote a promising candidate only through a new, distinct-seed confirmation.

The search stages are deliberately incapable of changing SOTA. They nominate work for the
promotion stage.

## Reference baseline

The default model-search lineage begins at
`karpathy/autoresearch@228791fb499afffb54b46200aca536f79142f117`. V2 retains a pristine
snapshot under `runs/code/upstream/` and uses `runs/code/train.py` as the editable baseline.
The adapter changes only executable seed/endpoint instrumentation and total-process timing;
model and optimizer research begins from the upstream bytes. `runs/code/prepare.py` is
stable, sealed code and corrects byte-fallback lengths with a versioned raw-byte table. That
correction deliberately defines a new scope rather than borrowing upstream historical
scores. The upstream dependency lock is retained verbatim.

## Stage 0: structured scope and calibration

A protocol-v2 experiment carries a structured `scope`. The validated core fields are:

- `id`;
- `hardware_class`;
- `dataset_split`;
- `tokenizer`;
- `evaluator`;
- `precision`;
- `metric`, currently `val_bpb` with direction `minimize`;
- `budget`, a positive `wall_seconds` or `training_seconds` value.

Additional scope fields are retained in canonical JSON and therefore participate in the
scope digest. They are useful for binding the data manifest, selected shards, sampler,
attention backend, dependency versions, or other benchmark facts not represented by the
validated core. The scope template intentionally includes such placeholders. Their meaning
is operator-defined; the core validator does not independently interpret them.

The scope is part of the immutable ExperimentSpec and is copied into the sealed manifest.
The Evidence Engine rejects a manifest that changes it, and bank/context fingerprints bind
it. A different training split, tokenizer artifact, evaluator implementation,
byte-accounting rule, precision, hardware class, or timing budget is a different
scope—not another point on the same leaderboard. Knowledge keys views by comparison group
but refuses to choose SOTA when one group contains different canonical scopes or stable
context fingerprints. Assigning a new scope ID remains the clear operational convention.

### Profile first

Calibration should establish the cost structure before architecture search begins. The v2
fast-lane contract requires the primary metric, optimizer-step count, executed seed,
training seconds, and total seconds. A `wall_seconds` scope preserves the strict end-to-end
contract: both emitted total time and the independent runner clock must match the sealed
frame. A `training_seconds` scope supports trainers whose fixed compute clock deliberately
excludes startup, compilation, and evaluation. In that mode, emitted training time must
match the sealed frame, the independent runner must last at least that long, and emitted
total time must reconcile with the runner within a small bounded allowance; total time is
not required to equal the inner training frame. In both modes the timing relationship must
be consistent, and telemetry sampling cadence cannot pad the runner clock.
`profile_health()` classifies a measurement as overhead-dominated when
non-training time exceeds 25% of total elapsed time. This is a
diagnostic policy gate, not a causal claim: it says to inspect data loading, compilation, or
evaluation before spending a large search budget; it does not say which component is
responsible.

A healthy calibration should also establish, per physical GPU:

- repeatability of the unchanged control;
- normal optimizer-step throughput for the exact argv;
- the physical GPU UUID reported by launch telemetry;
- whether the requested host concurrency preserves throughput.

The throughput baseline used by evidence is keyed by exact argv and physical placement. It
is not pooled across GPUs.

## Stage 1: per-GPU control bank

A bank entry is a valid pilot in search lane `bank`. The bank itself is not a new evidence
record: it is a rebuildable projection of immutable specs, manifests, results, and their
latest EvidenceDecisions.

An eligible control binds:

- the bank revision;
- the exact baseline fingerprint;
- the surrounding context fingerprint, including stable code and all data bindings;
- the verified search seed;
- the recorded resource and physical GPU UUID;
- its source result and evidence records.

Current bank policy keeps a control eligible for one hour and no more than eight candidate
uses. These are search-efficiency limits, not statistical guarantees. A changed champion or
changed stable context needs a new bank revision.

## Stage 2: banked candidate

A search-lane `candidate` is a one-arm, one-seed pilot. Before launch, it freezes the exact
eligible bank rows it may use. Dynamic fallback is forbidden: after seeing a result, the
system cannot substitute a newer, lower, or cross-GPU control.

The candidate is scored only when its frozen reference agrees on bank revision, baseline
fingerprint, context fingerprint, seed, and physical GPU. A control that was future-dated,
expired, or already over its use limit when the candidate started is not usable.

For the minimizing objective, a candidate delta below `-0.000426` enters the promotion
queue. That threshold is a search gate, not proof. The scored row explicitly remains
`sota_eligible: false`.

Mutable code must be declared as normalized relative execution paths. Every other code
binding and every data binding remains part of the stable context fingerprint. This permits
the intended candidate file to change without silently allowing the evaluator, tokenizer,
or data context to change with it. Candidate argv must enter a relative sealed code path
through a launcher whose exact token is trusted by every selectable resource. The CLI's
`@python` is local-only expansion to its current interpreter; remote resources declare an
explicit path such as `.venv/bin/python`. Bare PATH lookup, option-shaped fake entries,
inline/module execution, absolute payloads, and arm loader-environment overrides are
rejected. The inherited v2 child environment is scrubbed of the same loader variables.
Keeping argv unchanged while editing declared mutable bytes is the normal supported loop.

## Stage 3: distinct-seed promotion

Promotion is a new confirmation ExperimentSpec in search lane `promotion`; a candidate is
never relabeled after its outcome. Protocol v2 enforces all of the following for a
SOTA-eligible confirmation:

- at least three valid preregistered replicates;
- distinct planned seeds;
- `AUTORESEARCH_SEED` in every arm environment matching the replicate's planned seed;
- `seed` among the required metrics, verified from the emitted result;
- the same set of arms in every replicate.

Replicate arm order may be counterbalanced, for example control/candidate then
candidate/control. The executed order remains sealed and is checked exactly. The default
protocol constants reserve seed 42 for search and seeds 43–47 for promotion. Custom seeds
must also be disjoint from the bank and candidate pilot seeds.

Control and candidate code trees are materialized under canonical sibling roots in a
namespace derived from their immutable source manifest IDs, so both arms may execute
different bytes originally bound to `train.py`. Knowledge recomputes the roots and rejects
aliases or namespace collisions. It waits for every preregistered replicate to land and be
judged; the minimum-valid count tolerates terminal invalid measurements, not unfinished
seeds.

Only valid confirmation evidence can be claim-eligible. The Knowledge Engine aggregates
the preregistered replicates and uses their mean for SOTA; it does not promote the best
candidate seed or the banked search delta.

## Durable resident queue

`CampaignQueue` is the operational layer that keeps the agent out of the launch critical
path. An existing immutable manifest maps to one deterministic job file under
`operational/queue/`. The file moves atomically, under a queue lock, through:

```text
pending -> running -> complete
                   -> waiting   (no resource; bounded exponential backoff)
                   -> blocked   (uncertain execution or evidence failure)
waiting -> running
blocked -> pending              (after claim release and proof nothing launched)
blocked -> complete             (if immutable results later prove completion)
```

Resident workers can drain several manifests concurrently, follow for newly enqueued work,
poll at a selected interval, and stop after an idle timeout. A ResultBundle is judged
immediately. A manifest is complete only when every planned scientific replicate has a
ResultBundle and an EvidenceDecision.

Queue state is operational, not evidence. A failed or invalid ResultBundle still means the
scientific replicate executed and must not be retried. `NoResourceAvailable` is different:
the execution service proves that no arm launched, so the queue may wait and try allocation
again. An unexpected runner failure is blocked and re-raised.

Crash reconciliation is conservative:

- complete immutable results finish and judge the job;
- a live local owner/runner PID plus an inflight claim is never stolen;
- an uncertain or orphaned inflight claim blocks the job;
- artifacts without a ResultBundle block the job;
- only a dead queue owner with no claim and no artifacts is safely returned to `pending`.

There is intentionally no shell `while` loop, process-name polling convention, or mutable
queue text file in this path.

## Health circuit

The queue derives health from the latest decisions for the last 12 pilot results. It pauses
new claims when either condition holds:

- at least three most-recent pilots are consecutively invalid or unknown; or
- more than 25% are invalid or unknown once at least eight pilots are present.

Already-running work is not killed. The pause is a request to diagnose resource health,
bindings, instrumentation, or configuration—not permission to weaken evidence gates. An
explicit programmatic override exists for recovery work, but ordinary campaign operation
should resolve the cause before bypassing the circuit.

## Resource coordination is not a reservation

OPHIS uses `flock`-protected host slots and per-resource leases to coordinate workers using
the same state root. Lease JSON makes ownership inspectable, but the flock is the local
authority and disappears when its owner process dies.

This coordination cannot stop an unrelated process, another state root, another user, or an
external scheduler from using the same GPU or host. `reservation.mode` is validated and
recorded provenance. Setting it to `externally_reserved` does not acquire a reservation.
The operator must first obtain real exclusivity from the cluster, cloud, or job scheduler
and then replace the template's reservation ID with that real allocation identifier.

`host_id` groups resources for the cooperative host concurrency cap.
`max_concurrent_jobs` is therefore an OPHIS cap, not proof the host can sustain that width.
Use calibration to justify the chosen value. A shared host should begin with the single-job
template unless profiling demonstrates that wider execution remains valid.

## Workdir and host-concurrency hazards

Execution bindings are staged at relative paths beneath each resource's `workdir`. V2 holds
a workdir lock for the complete replicate, so cooperating workers that share a directory
serialize instead of overwriting one another. The eight-GPU fleet template gives each
physical GPU a distinct workdir so those jobs can actually run concurrently.

Before sealing a fleet configuration:

- create every remote workdir;
- declare the same explicit trusted launcher token on every resource that may run a given
  manifest, make it available from each workdir, and keep the sealed code entry relative
  (for example, `.venv/bin/python train.py`);
- make immutable data available consistently to every slot;
- keep resource IDs and host identity stable between bank and candidate work;
- ensure no concurrent job writes checkpoints, compiler caches, or temporary files to a
  shared path unless that sharing is explicitly safe;
- account for host-wide CPU, filesystem, network, and compilation contention. Eight idle
  GPUs do not imply the host can support eight healthy jobs.

Separate workdirs prevent OPHIS staging collisions; they do not provide process, storage,
or performance isolation.

## Claim and SOTA boundaries

The boundaries are intentionally asymmetric:

- queue state says what should run next;
- ResultBundle says what actually ran;
- EvidenceDecision says whether that measurement is valid and claim-eligible;
- the bank says whether a valid exploratory result deserves promotion work;
- a supported, eligible promotion confirmation may update SOTA.

Neither a queue completion, a raw low metric, a valid pilot, nor a `promotion_due` bank row
is a scientific claim. Conversely, deleting or retrying an invalid executed replicate does
not make it disappear; the immutable result remains part of the audit trail.

The current protocol-v2 authority is `evidence-v3`. Earlier `evidence-v2` decisions remain
immutable and inspectable, but they cannot populate the bank, beliefs, or SOTA. Run
`autoresearch --root STATE judge --all` to derive current decisions from existing immutable
ResultBundles before rebuilding views.

## Migration from v1

V2 can retain low-level compatibility with legacy records, but a v1 leaderboard or SOTA is
not automatically a v2 baseline. V1 experiments often encoded benchmark identity in an
informal comparison-group string or argv, and campaigns used different shard ranges,
tokenizer artifacts, evaluator byte accounting, precision, or execution frames.

Migration therefore means:

1. define and freeze a complete v2 scope;
2. bind the exact data, tokenizer, evaluator, and code artifacts;
3. establish fresh profile and per-GPU bank controls;
4. rerun any inherited candidate under that scope;
5. use distinct-seed v2 confirmation before promotion.

V1 records remain useful historical and diagnostic evidence. Their absolute scores must not
be placed on the v2 leaderboard unless the full scope is demonstrated identical—and a fix to
the evaluator, tokenizer, data split, or byte-fallback accounting makes it non-identical by
definition.

## Configuration templates

- `templates/scope_v2.json` lists the validated scope fields and optional digest-bound
  details that must be replaced for a real campaign.
- `templates/execution_remote_fleet.json` describes eight cooperative OPHIS slots on one
  externally reserved host, with one GPU and one workdir per resource.
- `templates/execution_remote_shared_single.json` is the conservative shared-host shape:
  at most one OPHIS job across the host, with no claim of external exclusivity.

These are configuration shapes, not executable examples. Placeholder paths, hostnames,
hashes, allocation IDs, and environment assumptions must be replaced and verified before
sealing.
