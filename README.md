# Simplify Autoresearch v2

V2 keeps the v1 evidence core—immutable specifications, sealed execution, structured
results, explicit evidence decisions, and rebuildable knowledge—but removes the agent from
the launch critical path. A resident durable queue, a reusable control bank keyed by
physical GPU, executable seed verification, and a profile-first gate make the common search
loop faster without making exploratory results look like confirmed claims.

The recommended workflow starts with scientific grounding, not an unstructured edit:

```text
agenda -> literature search -> full-text sources -> atomic claims
                                                -> mechanism graph
                                                -> falsifiable hypothesis
                                                          |
                                                          v
freeze scope -> calibrate controls -> run -> doctor -> stage candidate
                                                          |
                                                          v
                                              run -> bank gate -> promotion
                                                          |
                                                          v
                                        experiment-backed claim + belief update
```

## The scientific library is the front of the engine

Model ideas are no longer expected to begin as a one-line summary. A production `search`
must name an immutable scientific hypothesis with `--hypothesis-id`. That hypothesis must
name its literature topics, atomic supporting claims, a causal mechanism graph, prediction,
falsifiers, intervention, diagnostics, and competing explanations. CPU smoke tests and
explicit data/compile/evaluation/instrumentation diagnostics remain available without this
model-idea gate.

The authoritative scientific records are:

- `research_agenda`: questions, repeatable queries, refresh cadence, and minimum source and
  claim coverage;
- `literature_search`: provider, exact query, time, filters, and returned source snapshots;
- `literature_source`: stable work identity, bibliographic metadata, abstract, and optional
  content-addressed full-text snapshot;
- `scientific_claim`: one attributed, scoped claim with a supporting or opposing stance,
  exact locator, reported metrics, and an explicit evidence assessment;
- `scientific_mechanism`: a node/edge causal account whose every edge cites claims, plus
  assumptions, alternatives, predictions, and falsifiers; and
- `scientific_hypothesis`: a mechanism-derived, benchmark-scoped prediction tied to one
  minimal intervention and diagnostic measurements.

Start a scientific campaign with the provided templates:

```bash
uv run autoresearch --root .autoresearch init
uv run autoresearch --root .autoresearch agenda templates/research_agenda.json

# OpenAlex retrieval stores the exact query and immutable metadata/abstract snapshots.
uv run autoresearch --root .autoresearch literature-refresh \
  agenda_karpathy_short_budget --limit 50

# The agent then reads relevant full texts, registers their snapshots, and extracts
# atomic claims. Copy and fill these declarations; placeholders are intentionally invalid.
uv run autoresearch --root .autoresearch literature-source path/to/source.json
uv run autoresearch --root .autoresearch scientific-claim path/to/claim.json
uv run autoresearch --root .autoresearch mechanism path/to/mechanism.json
uv run autoresearch --root .autoresearch hypothesis path/to/hypothesis.json
uv run autoresearch --root .autoresearch science
```

OpenAlex retrieval is discovery, not reading. An abstract-only search result may inform
triage, but `RESEARCH_GAPS.json` continues to mark full-text analysis due. A claim becomes
part of a mechanism only through an explicit cited extraction. The agent must preserve a
full-text snapshot where available, inspect methods/results/ablations rather than trusting
the abstract, and record the location supporting its interpretation.

Confidence scoring is intentionally cheap: `science` is deterministic local Python over
immutable records. It makes no network, model, embedding, or GPU call. The score separately
publishes venue tier, peer-review status, study design, content depth, artifact status,
reproduction status, directness, scope match, and risk of bias. Repeated snapshots of the
same work are deduplicated. Independent opposing evidence creates a `contested` belief.
Venue prestige contributes only 6% of the quality score and can never establish a claim by
itself. The resulting number is a transparent heuristic evidence confidence, not a
statistically calibrated posterior probability.

By default, a model hypothesis cannot reach GPU search until:

- its agenda topics have met their configured search/source/claim coverage and include a
  full-text snapshot;
- its foundation confidence is at least `0.60`;
- it has at least two independent evidence units; and
- its mechanism and cited claims validate.

`--allow-weak-science` is an explicit exploration override. The immutable spec records the
hypothesis link, so an exploratory exception cannot masquerade as an unqualified mechanism.

After a decisive confirmation, turn the result into an experiment-origin claim:

```bash
uv run autoresearch --root .autoresearch conclude \
  hyp_replace_with_testable_hypothesis exp_promote_replace
uv run autoresearch --root .autoresearch science
```

The conclusion command accepts only a completed decisive confirmation and binds the new
claim to the exact EvidenceDecision IDs. Subsequent confidence synthesis combines that
experiment with literature support and opposition instead of overwriting history.

## Keep GPUs busy while research continues

Scientific reasoning is deliberately outside the GPU launch critical path. Run the GPU
queue in follow mode while a second agent/process reads papers, registers claims and
mechanisms, edits `train.py`, and stages new hypothesis-bound candidates:

```bash
uv run autoresearch --root .autoresearch run \
  --workers 8 --follow --idle-timeout-seconds 3600 \
  --science-agenda agenda_karpathy_short_budget \
  --literature-limit 50
```

`--science-agenda` performs due literature retrieval in a separate CPU thread while the
resident workers drain GPU work. Claim confidence is rescored after the queue drain. More
importantly, immutable scientific registration and candidate staging are safe from another
process while `run --follow` is active: new jobs are noticed without putting the research
agent between GPU launches. `IDEA_QUEUE.json` ranks untested hypotheses by foundation
readiness; `RESEARCH_GAPS.json` separates web search due from full-text/claim analysis due;
and `RESEARCH_TASKS.json` gives the agent a prioritized queue of search, reading, conflict
resolution, mechanism discrimination, experiment, refinement, and conclusion work.

## Default research baseline

Real campaigns now start from Andrej Karpathy's official
[`autoresearch/train.py`](https://github.com/karpathy/autoresearch/blob/master/train.py) at
commit `228791fb499afffb54b46200aca536f79142f117`, not from a later OPHIS champion or the
synthetic emitter. The active snapshot is `runs/code/train.py`; the byte-identical upstream
source is retained at `runs/code/upstream/train.py` with hashes and attribution in
`runs/code/provenance.json` and `runs/code/UPSTREAM.md`.

Only the protocol boundary differs from upstream: the active trainer executes and emits
`AUTORESEARCH_SEED`, measures total process time from before runtime imports, and appends
one structured evidence endpoint. The model, optimizer, hyperparameters, data loader, and
300-second charged-training clock start from Karpathy's file. The sealed `prepare.py` fixes
byte-fallback accounting with
`decode_single_token_bytes()` and versions the derived table, so this is a new
corrected-BPB scope rather than a number directly comparable with historical upstream
scores.

Bootstrap the baseline environment and data on the GPU host before opening a campaign:

```bash
cd runs/code
uv sync --frozen
uv run prepare.py
cd ../..

# Run this on the machine whose absolute cache path the workers will use.
# NOTE: the active prepare.py pins CACHE_DIR to ~/.cache/autoresearch_v2, so the
# manifest MUST be built from that root. Building it from the shared
# ~/.cache/autoresearch describes a different corpus than the one runs consume --
# on 2026-08-17 that shared cache held 41 shards and a byte table written by an
# unrelated campaign.
python3 tools/build_karpathy_manifest.py \
  --cache-dir "$HOME/.cache/autoresearch_v2" \
  --output runs/data_manifest.json
```

For SSH execution, run the same environment/data preparation in every declared workdir,
copy the generated manifest back to `runs/data_manifest.json`, then fill every placeholder
in `templates/execution_remote_h200.json` and
`templates/scope_karpathy_autoresearch.json`. In particular, record an immutable runtime
image and pinned Flash Attention kernel; upstream's dynamic kernel lookup is not itself a
reproducible runtime binding.

```bash
cp templates/scope_karpathy_autoresearch.json runs/scope.json
# Edit runs/scope.json and templates/execution_remote_h200.json now; placeholders are not
# valid campaign provenance.
```

The first scientific run is the unedited baseline calibration:

```bash
uv run autoresearch --root .autoresearch init

uv run autoresearch --root .autoresearch calibrate karpathy_228791f baseline \
  --scope runs/scope.json \
  --execution templates/execution_remote_h200.json \
  --mutable-code-path train.py \
  --argv .venv/bin/python train.py

uv run autoresearch --root .autoresearch run --workers 4
uv run autoresearch --root .autoresearch doctor --bank-id karpathy_228791f
```

Do not edit `train.py` until the unmodified control is valid and `doctor` reports
model-search ready. Thereafter, edit only `runs/code/train.py`, stage candidates with the
same execution path, name the research-ready hypothesis with `--hypothesis-id`, and let
immutable sealing preserve each revision.

## Harness smoke test

This deterministic CPU demo exercises the complete v2 path. It is a smoke test, not a
training benchmark or a result comparable with one.

```bash
uv sync --group dev

uv run autoresearch --root .demo init

uv run autoresearch --root .demo calibrate demo_champion initial \
  --scope examples/demo_scope_v2.json \
  --execution templates/execution_local.json \
  --mutable-code-path examples/metric_emitter.py \
  --argv @python examples/metric_emitter.py --value 1.0 --steps 1000

uv run autoresearch --root .demo run
uv run autoresearch --root .demo doctor --bank-id demo_champion

uv run autoresearch --root .demo search better \
  --bank-id demo_champion \
  --summary "lower deterministic demo metric" \
  --scope examples/demo_scope_v2.json \
  --execution templates/execution_local.json \
  --mutable-code-path examples/metric_emitter.py \
  --subsystem optimizer \
  --argv @python examples/metric_emitter.py --value 0.9 --steps 1000

uv run autoresearch --root .demo run
uv run autoresearch --root .demo bank
uv run autoresearch --root .demo promote exp_better
uv run autoresearch --root .demo run
uv run autoresearch --root .demo synthesize
uv run autoresearch --root .demo validate
```

`calibrate`, `search`, and `promote` seal immutable manifests and enqueue them; they do not
run the commands inline. The first `run` lands and judges the control. The second scores the
candidate against its frozen control. `promote exp_better` is accepted only because the
candidate cleared the bank gate, then stages paired confirmation on default seeds 43–47.
The resulting demo SOTA belongs only to `demo_cpu_val_bpb_v2`.

A protocol-v2 spec may remain unsealed, but once sealed it may have only one immutable
manifest. Sealing atomically rejects a different second manifest; `validate` also rejects
any imported/adversarial registry containing multiples, and promotion requires the sole
manifest.

## Why the loop is faster

- `run` owns a resident worker pool. It can drain many queued manifests without an agent or
  shell loop between launches.
- `calibrate` builds one reusable control per declared resource/GPU slot. A control remains
  fresh for one hour. The bank view caps landed candidate uses at eight; atomic search
  staging also counts active pending/running/waiting reservations against that cap.
- `search` runs a cheap seed-42 candidate and freezes the exact same-slot control before
  launch. There is no dynamic baseline substitution.
- Only a candidate that improves `val_bpb` by more than `0.000426` enters promotion.
- `doctor` exposes whether control timing is healthy before expensive search. Controls with
  missing or inconsistent timing cannot be selected. When training owns less than 75% of
  elapsed time, only explicitly labeled data/compile/evaluation/input/instrumentation/
  tokenizer bottleneck work proceeds; other subsystems require the operator to use
  `--allow-overhead-dominated`.
- Promotion spends multiple seeds only on a gate-clearing candidate. The default is five
  distinct seeds with four required valid replicates; the protocol floor is three distinct,
  valid, verified seeds.

GPU bank identity uses the declared resource ID plus the physical GPU UUID observed at
launch, not a remappable GPU index. The CPU fallback in the demo exists only so the policy
can be smoke-tested locally.

## Freeze a real scope first

Start from `templates/scope_karpathy_autoresearch.json` for the default baseline, or
`templates/scope_v2.json` for another trainer. A v2 scope names the hardware class, dataset
split, tokenizer, evaluator, precision, primary `val_bpb` metric, and timing budget. Its
`id` becomes the comparison group. Every execution resource must declare the same
`hardware_class`. If argv contains `--time-budget`, sealing requires it to equal the scope
budget. A `wall_seconds` scope checks emitted total time and the independent runner clock
against the frame. Karpathy's `training_seconds` scope checks its charged inner training
clock while independently requiring the complete process wall time to reconcile with its
emitted total time.

If any comparison-defining field changes, assign a new scope ID and therefore a new
comparison group. A score from a different shard range, tokenizer artifact, evaluator,
precision, hardware class, or budget is not a new record in the old leaderboard. It is a
different experiment family.

The candidate must name every intentionally edited binding with `--mutable-code-path` and
its argv must enter directly through a sealed relative execution path. The accepted grammar
is intentionally narrow. In the CLI, `@python` expands to the exact current Python
executable. Remote runtimes may declare an explicit `.venv/bin/python` (or another explicit
path) in every resource and use that exact token. `/bin/bash sealed_launcher.sh` is the
escape hatch for complex startup. Bare PATH-resolved launchers, `-c`/`-m` indirection,
unsafe option prefixes, unsealed positional programs, absolute payloads, and arm overrides
of `PATH`/module-loader variables are rejected; v2 launch also removes those variables from
the inherited child environment. Stable code, all data, requirements, metric,
comparison group, and scope remain in the exact-context fingerprint. Keeping the same argv
while changing declared mutable bytes is valid; keeping both unchanged is a rejected no-op.

## Resident queue

Use multiple workers and optionally keep them resident for work staged by another process:

```bash
uv run autoresearch --root .autoresearch run \
  --workers 8 \
  --follow \
  --poll-seconds 1 \
  --idle-timeout-seconds 600
```

`run` defaults to four workers. Resource, host-slot, GPU, and workdir leases can reduce the
actual width; `--workers` is an upper bound, not a claim of physical capacity.

Useful inspections are:

```bash
uv run autoresearch --root .autoresearch queue --jobs
uv run autoresearch --root .autoresearch queue --reconcile
uv run autoresearch --root .autoresearch status
uv run autoresearch --root .autoresearch bank
```

Queue status reports depth, active and blocked jobs, rolling health, and last progress;
`queue --jobs` adds every atomic job record.

An existing low-level manifest can be added with `enqueue MANIFEST_ID ...`. Each immutable
manifest maps idempotently to one deterministic job under `operational/queue/`. Queue state
is operational; ResultBundle and EvidenceDecision records remain the scientific authority.

No resource means no arm launched, so a job enters `waiting` with bounded exponential
backoff. An executed invalid or unknown replicate is landed and never automatically retried.
An uncertain runner/evidence failure is `blocked` and surfaced. Reconciliation only requeues
work when it can prove that execution did not cross a durable boundary.

The rolling circuit watches the latest decisions for the last 12 pilot results. It pauses
new claims after three consecutive invalid/unknown pilots, or when invalid/unknown exceeds
25% once at least eight pilots exist. Already-running work is not killed. `run
--ignore-health` is an explicit recovery override; it does not weaken evidence validation.

## `search` is the v2 path; `screen` is legacy

`search` is the recommended workflow. It requires a structured scope and fresh exact bank
control, freezes that control into the candidate spec, enqueues the manifest, scores the
result, and feeds a distinct-seed promotion path. For production model subsystems it also
requires `--hypothesis-id`; the CPU smoke test is intentionally exempt because it validates
the harness rather than making a scientific claim.

`screen` remains for v1 compatibility. It synchronously runs one arm at one seed and updates
the legacy leaderboard. It does not provide the v2 bank, scope, profile, queue, or promotion
contract. Existing scripts can keep using it, but new campaigns should not build on it.

The lower-level `design`, `review-template`, `seal`, `execute`, `judge`, `cycle`, and `paper`
commands also remain available. V2 calibrations, candidates, and promotions do not require
the v1 five-review council or paper cadence. A promotion may still receive an optional
digest-bound `--reviews` declaration, and knowledge synthesis still reports legacy paper
status, but reviews and papers are not in the v2 search hot path.

## Shared hosts and real exclusivity

OPHIS coordinates workers that share one state root with `flock`-protected host slots and
resource/GPU leases. `host_id` groups resources under a cooperative
`max_concurrent_jobs` cap. This prevents cooperating OPHIS workers from colliding; it does
not reserve a physical host or GPU against other state roots, users, processes, or a cluster
scheduler.

Setting `reservation.mode` to `externally_reserved` records an allocation ID but does not
obtain one. Acquire the real scheduler/cloud reservation first.

On a shared host, start from `templates/execution_remote_shared_single.json`, which limits
OPHIS to one job. Use `templates/execution_remote_fleet.json` only after arranging real
external exclusivity and profiling the host at the intended width. A workdir lock serializes
cooperating workers that share a directory; concurrent resources therefore need separate
workdirs for throughput. Separate workdirs still do not isolate host CPU, storage, network,
or compilation contention.

## Evidence and claim boundary

A completed queue job says execution and judging finished. A valid pilot says a measurement
is usable. A bank row marked `promotion_due` says a candidate is worth confirming. None of
those is SOTA.

V2 SOTA requires a supported promotion confirmation with preregistered distinct seeds,
verified emitted seed metrics, valid evidence, and at least three valid replicates. Knowledge
waits until every planned replicate has a terminal result and decision, then uses the valid
confirmation aggregate inside its structured scope; it never promotes the best single seed
or the seed-42 bank delta.

Knowledge does not trust a `promotion` label or copied source delta. Before publishing SOTA,
it reconstructs the sole candidate and promotion manifests, frozen control and bank score,
fingerprints and delta, scope and requirements, code/data snapshots, argv/env, held-out
seeds, and AB/BA order from immutable records. A failed lineage check is reported in
`sota_blockers` instead of becoming SOTA.

The current protocol-v2 authority is `evidence-v3`. Historical `evidence-v2` decisions stay
immutable and inspectable, but bank, belief, and SOTA views exclude them. After opening an
older v2 state, run `autoresearch --root STATE judge --all` before `synthesize` to create the
current decisions from the original ResultBundles.

## Source map

```text
autoresearch/
  records.py      schemas and digest validation
  store.py        immutable records, blobs, locks, leases, and views
  research.py     ExperimentSpec and optional Paper registration
  protocol.py     v2 scope, seed, metric, gate, and profile policy
  workflow.py     calibrate/search/promote staging
  sealing.py      immutable manifests and code/data snapshots
  campaign.py     durable queue, resident workers, recovery, health circuit
  execution.py    claims, resource selection, local/SSH launch, telemetry
  evidence.py     validity and provenance decisions
  bank.py         rebuildable controls and promotion candidates
  doctor.py       calibration/profile diagnostics
  knowledge.py    beliefs, portfolio, and scope-bounded SOTA
  science.py      literature retrieval, claim confidence, mechanisms, hypotheses, gaps
  loop.py         legacy synchronous screen path
  chart.py        legacy leaderboard rendering
  cli.py          command surface
```

For design invariants and migration notes, see `docs/ARCHITECTURE.md` and
`docs/V2_DESIGN.md`. The exact claim ontology, confidence formula, thresholds, experimental
updates, limitations, and agent task policy are specified in `docs/SCIENTIFIC_METHOD.md`.

## Development

```bash
uv run pytest
uv run ruff check autoresearch tests
```
