# Lessons taken, and what changed in the v2 workflow

Written 2026-08-16, after reading `simplify_autoresearch_v2` end to end and
mining `simplify_autoresearch_v1` and `vibeautoresearch_reimplementation` for
what those campaigns actually cost themselves.

The question this document answers is not "is v2 good". It is: **v2 is 12,400
lines of well-tested provenance machinery that has never executed a single real
experiment — do we run it, and if so, what has to change first?**

---

## 1. What the predecessor campaigns paid for

Seven findings, each with a number attached, each from a written post-mortem
rather than from recollection.

**L1 — Idle GPU is the dominant loss, by an order of magnitude.**
`simplify_autoresearch_v1/24_HOUR_CONCLUSION.md`: 48 result bundles in 21.6 h;
12.8 of those hours idle; every one of the 48 on **one GPU of an eight-GPU box**.
About **5% of available compute**. The individual experiments were already ~79%
efficient against their floor. The loss was entirely between experiments, and its
root cause was named exactly: *the agent was in the critical path between one
experiment finishing and the next starting.*

**L2 — Contention is the dominant noise, and only same-GPU adjacency cancels it.**
Same host, load average 198 with 27 users; four arms died `rc=255` on SSH drops;
two controls starved to 73–84% of clean step count. A contended window moves
`val_bpb` by **~0.02** — four times an entire day's SOTA gain. `plain_v2` measured
the same thing from the other side: same-seed repeat sigma ~0.0001, but seed-42
controls across a campaign span **0.013**. That is 130×, and it is the machine,
not the model.

**L3 — Winner's curse is quantified, not hypothetical.**
`REFLECTION_3`: a 4-arm sweep read at its best arm at n=4 is inflated by
1.029σ/√4 ≈ 0.26 gates. Two stacked selections predicted 0.52 gates and the
observed regression on disjoint seeds was 0.44 — fully explained. The adoption
gate is calibrated for **one pre-specified comparison** and was being applied to
maxima of k arms.

**L4 — A literature filter does not compound.**
`CAMPAIGN_REFLECTION_20260731`: **5.7% of the literature corpus ever did
predictive work** — 123 external claims, 7 ever bound to a mechanism. The two
durable results of that campaign both came from *measurement* (the fill-flood
mechanism; the 1e-3-vs-1e-2 selection rule derived from sixteen failures), not
from picking a claim by scope match. Also: fifteen hours of experiments ran before
the single most relevant paper in the corpus was opened.

**L5 — The knob box is closed; only mechanisms move the number.**
`REFLECTION_2`: twenty-one closed directions reduce to one free parameter — how
many tokens you consume before the repetition wall. Intrinsic refinements topped
out at ±1.3e-3 against a 2.6e-3 gate. A prior campaign spent its last ~60 runs
producing a top-30 leaderboard whose entire spread was ~3× its own control noise:
an unordered tie. Sixty runs that measured nothing.

**L6 — Written rules fail; structure works.**
The sharpest line in the v1 post-mortem: a failure class stops recurring *only
when the mechanism changes*. "Arm the wakeup every turn" was applied perfectly and
the class recurred anyway. Every recurring failure in the vibe campaign was fixed
twice — once with a rule that failed, once with a mechanism that worked.

**L7 — The agent's own error rate is high and the record model is what catches it.**
Seven self-inflicted errors in a single block, most caught within minutes by
refusals: an unpublished sixth confirmation, a five-hat review council, a
three-replicate pilot, an overwritten paper record, a screen whose two arms were
identical.

## 2. Scoring v2 against those lessons

| lesson | v2 as shipped | verdict |
|---|---|---|
| L1 idle | durable queue + resident `run --follow`; agent structurally out of the launch path | **wins** — this is the direct fix |
| L2 contention | bank keyed by *physical GPU UUID*, control frozen **before** launch, no cross-GPU fallback | **wins** — enforces what v1 only recommended |
| L3 winner's curse | promotion on held-out seeds 43–47, preregistered success rule, AB/BA counterbalance, aggregate mean not best seed, lineage recomputed from immutable records | **wins decisively** — `plain_v2` has no protection here at all |
| L6 structural | the entire design philosophy: immutable seal, no-op rejection, one manifest per spec | **wins** |
| L7 error rate | sealing, executable seed verification, scope digests, duplicate-option rejection | **wins** |
| L5 mechanisms vs knobs | **nothing.** `--direction` and `--subsystem` are free text. No exploration budget exists. | **loses to `plain_v2`**, whose `batch.py` refuses a batch that violates ≥5/8 |
| L4 literature | a full agenda→search→source→claim→mechanism→hypothesis ontology gated *in front of* GPU search | **risk** — this is the 5.7% failure mode with more machinery around it |

So v2 is the right vehicle on five of seven, and is uniquely the only one of the
three systems that protects against L3, which is the failure that produced a
*withdrawn published result*. That settles the vehicle question.

Four things had to change before running it.

---

## 3. Change 1 — bank TTL 60 min → 20 min, max uses 8 → 3

`autoresearch/bank.py`

This is the change that matters most, and shipping v2 unmodified on this host
would have quietly wasted most of its promotion budget.

v2's banked control is the comparator for every candidate. As shipped it stays
eligible for **one hour** and **eight uses**. Set that against L2's measurements:

```
same-seed repeat sigma, quiet slot ......... ~0.0001
seed-42 control spread over a campaign ..... ~0.013     (130x)
a contended window moves val_bpb by ........ ~0.02      (47x the gate)
the promotion gate is ...................... 0.000426
```

A one-hour-old control on a host that has been observed at load average 281 with
16 users carries **more contention drift than the gate it is being compared
against**. The bank would have nominated noise for promotion. Promotion (5 seeds,
both arms, same slot, AB/BA) would then have killed those nominations correctly —
so the cost is not false SOTA, it is ~65 GPU-minutes burned per phantom.

Twenty minutes with three uses keeps a control adjacent in time to the candidates
it scores. The cost is roughly one control slot in four — which is exactly the
control:treatment ratio a paired-wave campaign on this same host converged to
empirically. The constants carry that derivation in a comment so nobody raises
them later to "save control runs".

## 4. Change 2 — a real exploration budget, enforced inside the staging lock

`autoresearch/exploration.py` (new), wired into `V2Workflow._stage_candidate_locked`

This is v2's one genuine gap. It has extremely strong guarantees about *how* a
thing is measured and no opinion whatsoever about *what* gets measured — which is
precisely the axis on which both predecessor campaigns failed (L5).

`search` now requires `--track {mechanism,knob,throughput}` and takes an optional
`--family` (defaulting to `--subsystem`). Three rules fire **inside the same
reservation lock** that pins a control and seals a manifest, so a refused
candidate cannot consume a bank use or race another staging process:

1. **Breadth** — a non-mechanism candidate is refused unless ≥5 of the trailing 8
   *staged* candidates were mechanisms. Counting staged rather than landed matters:
   waiting for results before counting a slot is exactly how a queue drains into a
   knob sweep while the first knob is still running.
2. **No premature knob tuning** — a `knob` candidate is refused unless its family
   already cleared the bank gate. Tuning a constant whose mechanism is still inside
   the noise band is the motion that burned those sixty runs.
3. **No repair loop** — at most 2 consecutive candidates from one uncleared family.
   "A mechanism that fails is a result."

Overrides exist and are never silent: they demand a written reason of ≥20
characters which is copied into the immutable ExperimentSpec, alongside the full
budget report, so an audit can distinguish "passed cleanly" from "overridden".

Why in code rather than in `HEARTBEAT.md`? L6. The predecessor wrote this rule
down, followed it, and converged onto knobs anyway.

## 5. Change 3 — the science gate is demoted from launch gate to standing duty

Policy, not code. `--allow-weak-science` + a registered `--hypothesis-id` is the
default posture for this campaign.

v2's default refuses production model search until a hypothesis's agenda has met
12 sources / 6 claims per topic with full-text snapshots and foundation confidence
≥ 0.60. On L4's own evidence that gate would consume the session and would not
improve candidate selection — 116 of 123 claims in the campaign that built this
ontology were inert.

But the gate is not worthless; the *cheap half* of it is where the value was.
`--allow-weak-science` still requires `--hypothesis-id`, so every candidate still
carries a registered statement, prediction, falsifier, and competing explanation
before it runs. That is the part v1 explicitly praised — "falsifiers registered
before running... it fired exactly as written" — and it costs minutes, not hours.

Literature reading moves to a standing per-heartbeat duty running **concurrently**
with the GPUs (`run --science-agenda`, plus a scout agent), which is where L4 said
it belonged: the useful paper was found by reading during the run, not by gating
the run on reading.

## 6. Change 4 — the scope had to be pinned before it could be frozen

`runs/code/prepare.py`, `runs/scope.json`

Found while wiring the benchmark: `prepare.py` builds the training stream from
`sorted(os.listdir(DATA_DIR))`. The **contents of a directory silently define
`dataset_split`**. The shared `~/.cache/autoresearch` on this host currently holds
41 shards, not the intended 10, and another live campaign writes to it.

Freezing a scope whose definition another process can change is not freezing
anything. So `CACHE_DIR` is now a constant pointing at `~/.cache/autoresearch_v2`,
holding exactly shards 00000–00009 + 06542 — a constant rather than an environment
lookup, so the sealed `prepare.py` digest fully determines which corpus a run
consumed. `shard_00000` was missing and unreachable from this host
(`huggingface.co` resets the connection); it came from `hf-mirror.com`.

The adapter change is recorded in `runs/code/provenance.json`, and the two
`test_karpathy_baseline` tests caught the digest drift immediately and refused to
pass until provenance and the scope template were updated — the record model doing
its job (L7).

## 7. What was deliberately *not* changed

- **`profile_health`'s 25% overhead limit.** A 300s training frame plus ~90s of
  `max-autotune` compile plus eval will likely land at 22–33% overhead and trip
  this gate. The temptation is to pre-emptively widen it. Weakening a gate before
  seeing its measurement is exactly backwards; the calibration controls now running
  will say what the real number is, and if the gate is genuinely unreachable it
  gets raised *with the measurement attached*.
- **The health circuit** (3 consecutive invalid, or >25% of 12). On a box with SSH
  drops this will fire. It should fire — the answer is to diagnose, and to keep the
  queue deep so a pause is visible rather than silent.
- **`plain_v2`.** It is running on GPUs 1, 2, 5, 6 at ~100% duty with a replicated
  SOTA. It is left completely alone. This campaign took the four idle GPUs.
- **Judging on measured `val_bpb` only.** v1 built a token-law control variate that
  cuts contention noise 6.5×; `plain_v2` later banned all derived metrics from
  ranking. The later, more considered rule wins: `num_steps` is a diagnostic beside
  every run, never an adjustment to it.

---

## 8. The resulting structure

```
                    ┌─── scientific library (CPU, concurrent, never gates a launch)
                    │    agenda → literature_search → source → claim
                    │            → mechanism graph → hypothesis
                    │                                    │
                    │                        --hypothesis-id (required)
                    ▼                                    ▼
  runs/scope.json ────────────► FROZEN SCOPE  ──►  exploration budget
  (id, hardware_class, split,    digest a3cb6a9b       (mechanism / knob /
   tokenizer, evaluator,                                throughput; breadth,
   precision, val_bpb/minimize,                         no-premature-knob,
   training_seconds 300)                                no-repair-loop)
                                                             │
   runs/execution_fleet4.json ──────────────────────────────►│  ← inside ONE lock:
   (4 resources = GPU 0,3,4,7; ssh; workdir; .venv/bin/python)│    select control,
                                                             │    seal manifest,
                                                             ▼    admit to queue
     ┌──────────────────────── DURABLE QUEUE (operational/queue/) ─────────────┐
     │  pending → running → complete / waiting (no resource) / blocked         │
     │  resident worker pool, --follow, agent NEVER in the launch path         │
     └────────────────────────────────┬────────────────────────────────────────┘
                                      ▼
          calibrate ──► BANK: one control per physical GPU UUID
                        TTL 20 min, max 3 uses, no cross-GPU fallback
                                      │
          search    ──► CANDIDATE: 1 arm, seed 42, scored ONLY against the
                        same-slot control frozen before launch
                        delta < -0.000426  →  promotion_due (a work decision,
                                              sota_eligible: false)
                                      │
          promote   ──► CONFIRMATION: held-out seeds 43-47, both arms every
                        seed, AB/BA counterbalanced, ≥3 valid, preregistered
                        success rule, mean not best seed
                                      │
                                      ▼
       ResultBundle ─► EvidenceDecision ─► Knowledge recomputes lineage from
                                            immutable records ─► SOTA or
                                                                 sota_blockers
                                      │
          conclude  ──► experiment-origin claim rescored into the belief ledger
```

Authority is asymmetric on purpose: queue state says what should run next,
ResultBundle says what ran, EvidenceDecision says whether it is usable, the bank
says whether it deserves seeds, and only a supported held-out-seed confirmation
may touch SOTA.

## 9. Files changed

| file | change |
|---|---|
| `autoresearch/exploration.py` | **new** — the exploration budget and its three rules |
| `autoresearch/workflow.py` | budget enforced in the staging lock; `_staged_candidates`, `_gate_cleared_families`; track/family recorded in the immutable spec |
| `autoresearch/cli.py` | `search --track / --family / --exploration-override` |
| `autoresearch/bank.py` | TTL 3600→1200 s, max uses 8→3, with the derivation in a comment |
| `runs/code/prepare.py` | `CACHE_DIR` pinned to `~/.cache/autoresearch_v2` |
| `runs/code/provenance.json` | new adapter id and the three adapter changes enumerated |
| `templates/scope_karpathy_autoresearch.json` | evaluator digest follows the adapter |
| `runs/scope.json`, `runs/execution_fleet4.json`, `runs/data_manifest.json` | **new** — the live frozen campaign configuration |
| `tests/test_exploration_budget.py` | **new** — 7 tests over the budget |
| `tests/test_workflow.py` | the cap test opens a distinct family per arm so it measures the cap, not the budget |
| `HEARTBEAT.md` | **new** — the operating loop |

124 tests pass; `ruff check autoresearch` is clean apart from three pre-existing
`ISC004` findings in `science.py`.

---

## 10. First measurement: the profile gate was right, and it found the lever

Four calibration controls, one per physical GPU, seed 42, all judged **valid**:

| slot | val_bpb | num_steps | training_s | total_s | overhead | MFU |
|---|---|---|---|---|---|---|
| h200_gpu0 | 1.16968 | 192 | 301.6 | 486.9 | 38.1% | 7.64% |
| h200_gpu3 | 1.17102 | 190 | 300.1 | 491.7 | 39.0% | 7.60% |
| h200_gpu4 | 1.16984 | 192 | 301.5 | 493.6 | 38.9% | 7.65% |
| h200_gpu7 | 1.16890 | 192 | 300.9 | 497.0 | 39.5% | 7.66% |

`doctor` returns `model_search_ready: false`, `profile_states:
{overhead_dominated: 4}`. §7 said the gate would probably trip and that it would
not be widened before it had been measured. It tripped, and widening it would have
been the wrong call, because the measurement is not a nuisance — it is the finding:

- **MFU 7.6%.** The `plain_v2` campaign on this same host, same 300s frame, runs at
  **39–44% MFU and ~1450 steps**. This baseline gets **192**. That is a 7.5x gap in
  token exposure, and under a `training_seconds` budget token exposure is the
  currency.
- **The gap is independently explained.** `plain_v2`'s own SOTA writeup attributes
  "the large majority" of its 0.0134 improvement to the **bucketed packer plus token
  cache** — an input-pipeline fix. Upstream `prepare.py` reads parquet and tokenizes
  in the training loop. Two of that campaign's three banked wins were input-pipeline,
  and *fourteen architectural mechanisms were measured and thirteen did nothing.*
- **The four slots agree to a spread of 0.0021 in val_bpb and 190–192 steps**, so
  this is the configuration, not a contended window — even at host load average 361.

So the honest reading of the first wave is that **the largest available lever on
this baseline is the input pipeline, not the architecture**, and the harness reached
that conclusion from four controls without being told. This is the fill-flood shape
from the vibe campaign repeating: measurement defects get fixed upstream, not
screened around.

### A contradiction between two of my own gates, and its fix

Acting on that finding immediately exposed a bug in Change 2. `doctor` refuses model
search while a control is overhead-dominated and permits only
data/compile/evaluation/input/instrumentation/tokenizer work. The breadth rule
simultaneously refused any non-`mechanism` candidate for the first five slots. A
campaign whose first calibration comes back overhead-dominated could therefore stage
**nothing at all**.

Fixed by exempting diagnostic subsystems from all three budget rules, using the same
prefix list v2 already uses to exempt them from `--hypothesis-id`
(`exploration.DIAGNOSTIC_SUBSYSTEM_PREFIXES`). The justification is that these are
not model search, so spending a slot on one is not spending a slot instead of
exploring. Covered by `test_diagnostic_subsystems_are_exempt_so_the_two_gates_do_not_contradict`.

### Also learned, and now recorded

`env -u PATH` in the v2 launch path kills `torch.compile`: Triton shells out to
`/bin/gcc`, which needs PATH to find `ld`, and every arm died at `collect2: fatal
error: cannot find 'ld'` about two minutes in with an empty metrics dict. `libcuda`
was never the problem. The fix is the protocol's own escape hatch — a sealed
`launch.sh` entered as `/bin/bash launch.sh` — not a weakened scrubbing rule.
