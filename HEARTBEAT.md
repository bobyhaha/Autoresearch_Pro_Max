# HEARTBEAT — the loop the research director runs

> ## ⏱ EVERY 10 MINUTES: CHECK WHETHER YOU ARE RUNNING OR NOT.
>
> Not every heartbeat. Not when convenient. **Every ten minutes**, on the clock,
> including in the middle of writing, analysing, or reasoning about something else.
> Two separate questions, and both must be answered *yes*:
>
> 1. **Is the loop running?** Is there a live resident worker pool, and am I still
>    driving it — or did the pool exit, the wake path fail to arm, or the turn end
>    without anything scheduled to happen next?
> 2. **Is the GPU running?** Are our four GPUs executing our work right now — or is
>    the queue empty, the circuit paused, or the box holding leases with nothing on them?
>
> The failure this exists to prevent has already happened twice in this project: a
> predecessor campaign spent **12.8 of 21.6 hours idle** and used about **5% of the
> compute it had**, and in this very session the pool drained and exited while
> analysis continued, leaving all four GPUs at 0% until the next manual check caught
> it. Nothing else in this file costs as much as getting this wrong.

**Read this file at the top of every heartbeat.** It is the operating layer above
`program.md`. `program.md` says what the record model permits; this file says what
to do with the next twenty minutes.

The campaign runs on GPUs **0, 3, 4, 7** of a shared 8×H200 host. GPUs 1, 2, 5, 6
belong to the `plain_v2` campaign and are **not ours** — never launch on them,
never kill their workers, and count them as occupied in every free-GPU check.

---

## 1. Liveness — am I running? are the GPUs running? Every 10 minutes.

**The single most expensive failure available to this campaign is an idle GPU.**
The predecessor campaign measured it: 48 experiments in 21.6 hours, 12.8 of those
hours idle, on one GPU of an eight-GPU box — about **5% of the compute it had**.
Not one bad experiment in that campaign cost as much as the dead time between the
good ones.

Every ten minutes, answer all five of these. If any answer is wrong, fix it
**before** analysis, before writing, before staging the next candidate:

```bash
# 0. AM I RUNNING?  is a resident pool alive, and is a wake path armed so that
#    something happens next even if I stop typing?
pgrep -fl 'autoresearch .*run --workers' || echo 'POOL DEAD — restart it NOW'

# 1. ARE THE GPUS RUNNING?  our four UUIDs should each have a compute process
ssh -i "$OPHIS_SSH_KEY" -o IdentitiesOnly=yes -p "$OPHIS_SSH_PORT" "$OPHIS_SSH_TARGET" \
  'nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader'

# 2. is there work for the pool to pick up?
uv run autoresearch --root .autoresearch queue --jobs

# 3. did the circuit breaker pause new claims?
uv run autoresearch --root .autoresearch status

# 4. is anything actually progressing, or are leases held with nothing on them?
uv run autoresearch --root .autoresearch claims
```

Expected: a live `run --follow` process, four of our GPU UUIDs busy, queue depth
≥ 4, health not paused, no claim older than one arm's runtime.

**A live pool is not the same as running work, and running work is not the same as
progress.** All three fail differently: the pool can exit on idle timeout while the
queue is empty; the queue can be deep while the circuit is paused; a claim can be
held by a dead runner while the GPU sits at 0%. Check them separately — that is why
this is five commands and not one.

**Restarting the pool is always cheap and always safe.** The queue is durable and
jobs are idempotent, so if there is any doubt whether the pool is alive, restart it
rather than investigating first:

```bash
nohup uv run autoresearch --root .autoresearch run \
  --workers 4 --follow --poll-seconds 5 --idle-timeout-seconds 7200 \
  > runs/worker_pool.log 2>&1 &
```

**Keep the queue at least one full wave deep at all times.** A mediocre candidate
that runs beats a good one that does not. Queue speculative work rather than let
the pool drain — but queue it as `--track mechanism`, because the exploration
budget will refuse a speculative knob and it is right to.

**Set `--idle-timeout-seconds` longer than any gap you expect to leave.** A pool
that exits on idle is indistinguishable, ten minutes later, from a pool that
crashed — and this session lost four GPUs to exactly that.

If the health circuit has paused claims, diagnose before overriding: read
`doctor`, `queue --jobs`, the resource leases, and the last few EvidenceDecisions.
`run --ignore-health` is a recovery tool, not a way to keep the number moving.

If SSH fails, **retry until the box is back**. An SSH drop must never end a
heartbeat or silently stall the loop. If the host is genuinely unreachable for
more than ~15 minutes, say so as the headline of the next report rather than
quietly waiting.

### 1.1 The 10-minute wake is code, not a resolution

Two independent mechanisms, because they fail differently. Both must be armed at
the top of every session, and arming them is the **first** action of the session:

**(a) `tools/watchdog.py` — the unattended repair loop.** Runs headless every 10
minutes, auto-restarts a dead pool, and writes `runs/watchdog.log` +
`runs/WATCHDOG_STATUS.json`. It cannot decide *what* to run, so it escalates
`IDLE_GPUS_EMPTY_QUEUE` and stops there.

```bash
pgrep -f 'tools/watchdog.py' || nohup uv run python tools/watchdog.py \
  --interval-seconds 600 --workers 4 --idle-timeout-seconds 14400 \
  > runs/watchdog.out 2>&1 &
```

**(b) `tools/wake.py` — the thing that wakes *the agent*.** A watchdog that
restarts a pool does not fix an empty queue, a paused circuit, or a landed result
nobody read. Those need a research agent, so something must re-enter the loop on
the clock. `tools/wake.py` emits the one-screen liveness briefing the agent reads
on waking, and exits non-zero when the campaign needs a decision:

```bash
uv run python tools/wake.py          # the briefing; exit 3 == needs a decision
uv run python tools/wake.py --arm    # print the schedule command to install
```

Arm it for the session with the harness scheduler, which is what actually
delivers the wake:

```
/loop 10m uv run python tools/wake.py   # then act on anything it flags
```

Equivalently, a cron-scheduled agent at `*/10 * * * *` running the same command.
**A session with neither armed is a session that will go idle** — that is the
failure this whole file exists to prevent, and it has already happened twice.

**(c) `tools/gpu_guard.py` — the one that actually guarantees GPU work.** (a) restarts
a dead pool but will not queue anything; (b) needs an agent to be listening. Between
them sat the failure that has now happened **three** times: queue drains, every GPU
goes to 0%, and nothing happens until someone notices. The third time was this
session — a loop was cancelled during a reset and never re-armed, and the fleet idled
two hours.

```bash
pgrep -f 'tools/gpu_guard.py' || nohup uv run python tools/gpu_guard.py \
  --interval-seconds 600 > runs/gpu_guard.out 2>&1 &
uv run python tools/gpu_guard.py --once --dry-run   # say what it would do
```

Every 10 minutes it restarts a dead pool, then: our training processes alive → log and
exit; nothing running but work queued → resource-wait or setup, leave it alone; nothing
running and **nothing queued** → **stage a calibration wave**.

**It stages calibration controls ONLY, never a candidate.** That is the entire safety
argument, and it is why this does not violate "adding a candidate to the queue is a
scientific decision". A candidate encodes a hypothesis and choosing one is research; a
control re-measures a baseline already sealed into the scope. Re-measuring pays twice —
it keeps the fleet warm, and every replicate tightens the noise floor that currently
blocks reading any candidate at all (σ 0.00556 against a 0.000426 gate).

It escalates rather than persisting when staging cannot be the answer: a paused health
circuit, an unreadable store, an unreachable host, or `MAX_CONSECUTIVE_STAGES` waves
with no new landed result — because burning GPU on controls that never land is keeping
the fleet warm for show. Writes `runs/gpu_guard.log` and `runs/GPU_GUARD_STATE.json`.

## 2. Analysis and critique run in parallel with the GPUs, never in series

GPU time is the scarce resource; analysis is nearly free. Nothing in the analysis
path may sit between one experiment finishing and the next starting — that is what
the durable queue and `run --follow` exist to prevent.

**Use multiple agents for analysis and critique.** A single reader of one's own
results is the failure mode that produced a withdrawn result and a retracted
knowledge card in the predecessor campaigns. Every heartbeat that has results to
interpret dispatches, concurrently:

- an **analyst** — what do the landed results say, in numbers, against the frozen
  control and the measured noise floor;
- a **critic** — what confound, selection effect, or contended window explains
  this result *as well as* the proposed mechanism does;
- a **scout** — one new primary source from outside the repo, with its scope
  compared to this frame in numbers, converted into a concrete candidate.

They run while experiments run. Their disagreement is the useful output; where the
analyst and the critic agree, the result is probably real, and where they do not,
that is the next measurement.

## 3. Read papers first, and keep reading

**Before the search phase begins, read broadly.** Not one paper to justify a
candidate already chosen — a real reading pass over the mechanism space, up
front, while the calibration controls are burning GPU time and costing nothing to
overlap with. The order is *literature → claims → mechanisms → hypotheses →
experiments*, and this campaign has repeatedly run it backwards: pick a knob,
then find a citation for it. That is how a predecessor ended with 123 claims of
which 5.7% ever did predictive work.

Reading is free relative to GPU time and **must never be in series with it**. The
runner does this for you — literature refresh runs concurrently with the drain:

```bash
uv run autoresearch --root .autoresearch run --workers 4 --follow \
  --science-agenda runs/science/agenda.json --literature-limit 25 \
  --openalex-mailto baiyuzhu@mit.edu
uv run autoresearch --root .autoresearch literature-search "<query>" --limit 25
```

Target for the opening block: **enough sources that the mechanism space is
covered, not enough to justify one candidate.** A reading pass that produces no
claim the campaign did not already believe was not a reading pass.

### 3.0 Rate: at least 30 papers per hour, analysed for mechanism

**Read no fewer than 30 papers per hour while the GPUs run.** Reading is CPU-free and
overlaps with training, so anything less is leaving the cheap half of the campaign on
the floor. Bias hard toward **2025–2026**: this benchmark's levers are current
research, and a 2019 result about a 10× larger model at unbounded horizon transfers
badly to 50M parameters under 300 seconds.

```bash
uv run python tools/read_arxiv.py --plan --from-year 2025 --limit 30
```

**Retrieved is not read, and the count is the least interesting number.** Two traps,
both hit for real on 2026-08-17:

1. **The metadata provider served corrupted abstracts.** Correct titles, authors,
   years and DOIs — with the abstract of a *different paper* attached. "Scaling Laws
   for Neural Language Models" (2001.08361) came back describing a
   "transport-validity theory for agentic AI interventions". A claim extracted from
   that text would carry a real DOI and real authors on a fabricated statement:
   provenance that looks impeccable and is worthless. **Content comes from the
   publisher** — `tools/read_arxiv.py` reads the arXiv Atom API directly. Discovery
   may come from anywhere; content may not.
2. **Bulk retrieval buys quantity, not relevance.** Of 71 papers pulled in one pass,
   roughly 13 were on-topic and the rest included sea-ice classification and audio
   synthesis. 30/hour is a floor on *reading*, not a licence to register 30 rows.

**Analyse for mechanism, not for result.** For each paper that survives the relevance
filter, answer these before writing anything down:

- **What is the causal story?** Not "X improved loss by Y" but *through what
  mediator*. A result without a mediator cannot be transferred, only cargo-culted.
- **Does the mediator exist in OUR frame?** 50.33M parameters, 300 charged seconds,
  bf16, Pre-Norm RMSNorm without a learnable scale, ReLU² MLP, SSSL sliding window,
  MuonAdamW, and a step count that is *not known in advance* and varied 613–698
  across three identical controls. A mediator that runs through batch size at 7B, or
  through an iteration horizon fixed in advance, is not present here.
- **What would it cost us?** Under a fixed-time budget, any change that adds per-step
  work must pay for the steps it destroys. An optimization win at unbounded horizon
  can be a loss at 300 seconds.
- **What is the cheapest falsifier?** Write it before the experiment, and prefer the
  one that indicts the *mediator* rather than the metric.

**Read for the gap, not the confirmation.** The most valuable finding so far came
from noticing a component our baseline *lacks entirely* — `norm()` is a bare
`F.rms_norm` with no learnable scale, in a Pre-Norm net, which is exactly the
ablation arXiv 2605.26895 reports as substantially degrading pre-training. Look for
what the paper assumes that we do not have.

**Know why your confidence is capped.** `confidence-v1` weights reproduction at more
than three times venue, and content depth scores `fulltext_snapshot` 0.95 against
`abstract_only` 0.55. Two abstract-only claims from arXiv preprints scored 0.71
(`moderate`) — the ceiling for abstract-only, single-source evidence, no matter how
good the paper is. **To move a belief past ~0.75 you must read the full text**, and
`RESEARCH_TASKS.json` will keep emitting `read_fulltext_and_extract_claims` at
priority 85 until you do. Deep analysis is what raises confidence; volume is not.

### 3.0b Read the FULL TEXT, and take several claims from each paper

An abstract is an advertisement. Reading one and registering a claim from it is not
reading a paper, and the scoring says so: `content_depth` scores `abstract_only` 0.55
against `fulltext_snapshot` 0.95.

**One paper is worth several claims.** A paper that only yields one claim was skimmed.
Reading arXiv 2605.26895 in full turned a single vague claim into four sharp ones:

| from the abstract | from the full text |
|---|---|
| "removing scale vectors substantially degrades pre-training" | **0.028** terminal loss at matched LR, **0.015** after retuning (Fig. 1, 0.12B) |
| "we investigate the role of weight decay" | **Input-Norm** (a linear follows) → weight decay **helps**; **Output-Norm** (none follows) → **harmful**. Called IWD |
| — | overhead **1.04× wall clock**, 1.01× memory (Table 3, 1B) |
| — | three composable variants: branch-specific γ_Q/K/V, dual placement DP/DNP, magnitude-direction OR/ER |

Each became its own claim with its own locator, because they have different scopes,
different risks of bias, and different consequences.

**Why it mattered here, concretely.** The abstract gave a direction and no magnitude, so
the hypothesis could not be checked against our noise floor at all. The full text gave
both the effect size (0.015–0.028) and the cost (1.04× → about 0.0031 bpb at our
measured 0.0792 per e-fold), which turns "probably good" into arithmetic. It also
revealed that **the arm already running was misconfigured**: both our scale vectors sit
on Input-Norms, so IWD says weight decay should be ON, and we had set it to 0.

**Know what full text can and cannot buy.** It moved that belief only 0.705 → 0.714,
because `reproduction` (0.42, weighted more than three times venue) still dominates.
**Literature confidence saturates around 0.71–0.75 no matter how well you read.** Only
running the experiment moves it — a decisive confirmation contributes reliability up to
0.98. Reading tells you *what to run*; it cannot substitute for running it.

### 3.1 Break every paper into claims and mechanisms — the protocol chain

`docs/SCIENTIFIC_METHOD.md` defines the chain, and every arrow is an immutable
identifier. Nothing may skip a link:

```text
ResearchAgenda -> LiteratureSearch -> LiteratureSource
  -> ScientificClaim (one scoped proposition, one belief_key, supports|opposes)
    -> ScientificMechanism (a causal graph; every edge cites its claims)
      -> ScientificHypothesis (prediction + minimum effect + falsifier)
        -> ExperimentSpec -> EvidenceDecision
          -> experiment-origin ScientificClaim   (via `conclude`)
```

```bash
uv run autoresearch --root .autoresearch literature-source runs/science/lit_<x>.json
uv run autoresearch --root .autoresearch scientific-claim  runs/science/claim_<x>.json
uv run autoresearch --root .autoresearch mechanism         runs/science/mech_<x>.json
uv run autoresearch --root .autoresearch hypothesis        runs/science/hyp_<x>.json
uv run autoresearch --root .autoresearch science           # rebuild beliefs/gaps/ideas
```

The extraction rules are enforced by the scoring, so honour them at write time:
**one claim states one scoped proposition**, carries an exact locator and a short
attributable excerpt, and declares its own weakness — study design, artifact
state, reproduction state, directness, scope match, risk of bias. Do not merge
several paper conclusions into one claim. Do not infer a mechanism from a
benchmark win without marking it `indirect` or `speculative`. Do not paste an
abstract as if it were a methods audit. Reproduction weighs more than three times
what venue prestige does; an honest `not_attempted` costs less than a claim that
gets withdrawn.

Mechanism confidence is the **weakest edge**, because a causal chain is no
stronger than its least supported necessary link. If a mechanism is scoring low,
the useful move is to find the weak edge and measure it, not to add prose.

### 3.2 Results refine beliefs — the loop closes both ways

Papers are not a one-way input. **Every landed result updates the belief ledger**,
and the updated ledger is what generates the next claims and mechanisms:

1. `conclude <hypothesis_id> <spec_id>` — materialize a decisive result as an
   experiment-origin claim. The **executed result sets the direction**; a declared
   stance that disagrees is reported as `declared_stance_mismatch` and does not
   get to reverse the evidence. A refuted hypothesis is a claim too, and an
   `opposes` claim against a belief you previously supported is one of the most
   valuable records this campaign can produce.
2. `science` — rebuild beliefs, mechanisms, gaps and ideas, then **read
   `RESEARCH_TASKS.json`**. It is the derived work queue and it already ranks the
   next reading: sparse/stale/contested topics, unread full texts, unresolved
   contradictions, weak mechanism edges, hypotheses ready to stage, hypotheses
   needing refinement.
3. **Re-read against the result.** When a measurement disagrees with a claim, go
   back to the source and ask whether the claim was mis-scoped rather than wrong —
   scope transfer is the most common failure here, and the corrected claim with a
   narrowed `scope` is a better record than a deletion.
4. **A `contested` belief is a measurement, not a defeat.** Independent support
   and opposition both above .45 mass means the discriminating experiment is
   cheap and obvious. Stage it.

Old beliefs get **refined or retired, never quietly kept.** A belief that has
survived three contradicting results is a bug in the ledger, not a strong belief.

### 3.2b Every result gets YOUR mechanistic explanation, and that is where the next
### hypotheses come from

A landed number is not a finding until someone says *why* it happened. The literature
supplies mechanisms for things other people measured; **our results need mechanisms we
author ourselves**, because nobody else has run this frame.

For every landed result — wins, nulls, and especially the surprises:

1. **Write the causal story before reading anyone else's.** What mediator moved?
   `num_steps`, MFU, gradient noise, receptive field, optimizer conditioning, memory
   traffic? Name it, and name the *sign* it should have had.
2. **Check the mediator against its own diagnostics.** Every spec carries
   `num_steps`, `training_seconds`, `mfu_percent`, `total_tokens_M` for exactly this.
   A story that predicts more steps and got none is refuted regardless of `val_bpb`.
   This is the cheapest way to catch a "win" that arrived through the wrong door.
3. **Explain the nulls too, and prefer the common cause.** Several nulls with one
   shared explanation is a stronger finding than one win — a predecessor's real
   result was that its whole family sat inside the noise band, which is a fact about
   the frame, not about the candidates.
4. **Register it.** `mechanism` for the causal graph, with every edge citing claims
   and every assumption written down. If it contradicts a registered belief, say so:
   an `opposes` claim against something we previously supported is among the most
   valuable records this campaign can produce.

**Then use the mechanism to generate hypotheses — that is what mechanisms are for.**
A mechanism is a generator, not a summary. From one causal graph, derive:

- **the confirmatory test** — the cheapest intervention that moves the mediator and
  nothing else;
- **the discriminating test** — the one that separates your explanation from the best
  rival, which matters more than the confirmatory one and is usually skipped;
- **the exploratory extensions** — *if this mediator is real, what else does it
  imply?* Push it somewhere the literature has not been. This is where the 6 in the
  6:4 comes from (§7.1): exploration is not random candidates, it is mechanisms taken
  seriously enough to have consequences.

**Good exploratory hypotheses come from taking your own mediator literally.** If the
story is "removed FLOPs return as steps", ask what *else* is redundant. If it is "the
GPU is gap-bound not kernel-bound", ask what else runs while it waits. If it is
"scale vectors precondition the next linear map", ask which other unparameterised
operation in this net is doing the same job badly. The generative move is always:
*take the mediator seriously, then find the next place it applies.*

Two discipline notes, both learned expensively here:

- **Do not explain noise.** With control σ at 0.00556 against a 0.000426 gate, most
  deltas are ties. Building a mechanism on an unordered tie manufactures a belief out
  of contention, and it will be withdrawn later at cost. Check the band first.
- **An implementation failure is not evidence about the mechanism.** When the CUDA
  graph candidate segfaulted, the gap mediator was untested, not refuted. Record it
  as an implementation null and say which it was.

## 3.3 Every 10 experiments — Fable writes the paper

**On each 10-landed-experiment boundary** (not hourly, not batched at the end),
Fable performs a standing analysis-and-critique pass over the campaign so far and
writes it to `AI_papers/`, using `templates/paper_template.md`. It reads the
immutable records and the derived views — not this session's narration of them —
and reports:

1. what the landed evidence actually supports, with replicate counts and scope IDs;
2. which reported claim is most likely to be withdrawn, and why;
3. what the failures have **in common** — the common cause is the finding, not
   each individual null;
4. what was assumed and never measured, ranked by cheapness to measure;
5. **which beliefs this block changed** — every claim registered, refined,
   contested or concluded, and the mechanism edges that moved with them;
6. the next experiments it would run, as concrete stageable candidates.

Points 5 and 6 are not optional. A critique that produces no next experiment is
commentary, and a block of ten experiments that moved no belief means the reading
and the running have come apart.

Count landed experiments from the immutable store, never from memory:

```bash
uv run autoresearch --root .autoresearch bank | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['controls']+d['candidates'])"
```

## 4. Per-heartbeat order of work

```
1  LIVENESS   the four checks above (also on the 10-minute clock between beats,
              armed as code per 1.1 -- watchdog.py AND wake.py)
2  BANK       uv run autoresearch --root .autoresearch bank && ... status && ... doctor
3  DISPATCH   analyst + critic + scout agents, concurrently, while the GPUs run
4  CLASSIFY   every landed candidate: MECHANISM / KNOB / THROUGHPUT, and whether
              it beat its own frozen same-GPU control or merely a different one.
              Then write YOUR OWN mechanistic explanation of the result, check it
              against its diagnostics, and derive the confirmatory, discriminating
              and exploratory hypotheses it implies (3.2b)
5  READ       >=30 papers/hour concurrent with the GPUs, 2025-2026 biased, content
              from the publisher not the metadata provider; analyse each for its
              MEDIATOR and whether that mediator exists in our frame (3.0), then
              new sources -> claims -> mechanisms (3.1). Never in series with GPU time.
6  REFINE     conclude landed hypotheses, rebuild `science`, work RESEARCH_TASKS.json,
              update or retire the beliefs the results moved (3.2)
7  STAGE      refill the queue to >= 4 jobs under the exploration budget
8  PROMOTE    anything in views/PROMOTION_QUEUE.json -> promote (held-out seeds)
9  RECORD     LOG.md one line per landed run: what was learned, not what was run
10 CHART      refresh the artifact chart after EVERY experiment, never batched
11 PAPER      on every 10th landed experiment: Fable writes to AI_papers/ (3.3)
```

## 5. What may be called a result

The boundary is asymmetric on purpose and the record model enforces it:

| statement | authority |
|---|---|
| "the job finished" | queue state — **not evidence** |
| "the measurement is usable" | EvidenceDecision `valid` |
| "worth spending seeds on" | bank row `promotion_due` — **not a claim** |
| "this is an improvement" | a supported promotion on ≥3 held-out seeds |

A bank delta is a **work-allocation decision**, not a finding. Never report a
seed-42 candidate delta as an improvement, and never report a best-of-N single
run at all. Every number carries its replicate count and its scope ID.

**Winner's curse is quantified here, not hypothetical.** A 4-arm sweep read at its
best arm at n=4 is inflated by ~1.03σ/√4 in expectation; two stacked selections
regressed a measured 0.44 gates when re-measured on disjoint seeds. The bank gate
is calibrated for *one pre-specified comparison*. That is exactly why promotion
runs new, held-out seeds and uses their mean — not the best seed, not the bank
delta.

## 6. Measurement hygiene on this specific box

- Load average on this host has been observed above **280 with 16 users**. A
  contended window can move `val_bpb` by ~0.02, which is *forty times* the
  0.000426 promotion gate. Contention is the dominant noise source, not the model.
- Therefore: a candidate is only ever read against **its own frozen control, on
  its own physical GPU, from within the last 20 minutes**. That is what
  `BANK_TTL_SECONDS = 1200` and `BANK_MAX_USES = 3` buy, at a cost of about one
  control slot in four. Do not raise them to save control runs.
- **Judge on measured `val_bpb`.** No throughput correction, no adjusted bpb, no
  imputed value is ever ranked, promoted, or reported. `num_steps` is reported
  beside every run as the diagnostic that explains where a number came from. If
  you want to know how a config scores at equal steps, that is an experiment to
  run, not arithmetic to do.
- A win that arrived with +3% steps is a **throughput** win. Label it as one.
- `prepare.py` and the data manifest are frozen. They define the metric. Touching
  either is a new scope ID and a new leaderboard, not a new record on this one.

## 7. Exploration is enforced by the harness, not by intention

`autoresearch/exploration.py` refuses, inside the staging lock:

- a non-mechanism candidate when fewer than 5 of the trailing 8 staged candidates
  were mechanisms;
- a `knob` candidate whose family has never cleared the bank gate;
- a third consecutive candidate in one uncleared family.

This exists because a predecessor campaign wrote the same rule down, followed it,
and converged onto knobs anyway. A recurring failure stops recurring when it
becomes structurally impossible, not when it becomes discouraged. Overrides are
available and are recorded in the immutable spec with a written reason — if you
find yourself writing three overrides in a row, the budget is not the problem.

### 7.1 The mix changes with the regime: knobs first, then 6:4 forever

**Exploration versus exploitation is not a fixed ratio, and the early answer is not
the late one.**

**Before saturation, tune the knobs.** On a fresh frame the cheap settings are
genuinely uncalibrated and the gradient is steep — the v2b→v2c jump came from a
throughput lever nobody had pulled, worth **0.1035 bpb**, roughly a hundred times
the 0.000426 gate. While a knob still moves the metric by multiples of the noise
floor, turning it is the highest-value thing available and calling that
"unambitious" is a mistake. Exploit hard here.

**After saturation, hold exploration:exploitation at 6:4.** A family is saturated
when its last two candidates land inside the control band — an unordered tie, not a
small win. From then on, of every ten staged candidates roughly **six explore a
mechanism the campaign has not tested** and **four exploit a direction that has
already flashed a signal above the noise band**. Not 10:0: a direction that has
shown real signal deserves follow-up, and abandoning it to chase novelty is how a
campaign ends with many interesting nulls and no record. Not 3:7 either — that is
the drift the exploration budget exists to prevent, and every predecessor found it
by accident.

Count the ratio over the trailing ten *staged* candidates, not over the ones that
happened to land or happened to win — counting outcomes lets a lucky exploit
justify the next four.

**A signal, for the 4:** a delta beyond ±2σ of that GPU's control replicates, from a
candidate read against its own frozen same-GPU control. A bank delta inside the band
is not a direction; it is a tie, and exploiting it is how the winner's curse enters.

## 8. End-of-block checklist

- `queue` has no unexpected `running`, `waiting`, or `blocked` jobs.
- `doctor` shows fresh controls and healthy profiles.
- `bank` has no unexplained unscored candidates.
- `validate` returns `valid: true`.
- Every reported best value names its scope ID, its promotion spec, its result
  IDs, and its distinct verified seeds.
- The artifact chart is current.

## 9. The chart

`tools/make_chart.py` reads the immutable store directly — no hand-maintained TSV,
no numbers typed by an agent — and writes `runs/val_bpb_progress.html`:

```bash
uv run autoresearch --root .autoresearch bank    # refresh the scored view first
uv run python tools/make_chart.py
```

Regenerate and republish it **after every landed experiment, never batched**. It is
the fastest way to see the two things that matter at a glance: whether the frontier
is moving, and whether the spread of the control replicates has swallowed it.

**Regeneration is automatic; republishing is yours.** `tools/watchdog.py` runs both
commands on every 10-minute tick (`--no-chart` opts out) and logs `chart=ok (N
landed runs, M valid, ...)`, so the local HTML is never stale. What the watchdog
cannot do is push it to the artifact — do that every heartbeat, and always by
**updating the existing artifact URL** rather than creating a second one, or the
campaign ends up with a trail of half-current charts and no canonical link.

**One chart per scope ID, and never one chart across two.** The v2a artifact holds
the pre-reset `..._v2a` campaign (10 runs, best 1.1689 at ~192 steps / 7.6% MFU);
`..._v2b` is a *different leaderboard* and gets its own artifact. Overwriting one
with the other destroys a record and silently invites the cross-scope comparison
the rules forbid. Keep the scope ID visible in the chart subtitle so the link can
never be misread. `make_chart.py` enforces this rather than trusting care: it reads
the scope ID from `runs/scope.json` and plots only runs whose spec carries that
scope, reporting the rest as `N from earlier scopes excluded`.

**Experiment #1 is the first valid run of the current scope** (operator decision,
2026-08-16). Numbering happens *after* the scope filter and the validity filter, so
the count starts at the corrected baseline rather than at the runs which only
established that the harness was broken. A calibration wave that proved a stale
binding or a sleeping laptop is not experiment 1; the first trustworthy measurement
is. When a throughput or evaluator fix forces a new scope ID the counter restarts
at 1 by construction — that is the intended behaviour, not something to work around.

What the marks mean, and why they are drawn this way:

- a grey **square** is a bank control; a **circle** is a candidate;
- a dashed **ring** joined by a line to a candidate is that candidate's *exact
  frozen control on the same physical GPU* — pairing is read from the immutable
  spec reference, never inferred from adjacency;
- the grey band is **±2σ of the control replicates**. Anything inside it is an
  unordered tie, and the band is drawn behind the data so a delta can never be read
  without its noise floor;
- **only `valid` runs are plotted** (operator decision, 2026-08-16). An invalid run
  is a harness or host failure — a stale binding, a laptop that slept — not a score,
  and putting it on a bpb axis makes the frontier harder to read for no information
  gained. It is *excluded, not deleted*: the count appears in the subtitle and in the
  `Plotted runs` tile, and every one stays in the immutable store. `--show-invalid`
  puts them back as ✕ marks, which is the right flag to reach for when diagnosing a
  run of failures rather than reading the frontier. What must never happen is a
  *valid* replicate disappearing because it read badly — that is selection, and the
  bank gate is calibrated assuming it does not occur;
- green means `promotion-due`, which is a **work-allocation decision, not a result**.
  The page says so in its own subtitle because a chart is where that distinction
  gets lost first.

Colour is never the only encoding: status is carried by shape (square / circle / ✕)
and repeated as a text pill in the table, and the palette is validated for
colour-vision deficiency in both light and dark. Do not swap in a red/green pair
without re-validating — the obvious one fails deutan separation.
