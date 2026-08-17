# More steps can make the model worse

### What five experiments on a shared H200 taught us about a benchmark that pays in wall-clock time

**OPHIS campaign report · 17 August 2026**
**Scope:** `ophis_v3b_karpathy228791f_model_correctedbpb_pinned10_tokencache_h200_train300s`
**Status of every number here:** exploratory. Nothing has been promoted. There is no SOTA.

---

## The short version

We are trying to improve a small language model — 50 million parameters — under a hard
budget of **300 seconds of training time**. Not 300 steps. Not a fixed number of
tokens. Three hundred seconds on the clock.

That framing seems innocuous and it is not. It means the benchmark rewards *speed*: if
you make each step cheaper, you get more steps, and more steps should mean a better
model. Every optimisation we tried was aimed at that.

Then the machine went quiet, our runs suddenly got 60% faster — and the model got
**dramatically worse**. Same code. Same random seed. Same data. Just more steps.

That single observation reframed the whole campaign, and it is the reason this report
exists.

---

## Part 1: The baseline, and why it took so long to trust

Before any of this, we had to establish what "unchanged" even means. That was most of
the work, and almost none of it was interesting.

The short history: we found our baseline was **mislabelled** (described as reproducing
a well-known reference benchmark when it had three changes that broke comparability),
our target number was **unsourced** (a figure that appears nowhere except our own
notes), and our "pristine" reference measurement was **contaminated** — the code was
byte-identical to upstream, verified by hash, but it read a shared cache that another
project had left 41 data shards and a modified byte-accounting table in. Pristine code
does not imply a pristine measurement.

What survived is a baseline whose provenance is cryptographic rather than asserted, and
a habit that has paid for itself repeatedly: **check the boring explanation first.**

### The baseline behaves beautifully — in one regime

Twenty-one runs of the identical baseline, between 579 and 739 steps:

| steps | val_bpb |
|------:|--------:|
| 579 | 1.033380 |
| 603 | 1.029384 |
| 629 | 1.026257 |
| 656 | 1.022590 |
| 704 | 1.017555 |
| 739 | **1.014023** |

Fit that and you get a clean law:

```
val_bpb  =  1.5299  −  0.0782 · ln(steps)
```

The residual scatter around it is **σ = 0.000262**. For context, the threshold this
campaign uses to call something a real improvement is 0.000426. So once you know how
many steps a run completed, you can predict its score to *better than the size of an
improvement worth reporting*.

This was a genuine breakthrough for us, because the raw scatter is σ = 0.005427 — about
twenty times larger. The spread isn't randomness in the model; it's the shared machine.
When neighbouring jobs steal CPU, our runs get fewer steps and score worse. Correct for
steps and the noise nearly vanishes.

We refit it four times as data arrived — at 6, 15, 18 and 21 runs. The slope moved from
−0.0792 to −0.0782. It held.

---

## Part 2: The result we did not expect

Then the machine emptied out. Host load fell from ~350 to ~113, and four baseline runs
completed **984 to 1010 steps at 41–42% GPU utilisation** — far more than anything we'd
seen.

The law says more steps means a lower score. Extrapolated to 1000 steps it predicts
**0.990**.

The four runs scored a mean of **1.084694**.

```
        predicted at 1000 steps    0.990
        actually observed          1.085          ← worse by 0.095
        the improvement threshold  0.000426       ← 220× smaller than the miss
```

The two regimes, side by side:

| | runs | steps | mean val_bpb |
|---|---:|---:|---:|
| contended box | 21 | 579–739 | **1.024932** |
| quiet box | 4 | 984–1010 | **1.084694** |

**A faster machine produced a worse model.** Reliably, across four runs, with a
between-run scatter of 0.0097 — an order of magnitude smaller than the effect.

### Ruling out the boring explanations

*Was it a data boundary?* No. At 984 steps the run has consumed 0.82 of one pass
through the corpus. Nothing repeats, nothing runs out.

*Did our code change?* This was the explanation we most expected, because we edit the
training script constantly to test ideas and restore it from a copy each time. If a
restore were imperfect, "the baseline got worse" would be a bug wearing a discovery's
clothes. We checked every one of the 25 baseline runs against the cryptographic
fingerprint sealed into its own launch record. **All 25 are identical.** The finding
survived its most likely refutation.

*Was it the momentum schedule?* The momentum ramp is driven by step count rather than
time, which looked promising — but it saturates at step 300, so it is identical at 600
steps and 1000. Not the cause.

### What we think is happening

The learning rate follows a schedule parameterised by **elapsed time**, not by step.
Since steps arrive at a fairly steady rate, that turns out not to change the *shape* of
the schedule — a run doing 1000 steps and a run doing 600 both trace the same curve
from start to finish. Our first explanation got this wrong, and the correction matters.

What does change is how many updates you take along that curve. Total distance travelled
through parameter space grows roughly as (number of steps) × (average learning rate).
Do 67% more steps at the same rate and you travel 67% further.

And nothing in the code adapts the rate to the number of steps. There is an interesting
asymmetry here: the optimiser's Adam-family parameter groups **are** rescaled — by
`1/sqrt(model_width/768)`. The learning rate for the matrix parameters, which is where
most of the model lives, is a bare constant.

So the story is: a quiet machine buys extra steps, extra steps buy extra movement, and
past roughly 750 steps the run **overshoots**.

### Why we are not calling this settled

One flaw dominates, and it is structural rather than fixable by more of the same data:
**the high-step runs happened because the machine went quiet.** Step count and machine
contention moved together perfectly. Everything above is equally consistent with "a
quiet machine harms us for some other reason" — cache behaviour, clock states,
neighbouring workloads. Four runs cannot separate two variables that always move
together.

The claim is recorded with `risk_of_bias: high` for exactly this reason, and with the
two experiments that would separate them (below).

---

## Part 3: What the five experiments actually showed

Experiment 1 *is* the baseline. Only real ideas advance the counter.

### Experiments 2 and 3 — adding a learnable scale to normalisation. Refuted.

A 2026 paper reports that the learnable scale vector in normalisation layers, though a
negligible fraction of parameters, substantially helps training — and that removing it
hurts. Our baseline has **no such scale at all**. That looked like free money: a
component the literature says matters, simply absent.

Reading the paper's full text (not just its abstract) gave two things the abstract
didn't: a magnitude (0.015–0.028 in loss on a similar-scale model) and a taxonomy —
weight decay *helps* these vectors when a linear layer immediately follows, and *hurts*
when one doesn't. That told us our first attempt was misconfigured before it finished
running, so we ran both variants.

| | val_bpb | steps | vs matched baseline |
|---|---:|---:|---:|
| exp2 · scale vectors, no weight decay | 1.175021 | 707 | **+0.157466** |
| exp3 · scale vectors, with weight decay | 1.181510 | 655 | **+0.158764** |

Both far worse, and — the informative part — **worse by almost exactly the same amount**,
despite different weight decay and 52 steps of difference. A penalty indifferent to how
the scale vectors are trained is not a story about training them.

That killed one hypothesis (weight decay was the problem) and pointed at another: our
optimiser for matrix parameters orthogonalises its updates, deliberately discarding
per-channel scale information. A learnable per-channel input scale reintroduces exactly
what the optimiser just removed. The literature result was established with optimisers
that don't do this.

We also ruled out the dumbest explanation — that mixing precisions changed the numerics
— with a ten-second tensor probe rather than another 300-second run. Values were
bit-identical.

### Experiments 4 and 5 — spending less on attention. Promising, and confounded.

Both reduce attention work, on the theory that the model's global-context layers already
carry information the local layers are re-deriving. Under a time budget, saved compute
should come back as steps.

It did — both ran far more steps than any baseline had:

| | val_bpb | steps | MFU | vs nearest-step baseline |
|---|---:|---:|---:|---:|
| exp4 · more local layers per global layer | 1.038882 | 985 | 41.1% | **−0.046457** |
| exp5 · halve the local attention span again | 1.015036 | 1040 | 40.0% | **−0.082203** |

Read naively, those are enormous. They are not that, and the reason is Part 2.

Both landed **inside the broken regime** — above 750 steps, where the baseline itself
degrades. So a large share of that apparent advantage may be "less damaged by overshoot"
rather than "better model". They should be compared to a baseline that isn't broken at
1000 steps, and we don't have one yet.

The sober comparison: exp5's **1.015036 at 1040 steps** is indistinguishable from the
baseline's own best, **1.014023 at 739 steps**. **No experiment has beaten the best
baseline run.**

### Experiment 6 — testing the mechanism directly. Running now.

If extra steps hurt because extra steps mean extra movement, then damping the learning
rate in proportion should remove the penalty. Experiment 6 estimates how many steps the
run will finish (steps so far ÷ fraction of time elapsed) and scales the rate by
`sqrt(650 / estimated total)`.

It is clamped so it can only ever *reduce* the rate. Two reasons: the low-step regime is
already well behaved and raising its rate is untested speculation, and the clamp makes
the arm a **no-op below 750 steps** — which is a free correctness check. If it changes
anything down there, the implementation is wrong and we'll know immediately.

Deliberately staged as an experiment, **not** as a fix to the baseline. Folding it in
would make all 25 baseline runs incomparable, which is precisely the mistake that
produced the mislabelling mess described in Part 1.

---

## Part 4: The failures, which are their own finding

Thirty-four runs have landed. Five were rejected as invalid — and **every single one was
the environment, not a wrong idea**:

| cause | count | what it means |
|---|---:|---|
| SSH connection refused | 3 | our own tools opened too many connections at once |
| a neighbour appeared on our GPU mid-run | 1 | the co-tenancy guard doing its job |
| run was 11% slower than its own history | 1 | the throughput guard doing its job |

Not one run was invalidated because a hypothesis was wrong. Every failure was
infrastructure — and two of those rejections are *guards working correctly*, keeping
contaminated measurements out of the baseline. That exclusion is part of why the
step-law fit stays clean while the raw scatter drifts with the machine.

The SSH one is worth confessing: the failures were reported as *"code or data did not
match the sealed manifest"*, which is the one error class that should stop a campaign
dead. It wasn't that. The connection had been refused, so the verifier couldn't read
any file, and the system reported every file as mismatched. An environmental fault
wearing a scientific fault's clothes. We'd caused it ourselves by running four
monitoring tools that each opened their own connection every ten minutes.

---

## Part 5: What to do next, in order

**1. Separate steps from quietness.** The single most valuable experiment available, and
it needs no new ideas. Throttle a quiet machine so it completes ~650 steps *without* any
learning-rate change. If the penalty disappears, extra steps were the cause. If it
persists, the quiet machine was, and Part 2 needs rewriting.

**2. Finish experiment 6** and read it against a high-step baseline. If damping removes
the penalty, the mechanism is real and the benchmark has a genuine trap in it: *any*
speed improvement silently degrades quality unless the rate is adapted.

**3. Re-run experiments 4 and 5 in the fixed regime.** Their apparent gains are
uninterpretable until the baseline is well-behaved at 1000 steps. If they survive with
a corrected schedule, they are real; if they shrink to nothing, they were measuring
overshoot resistance.

**4. Then, and only then, tune constants.** We have a literature-predicted value for the
learning-rate warmdown fraction (33%, from a 720-run study covering our exact model
size). It's a one-line change with a specific predicted number, which makes it a real
test rather than a search. It has been correctly refused twice by the exploration
budget on the grounds that its family hasn't yet cleared the threshold — and both times
the refusal was right.

**5. Promote something.** Nothing in this campaign has ever been confirmed on held-out
seeds. Until one result survives that, the leaderboard is a list of work allocations,
not findings.

---

## What we would tell someone starting this

**The measurement was harder than the modelling.** Thirty-four runs in, we have tested
five ideas and rebuilt the measuring apparatus perhaps a dozen times. That ratio looked
like failure until the moment the apparatus caught something real — the step-count law
is what made the >750-step anomaly visible at all. Without it, four unusually fast runs
would have looked like ordinary noise on a shared machine.

**Check the boring explanation first, and check your own code before the world's.** The
baseline-drift check took two minutes and would have saved us from publishing a bug as
a discovery. The dtype check took ten seconds and closed off a whole branch of
speculation.

**A time budget is not a step budget, and the difference bites.** Every result here
depends on which of those you think you're measuring. It's the kind of assumption
nobody writes down because it seems too obvious to state.

---

### Reproducing any number above

```bash
uv run python tools/preflight.py                  # 20+ checks on code, scope, machine
uv run autoresearch --root .autoresearch bank     # scored baseline and candidates
uv run autoresearch --root .autoresearch science  # claims, confidence, open questions
uv run python tools/make_chart.py                 # regenerate the chart from the store
```

Every run cited is an immutable record with a cryptographic fingerprint and its own
verdict. The two withdrawn numbers from Part 1 are marked `WITHDRAWN` with reasons
rather than deleted. Live chart: the campaign's published artifact, where experiment 1
is the baseline and every later number is one idea tested.
