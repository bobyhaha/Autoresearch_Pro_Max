# What 34 runs bought us, and what the next 20 should buy

**OPHIS campaign report #1 · 2026-08-17**
**Scope under study:** `ophis_v3b_karpathy228791f_model_correctedbpb_pinned10_tokencache_h200_train300s`
**Runs covered:** 34 landed (23 valid, 11 invalid) across four scope IDs
**Status of every number below:** exploratory. Nothing here is SOTA; nothing has passed held-out-seed confirmation.

---

## The one-paragraph version

We spent 34 runs and found almost nothing about language models. We found a great deal
about *measurement*: that our baseline was mislabelled, that our reference number was
unsourced, that our "pristine" control was contaminated, that a corrupted literature
feed would have let us publish fabricated citations, and that three quarters of our
failures were the measuring apparatus rather than the thing measured. The single
genuine modelling result is that the input pipeline — not the GPU, not the model — was
consuming 86% of a 4.67× throughput deficit. The most useful *scientific* result
arrived in the last hour: **step count explains ~95% of our control variance**, which
turns an unmeasurable regime into a measurable one. That is what the next 20 runs
should exploit.

---

## 1. Where the runs went

| scope | runs | valid | mean val_bpb | mean steps | what it was |
|---|---|---|---|---|---|
| `..._corrected_bpb_..._v2b` | 8 | 4 | 1.111699 | 277 | prefetch-thread dataloader |
| `..._corrected_bpb_..._v2c` | 14 | 10 | 1.025314 | 644 | pre-tokenized token cache |
| `..._pinned10_..._v3b` | 3 | 3 | 1.024381 | 643 | post-restart, pre-relabel |
| `ophis_v3b_...train300s` | 9 | 6 | 1.028486 | 612 | current, after honest relabel |

**These four columns are not comparable to each other.** Different `prepare.py` digests
mean different evaluators, and `prepare.py`'s digest *is* the scope's evaluator. The
v2b→v2c drop of 0.086 bpb is real but it is a throughput effect inside one scope, not
progress against a fixed target. The last two rows are the *same code* under two
different labels — an artefact of correcting the label mid-campaign, which fragmented
the bank and is called out as a lesson below.

**11 of 34 runs were invalid, and every single one was environmental:**

| cause | count | resolution |
|---|---|---|
| stale `baseline_provenance.json` on the box (mode 444 blocked scp) | 2 | synced with chmod dance |
| macOS idle sleep breaking timing reconciliation | 2 | `caffeinate` held |
| foreign tenant grabbed a GPU mid-launch | 1 | co-tenancy guard working as designed |
| SSH refusal reported as "all bindings mismatched" | 3 | connection multiplexing |
| CUDA graph capture segfault | 1 | implementation null, mechanism untested |
| host environment could not be established | 2 | transient |

**Zero were scientific faults.** No run was invalidated because a hypothesis was wrong.
That ratio is the headline finding of this campaign so far, and it is not flattering.

---

## 2. The one real modelling result

The baseline was 4.67× short of the step count we expected. We decomposed it by
measurement rather than argument:

```
tokenizer starvation   1.78x   (1106ms steady-state step ÷ 620ms when fed from buffer)
remaining when fed     2.08x   ( 620ms fed              ÷ 298ms reference)
                     -------
                       3.71x   vs observed 3.70x
```

The tell was a **bimodal step time on all six runs**: the 64-batch prefetch queue fills
for free during `torch.compile`, so steps 1–20 ran at 526–684 ms and every later step
sat flat at 959–1157 ms. The producer was not slow because BPE is slow —
`encode_ordinary_batch` already runs 8 threads — it was slow because every run re-did
identical work: parquet read, `.to_pylist()` into millions of Python strings,
per-document Python lists.

Pre-tokenizing to a flat memory-mapped cache took **215 → 809 steps and 8.66% → 33.7%
MFU**. An nsys profile then showed the GPU still idle ~43% of wall with ~572 kernel
launches per step, so the remaining limit is dispatch, not arithmetic.

**A correction we owe the record:** we described the token cache as "a throughput change
only" because the token stream is byte-identical. That reasoning was wrong. Under a
300-charged-*second* budget, a faster pipeline buys ~3.76× more optimizer steps and
therefore a better model. Byte-identity shows the *data* is unchanged; it does not show
the *result* is comparable to anything upstream.

---

## 3. The result that changes what we can do next

Fitting all six current-scope controls against their own step counts:

```
val_bpb = 1.5364 − 0.0792·ln(steps)

raw σ across controls      0.004430
residual σ after the fit   0.000206      (21.5× reduction)
promotion gate             0.000426
```

**Step count explains ~95% of the control spread, and the residual falls below the
gate.** Two waves make the mechanism visible: wave 1 spanned 613–698 steps and had
σ = 0.005562; wave 2 spanned 603–611 steps and had σ = 0.000596. The spread *is*
contention, and contention acts almost entirely through step count.

If this holds, candidates become readable at small *n* instead of requiring many
replicates — which is currently the binding constraint on the entire campaign.

**Three caveats, none of them small.** n=6 against a two-parameter fit leaves 4 residual
degrees of freedom. The slope leans heavily on one leverage point, the 698-step run.
And it is valid **only for model-preserving changes** — applying it to a throughput
candidate would subtract exactly the effect being measured. The implied 0.0792 bpb per
e-fold sits just under the 0.09–0.13 band quoted in the literature, which is at least
consistent.

---

## 4. Things we believed that were not true

Recorded in full because these cost more than the experiments did.

**"Reproduce val_bpb 0.9912 at 1006 steps."** Used as the campaign's target and as the
anchor of several throughput arguments. It appears in exactly one place in this
repository — our own handover note — and is backed by no upstream log. **Withdrawn.**
The pinned program shows 0.997900 / 953 steps / 39.8% MFU, and the benchmark's author
states results are platform-specific and not comparable across hardware. There is no
upstream target for this box.

**"We measured the pristine baseline at 1.140787 ± 0.000614."** The *code* was pristine,
verified by sha256 and git blob hash against commit `228791f`. The *measurement* was
not. Upstream `prepare.py` reads a shared cache and never version-checks
`token_bytes.pt`, so it silently consumed 41 training shards and a corrected byte table
left by an unrelated campaign in July. **Withdrawn.** Pristine code does not imply
pristine measurement.

**"This is the Karpathy baseline."** It is not. We modify `prepare.py`, which the pinned
protocol declares immutable — *"only train.py is the normal mutable path"* — and which
defines the ground-truth loader and evaluator. We also changed the BPB denominator. The
accurate label is *OPHIS v3b using Karpathy's model/optimizer at 228791f, with an OPHIS
protocol adapter, corrected BPB, pinned corpus, and accelerated token pipeline.*

**"71 papers read."** 71 were *retrieved*. Roughly 13 were on-topic; **zero full texts
have been read.** The metadata provider returned correct titles, authors, years and
DOIs attached to *other papers' abstracts* — Kaplan's scaling-laws paper came back
describing "a transport-validity theory for agentic AI interventions". Extracting
claims from that would have produced fabricated statements wearing real DOIs. Content
now comes from the publisher.

---

## 5. What the failures have in common

Not one of the eleven invalid runs was a wrong hypothesis. Every one was the
apparatus: a file that did not sync because of its permission bits, a laptop that slept,
a shared GPU, an SSH limit, a compile mode that segfaults on this stack.

The common cause is that **this campaign's measurements depend on a long chain of
things that fail silently** — remote file digests, wall-clock reconciliation across two
machines, a shared corpus directory, a shared GPU, a shared SSH daemon. Each link
failed at least once, and each failure *looked like* something else: a stale provenance
file looked like a broken campaign, a sleeping laptop looked like a timing bug in the
harness, an SSH refusal looked like a code mismatch.

The response has been to convert each one into a check that fails loudly:
`tools/preflight.py` (21 checks), `tools/wake.py` (agent briefing, exit 3 = decision
waiting), `tools/gpu_guard.py` (stages controls when the fleet idles, never candidates),
and a token-stream equivalence verifier. That is where most of the 34 runs actually
went, and it is why the next 20 should be cheaper.

---

## 6. The next 20 runs

Ordered. Each line is one stageable candidate with a falsifier.

### Phase A — make the measurement usable (runs 1–6)

The step-law result is the highest-leverage thing we have, and it rests on n=6.

1. **Runs 1–6: six more baseline controls.** Not glamorous. They halve the uncertainty
   on the step-law slope, add leverage away from the single 698-step point, and test
   whether residual σ stays below the gate. `gpu_guard` produces these automatically
   when the fleet would otherwise idle. **Falsifier:** residual σ rises above 0.000426,
   in which case the step law is not a usable control variate and every later run needs
   n≥3.

### Phase B — mechanisms with a literature-grounded story (runs 7–14)

Staged as mechanisms, not knobs, because the exploration budget correctly refuses knob
tuning inside a noise band 10× the gate.

2. **Runs 7–8: learnable RMSNorm scale vectors** *(already running)*. The baseline
   normalises the residual stream with a bare `F.rms_norm` and no learnable scale, so it
   sits on the degraded side of the ablation in arXiv 2605.26895 by construction. That
   work reports the scale vector adds no expressivity in Pre-Norm nets but improves
   optimization as a self-amplifying preconditioner on the following linear map.
   **Falsifier:** no gate-clearing delta after adjusting for `num_steps`; or `num_steps`
   falls enough that the added per-step cost cancels the gain, which prices the
   mechanism rather than refuting it.

3. **Runs 9–10: fused RMSNorm + adjacent elementwise.** The nsys profile puts RMSNorm at
   18.6% of kernel time over 10,653 launches and other elementwise Triton at 8.1% over
   32,071 launches — 26.7% of kernel time in memory-bound work. **Falsifier:** a
   post-intervention profile shows the RMSNorm share unchanged, which refutes the
   memory-traffic mediator regardless of what val_bpb did.

4. **Runs 11–12: reduce kernel launches per step.** 572 launches per optimizer step,
   with 32,071 elementwise launches accounting for only 8.1% of kernel time, is the
   signature of a dispatch-bound loop. CUDA-graph capture segfaulted on this stack — an
   *implementation* null, so the gap mediator remains untested rather than refuted.
   Retry via a different route. **Falsifier:** the GPU-idle fraction is unchanged in a
   post-run profile.

5. **Runs 13–14: the discriminating test.** Runs 9–12 attack two different mediators —
   kernel time vs launch gaps — and both predict "more steps". Design one arm whose
   outcome differs between them. This is the run most likely to be skipped and the one
   most likely to be worth it.

### Phase C — knobs, once a family has cleared the gate (runs 15–18)

6. **Runs 15–16: warmdown fraction 0.5 → 0.33.** A 720-run factorial grid over 15M–100M
   decoders (arXiv 2605.25966) puts the loss-optimal warmdown at 33% at every
   (bit-width, size) cell, robust across optimizer, schedule shape and 9× training
   length. We are 50.33M and use 0.5. Currently **blocked** by the exploration budget,
   correctly. **Falsifier:** no gate-clearing delta — which would also be evidence that
   iteration-horizon schedule results do not transfer to a wall-clock horizon, itself
   worth knowing.

7. **Runs 17–18: whichever Phase B mechanism cleared the gate, tuned once.** Exploit,
   per the 6:4 rule, only on a direction that showed signal beyond ±2σ.

### Phase D — confirm something (runs 19–20)

8. **Runs 19–20: held-out-seed confirmation** of the single best candidate on seeds
   43–47. Nothing in this campaign has ever been promoted. Until one thing survives
   disjoint seeds, the leaderboard is a list of work-allocation decisions.

### What would make me abandon this plan

If Phase A shows the step-law residual is not below the gate, Phase B and C are
premature — every candidate would need n≥3 and only ~6 candidates fit in 20 runs. In
that case the right plan is: fix the variance first, by running when the box is quiet
or by pairing arms within a wave, and treat the entire 20 as a variance-reduction study.
Saying that in advance is cheaper than discovering it at run 14.

---

## 7. Open questions we have not measured

Ranked by cheapness.

1. **Does the step law hold across model changes, or only across contention?** Cheap:
   already have the data structure for it, needs the Phase B arms to land.
2. **What is the remaining 43% GPU idle?** One nsys capture on a quiet window separates
   dispatch overhead from contention.
3. **Is the consumer-side best-fit packing loop now the bottleneck?** It scans a
   1000-document buffer per placed document, in the training thread. Never profiled.
4. **Does `resid_lambdas`/`x0_lambdas` already supply the preconditioning that scale
   vectors would add?** If so runs 7–8 are redundant, and the model already has the
   mechanism in a coarser form.
5. **Would a quiet-window run reach the reference step count?** Would separate "shared
   box" from "our software" once and for all.

---

## Appendix: how to check every number in this report

```bash
uv run python tools/preflight.py                 # 21 checks: code, scope, remote, corpus, GPUs
uv run autoresearch --root .autoresearch bank    # scored controls and candidates
uv run autoresearch --root .autoresearch doctor --bank-id karpathy_228791f
uv run autoresearch --root .autoresearch science # beliefs, confidence, research tasks
uv run python tools/make_chart.py                # regenerate the chart from the store
```

Every run cited is an immutable `result_bundle` with an `evidence_decision`; archived
scopes live under `../ophis_v2_archive/`. The two withdrawn numbers are marked
`WITHDRAWN` in `CLAUDE.md` §0.5 with the reason, not deleted.
