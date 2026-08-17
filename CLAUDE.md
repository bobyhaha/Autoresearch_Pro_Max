# CLAUDE.md — read this first, every session

You are the research director for an autonomous ML research campaign. This file is
the handover. Read it completely before running anything.

**Order of reading:** this file → `HEARTBEAT.md` (the operating loop) →
`docs/LESSONS_AND_WORKFLOW_DECISION.md` (why the system is shaped this way) →
`program.md` (what the record model permits).

---

## 0. The one thing that must be true at all times

**Every 10 minutes, check whether you are running.** Two separate questions:
is the worker pool alive, and are the GPUs executing our work?

This is automated. Do not rely on remembering it:

```bash
cd ~/Desktop/OPHIS/simplify_autoresearch_v2
pgrep -f 'tools/watchdog.py' || nohup uv run python tools/watchdog.py \
  --interval-seconds 600 --workers 1 --idle-timeout-seconds 14400 \
  > runs/watchdog.out 2>&1 &
```

`tools/watchdog.py` checks every 10 minutes, **auto-restarts a dead pool** (safe:
the queue is durable and jobs are idempotent), and escalates the cases a monitor
must not decide on its own — empty queue, paused circuit, unreachable host. It
writes `runs/watchdog.log` (one line per tick) and `runs/WATCHDOG_STATUS.json`.

**Start it as your first action of the session, and read `runs/watchdog.log`
before believing anything about what is or isn't running.** A predecessor campaign
spent 12.8 of 21.6 hours idle and used ~5% of the compute it had; earlier in this
campaign the pool drained and exited mid-analysis and four H200s sat at 0%.

Adding a candidate to the queue is a *scientific* decision — the watchdog will
never do it for you. If it reports `IDLE_GPUS_EMPTY_QUEUE`, that is your job.

## 0.5 What the active baseline actually is — read before quoting any number

**The active baseline is Karpathy-derived. It is NOT Karpathy's benchmark baseline.**

The accurate label is: *OPHIS v3b baseline using Karpathy's model/optimizer at
`228791f`, with an OPHIS protocol adapter, corrected BPB, pinned corpus, and
accelerated token pipeline.* Use that phrasing; "the Karpathy baseline" is wrong.

The executables are not pristine. Upstream `train.py` hashes `2954175f…`, active
hashes `0c41f8da…`; upstream `prepare.py` hashes `4f2ba9cb…`, active `2ef2c56d…`.

Three departures, each of which independently breaks comparability upstream:

1. **We modify `prepare.py`, which the pinned protocol declares immutable.**
   `program.md` §1: *"Keep `prepare.py`, `pyproject.toml`, `uv.lock`, and baseline
   provenance sealed and stable; only `train.py` is the normal mutable path."*
   `prepare.py` defines the ground-truth data loader and evaluator. Changing it is
   a protocol violation, recorded deliberately rather than quietly.

2. **The token cache is not neutral, and earlier notes here were wrong to say so.**
   The token stream is byte-identical — that was verified and it is true. But the
   budget is **300 charged training seconds**, so a faster input pipeline buys
   ~3.76× more optimizer steps and therefore a better model. Under a wall-clock
   budget, input-pipeline speed *is part of the benchmark*. Byte-identity shows the
   **data** is unchanged; it does not show the **result** is comparable.

3. **The metric denominator differs.** Upstream computes byte lengths via
   `decode([id]).encode("utf-8")`; OPHIS uses `decode_single_token_bytes()`. These
   are **different metrics**, not a better and worse estimate of one metric.

### Two numbers previously quoted here are withdrawn

**`val_bpb` 0.9912 at 1006 steps / 42% MFU — WITHDRAWN, unsourced.** No upstream log
in this repository supports it. It was used as a reproduction target and as the
denominator of several throughput arguments; none of those inferences carry. The
official pinned program instead shows a baseline example of **0.997900, 953 steps,
39.8% MFU** — and Karpathy states results are **platform-specific and not comparable
across hardware**, so no upstream figure is a target for this box at all.

**The local "pristine" reference 1.140787 ± 0.000614 — WITHDRAWN, contaminated.**
The code was pristine; the measurement was not. Upstream `prepare.py` reads the
shared `~/.cache/autoresearch`, which held **41 training shards** and a **corrected
`token_bytes.pt`** left by another campaign, and upstream never version-checks that
file. So it measured someone else's corpus with someone else's byte accounting. It
is neither Karpathy's clean default nor comparable with v3b.

A genuinely pristine reference requires the code in `karpathy_pristine/` run against
a **freshly prepared, uncontaminated** `~/.cache/autoresearch`. Until that exists,
this repository has **no** upstream reference measurement, and v3b numbers stand on
their own or not at all.

## 1. Where things stand

The campaign was reset to zero on 2026-08-16 after a deep throughput fix. **There
are no results.** The first thing that will land is a fresh baseline calibration.

~~The immediate objective the operator set: reproduce Karpathy's reported `val_bpb`
≈ 0.9912 on the pristine baseline.~~ **WITHDRAWN — see §0.5.** That figure is not
backed by any upstream log in this repository, and Karpathy states results are
platform-specific and not comparable across hardware. There is no upstream target
for this box. The objective is to improve v3b against its own frozen controls.

### What went wrong before the reset, and what was fixed

Four valid controls scored `val_bpb` ≈ 1.169 at **192 steps and 7.6% MFU**.
An earlier note here claimed Karpathy's own log recorded **1006 steps, 42% MFU,
val_bpb 0.9912** for this commit. **That claim is withdrawn: no such log is
vendored here.** The comparison below is retained only to show how the reasoning
ran, and its upstream anchor should be treated as unverified. Nothing was broken in the model — the
number sat exactly on Karpathy's own token law (~0.09–0.13 bpb per e-fold of
tokens), 5.2× short on throughput.

The cause: upstream's `make_dataloader` runs the BPE encoder **inline in the
training loop**, single-threaded, with no prefetch. On a quiet box the encoder
keeps ahead of the GPU. On this host — load average ~350, 20 tenants — it starves
the GPU. The decisive control was already on the box: `plain_v2`, *same host, same
hour*, reached **1455 steps at 39.6% MFU**. The hardware was never the limit.

The fix (`runs/code/prepare.py`): tokenization moved to a bounded-FIFO producer
thread so it overlaps with GPU compute. **Verified byte-identical**: 12 batches of
int64 inputs+targets hash to
`f4ab4f330294964c6a070fb6f400e06758822a4fa43f35ab672324e8546550ef` both before and
after, with identical epoch boundaries. `evaluate_bpb` is untouched. This is a
throughput change and nothing else.

Because `prepare.py`'s digest is the scope's `evaluator`, the scope ID moved to
`karpathy_228791f_corrected_bpb_h200_train300s_v2b`.

### Update 2026-08-16: v2b measured, diagnosed, and superseded by v2c

The prefetch fix worked but did not go far enough, and the campaign moved to scope
`karpathy_228791f_corrected_bpb_h200_train300s_v2c`. The reasoning is worth keeping
because it is the first time this campaign measured *why* it was slow instead of
guessing.

**v2b baseline, 4 valid controls:** `val_bpb` 1.1098–1.1142, mean **277 steps**,
**11.3% MFU**, σ = 0.00226. Against the (withdrawn, unsourced) 1006 steps / 42% / 0.9912 anchor that is a
3.7× step deficit, and his own token law (0.09–0.13 bpb per e-fold) predicts a
0.118–0.170 penalty for it. Observed gap: **0.123**. The whole difference was
throughput; nothing was wrong with the model. `train.py` was diffed against the
pristine `228791f` copy and differs only in seed plumbing, the clock start, and the
metrics endpoint — the batch math is byte-identical, so step counts compare 1:1.

**The bottleneck, measured not assumed.** Step time is bimodal on all six landed
runs: the 64-batch prefetch queue fills for free during the 22–58 s `torch.compile`,
so steps 1–20 run at 526–684 ms, and once it drains every later step sits flat at
959–1157 ms. That 1.4–1.9× is the GPU waiting on the producer. The producer is not
slow because BPE is slow — `encode_ordinary_batch` already runs 8 threads — it is
slow because every run re-does identical work: parquet read, `.to_pylist()` into
millions of Python strings, encode, per-document Python lists. Four arms meant 32
encode threads on a host already at load ~325.

Decomposition, which closes exactly:

```
tokenizer starvation  1.78x  (1106ms steady / 620ms when fed)
remaining when fed    2.08x  ( 620ms fed    / 298ms Karpathy)
                     ------
                      3.71x  vs observed 1006/272 = 3.70x
```

So roughly half the deficit was the dataloader and half is this shared box. **Do not
expect sub-1.0 from the dataloader fix alone** — it predicts ~484 steps and
`val_bpb` ≈ 1.05. The second half needs the contention addressed, and that is the
open question v2c exists to measure.

Also note the target itself: `UPSTREAM.md` records that `prepare.py` deliberately
corrects a byte-fallback denominator bug, so **0.9912 is not this scope's number**.
It is a throughput sanity check, not a goal.

### Verify the fix worked — this is your first real task

Read the first control's `num_steps` and `mfu_percent`:

```bash
uv run autoresearch --root .autoresearch bank
uv run autoresearch --root .autoresearch doctor --bank-id karpathy_228791f
```

| outcome | reading | what to do |
|---|---|---|
| ~1000 steps, ~40% MFU, bpb ≈ 0.99 | fix worked, box is quiet | proceed to mechanism search |
| 400–800 steps, 15–30% MFU | fix worked, box still contended | proceed, but report the gap honestly |
| still ~200 steps, ~8% MFU | **the prefetch is not the whole story** | stop and profile before spending GPU |

In the third case the next suspects, in order: is the producer thread actually
overlapping (does `rustbpe.encode` release the GIL?); is the parquet row-group read
the real cost rather than the encode; and is another tenant sharing our physical
GPU (v2 detects this and invalidates the run — check the EvidenceDecision reasons).
Consider a pre-tokenized flat token cache, which is what `plain_v2` did and what
its own SOTA writeup credits for "the large majority" of its 0.0134 gain.

## 2. Hard rules — these are not negotiable

**GPUs 0, 3, 4, 7 are ours. GPUs 1, 2, 5, 6 belong to the `plain_v2` campaign.**
Never launch on them, never kill their workers, count them as occupied. `plain_v2`
is a live campaign at ~experiment 209 with a replicated SOTA of 0.95858.

**Start at width 1, and profile before widening.** The previous session ran 4-wide
without profiling and the CPU contention was self-inflicted. `runs/execution_solo_gpu0.json`
is one GPU; `runs/execution_fleet4.json` is four. Do not move to the fleet config
until a 1-wide control shows healthy MFU *and* you have measured that 2-wide does
not degrade it. v2's own design doc: *"Eight idle GPUs do not imply the host can
support eight healthy jobs."*

**`max_concurrent_jobs` is a HOST-wide slot count, not a per-GPU one.** It reads
like "jobs allowed on this GPU"; it is not. `execution.py` turns it into
`host_slots`, and every resource sharing `host_id: zp-nc71` competes for the same
`host_<host_id>_<slot>` locks — so `4` on a four-GPU fleet means *four jobs on the
host*, which is what you want. Setting it to `1` (attempted 2026-08-16, reverted
within minutes) caps the **entire box at one running job** and idles three H200s
while the queue reports four `running`. Two jobs can never land on one GPU anyway:
the per-GPU `resource_lease` is keyed `host_resource_gpu` and each GPU has its own
workdir lock. Leave it at the fleet width.

**Jobs are GPU-pinned, so a tenant-held GPU starves the free ones.** A worker that
claims a job pinned to an occupied GPU sits in resource-wait for
`resource_wait_seconds` (900) holding its slot, while a job pinned to a genuinely
free GPU stays `pending` behind it. With `--workers 4` and four pinned jobs, two
blocked GPUs meant **two idle H200s and a queue that looked busy** — `status` read
`running: 4` the entire time. Run more workers than pinned jobs so a free-GPU job
can never queue behind an occupied one; `max_concurrent_jobs: 1` makes the extra
workers harmless. `tools/wake.py` reports this state as
`CLAIMED_BUT_NOT_EXECUTING`.

**Nothing is SOTA except a held-out-seed confirmation.** A bank delta is a
work-allocation decision. Never report a seed-42 candidate delta as an
improvement, never report a best-of-N single run. Winner's curse here is measured,
not theoretical: two stacked selections regressed 0.44 gates on disjoint seeds.

**Never compare across scopes.** `v2b` is not comparable to `v2a`, to upstream
scores, or to `plain_v2`. Different corpus, different byte accounting, different
champion. Report each with its scope ID.

**Judge on measured `val_bpb` only.** No throughput correction, no adjusted bpb.
`num_steps` is a diagnostic printed beside every run, never an adjustment to it.

**Do not weaken a gate to make a number move.** When `doctor` says
`model_search_ready: false`, it is telling you the frame is overhead-dominated and
the measurement is the finding. That gate has been right every time so far.

## 3. The workflow

```
freeze scope → calibrate (per-GPU control bank) → search (candidate vs its own
frozen same-GPU control) → bank gate → promote (held-out seeds 43–47) → conclude
```

Commands, in the order you will need them:

```bash
uv run autoresearch --root .autoresearch status         # live state
uv run autoresearch --root .autoresearch queue --jobs   # what is queued/running
uv run autoresearch --root .autoresearch bank           # rebuild scored views
uv run autoresearch --root .autoresearch doctor --bank-id karpathy_228791f
uv run autoresearch --root .autoresearch validate       # must stay valid: true
uv run python tools/make_chart.py                       # regenerate the chart
```

Staging a candidate (note `--track`, and `--argv` must come last):

```bash
uv run autoresearch --root .autoresearch search my_label \
  --bank-id karpathy_228791f \
  --summary "one precise change and its expected mechanism" \
  --scope runs/scope.json --execution runs/execution_solo_gpu0.json \
  --mutable-code-path train.py \
  --hypothesis-id hyp_short_window_quarter_span \
  --track mechanism --family attention_span \
  --subsystem attention --allow-weak-science \
  --argv /bin/bash launch.sh
```

`--allow-weak-science` is the **default posture** for this campaign, deliberately:
the full literature gate would consume the session, and 5.7% of a predecessor's
123 claims ever did predictive work. `--hypothesis-id` is still required, because
a registered prediction and falsifier before launch is the cheap half that paid off.

Two hypotheses are already registered and ready to stage, with mechanisms,
predictions and falsifiers, in `runs/science/`:
`hyp_short_window_quarter_span` (S-layer span `//2` → `//4`) and
`hyp_window_pattern_ssssll` (SSSL → SSSSLL). Variants apply with
`python3 /tmp/variant.py <name>` if that file survives; otherwise the edits are
one-liners described in each hypothesis's `intervention.summary`.

## 4. Things that will bite you

**`--argv /bin/bash launch.sh`, never `.venv/bin/python train.py`.** v2 scrubs
`PATH` from the arm environment; Triton then shells out to `/bin/gcc`, which cannot
find `ld`, and every `torch.compile` dies with an empty metrics dict about two
minutes in. `runs/code/launch.sh` is the sealed launcher that fixes this. Do not
"fix" it by weakening the scrubbing.

**A stale `baseline_provenance.json` on the box fails every single arm.** Observed
2026-08-16: all four remote workdirs still held the pre-throughput-fix provenance
record (adapter `...-and-pinned-cache-v2`, `prepare_py_sha256 d7cece3e...`) while
`runs/code/provenance.json` had moved to the v3 prefetch adapter. Every arm died at
preflight in ~20 s with `execution status is preflight_failed` and one `mismatch`
in `binding_checks`, so nothing landed and the campaign looked idle rather than
broken. The reason it survives an ordinary re-sync: the file is mode **444** on the
box, so a plain `scp` fails with `Permission denied` and a scripted sync can skip
it silently while `prepare.py` and `train.py` (mode 644) update correctly. The
repair is `chmod u+w` → copy → `chmod 444`, then verify all four digests against
the local file. **When a whole wave fails fast, read `binding_checks` first** — the
mismatching `execution_path` names the file, and it is usually the one nobody
thinks of as code.

**If this Mac idle-sleeps, it invalidates every run in flight.** `wall_seconds` is
`time.monotonic()` measured *locally* around the ssh subprocess; `total_seconds` is
`time.time()` measured *inside train.py on the box*. macOS `monotonic` does not
advance across system sleep, so a nap makes the runner under-measure while the H200
keeps training, and `evidence.py` correctly refuses to reconcile the two:
`arm control emitted total_seconds=425.566, which does not match runner
wall_seconds=397.867 within 15s`. Observed 2026-08-16: a 36 s `'Idle Sleep'` at
06:01:08 PDT fell inside two concurrent runs and invalidated both with nearly
identical deficits (+27.7 s, +28.6 s) — **the matching offsets are the signature**,
because independent causes would not agree that closely.

Hold sleep off for the whole campaign, and check it before believing a timing
invalidation is about the fleet:

```bash
pgrep -x caffeinate || nohup caffeinate -i -m >/dev/null 2>&1 &
pmset -g log | grep -E '\bSleep\b' | tail -5      # did we nap during that run?
```

Do not respond by widening the 15 s tolerance. The check is right; the laptop is
the fault, and a widened tolerance would also hide real remote stalls.

**The health circuit will pause the queue** after 3 consecutive invalid pilots or
>25% of the last 12. It already did once. Diagnose the cause — read the
EvidenceDecision `reasons`, they are specific — before reaching for
`run --ignore-health`.

**GPU co-tenancy invalidates runs, correctly.** One run was killed with
`"arm control observed GPU co-tenancy"` because another tenant grabbed the card
mid-run. That is the single most valuable protection on this box. Never disable it.

**Editing `train.py` is the normal loop.** It is the only declared mutable path;
everything else is in the stable context fingerprint. Same argv + changed bytes is
valid; same argv + same bytes is a rejected no-op.

**The exploration budget will refuse candidates**, inside the staging lock:
≥5 of the trailing 8 must be `mechanism`; a `knob` needs its family to have cleared
the gate; no third consecutive candidate in one uncleared family. Diagnostic
subsystems (compile/data/input/evaluation/tokenizer/instrumentation) are exempt.
Overrides need a written reason ≥20 chars and are recorded immutably. If you are
writing three overrides in a row, the budget is not the problem.

**Data lives at `$HOME/.cache/autoresearch_v2`** on the box (shards 00000–00009 +
06542), pinned so the shared `~/.cache/autoresearch` cannot redefine the split.
`huggingface.co` is unreachable from that host; use `hf-mirror.com`.

## 5. Reporting

Regenerate and republish the chart after **every** landed experiment, never
batched:

```bash
uv run autoresearch --root .autoresearch bank && uv run python tools/make_chart.py
```

Then publish `runs/val_bpb_progress.html` as an artifact. If a previous artifact
URL exists, update it rather than creating a new one.

Report a concise line each cycle: new deltas, launches/completions, free-GPU count,
any promotion, any paper. Do not run silent. If the box is fully held by other
tenants for an extended period, say so as the headline rather than waiting quietly.

Per `HEARTBEAT.md`: dispatch **analyst / critic / scout agents concurrently** with
the GPUs, and **once per hour** have Fable write an analysis-and-critique pass to
`AI_papers/` that ends in concrete stageable experiments.

## 6. Environment

```bash
cd ~/Desktop/OPHIS/simplify_autoresearch_v2      # uv project, Python 3.11+
uv run pytest -q                                  # 129 tests, all should pass
uv run ruff check autoresearch tests              # 4 pre-existing findings in science.py

ssh -i "$OPHIS_SSH_KEY" -o IdentitiesOnly=yes -p "$OPHIS_SSH_PORT" "$OPHIS_SSH_TARGET"
# remote workdirs: $OPHIS_REMOTE_ROOT/gpu{0,3,4,7}, each with its own .venv (torch 2.9.1+cu128)
```

The box is shared and volatile: 20 tenants, load average 280–370, `nvidia-smi`
showing free GPUs is not a reservation. Check free VRAM immediately before
launching, and expect SSH drops — retry rather than ending a turn.
