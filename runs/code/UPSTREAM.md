# Karpathy autoresearch baseline

The active `train.py` starts from Andrej Karpathy's official
[`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) baseline at commit
`228791fb499afffb54b46200aca536f79142f117`. The untouched source is retained under
`upstream/` and pinned by `provenance.json`.

The model, optimizer, hyperparameters, data loader, 300-second charged-training clock, and
evaluation procedure in the active `train.py` start from that source. Its only OPHIS
protocol changes are:

- read the executable seed from `AUTORESEARCH_SEED`, falling back to upstream seed 42 for
  a standalone run;
- seed both Torch CPU and CUDA state with that value;
- start the emitted total-time clock before Torch/kernel imports so it reconciles with the
  independent runner wall clock, without changing the charged training clock;
- retain the upstream `---` summary and append one `AUTORESEARCH_METRICS` JSON endpoint
  containing the same measurements plus the executed seed.

The active `prepare.py` also starts from the same upstream commit, but deliberately fixes
the known byte-fallback denominator error. It uses `decode_single_token_bytes()` and
versions `token_bytes.pt` as format 2. An existing upstream tokenizer is reused; only the
derived byte-length table is rebuilt. This corrected-BPB scope is not numerically
interchangeable with historical upstream scores produced by
`decode(...).encode("utf-8")`.

Upstream's README declares the project MIT licensed, although that commit does not contain
a standalone `LICENSE` file. Preserve this attribution, repository URL, commit, and source
digests when redistributing the snapshot.

The upstream runtime resolves its Flash Attention kernel dynamically. For a real campaign,
pre-cache and pin that kernel in the runtime image and record the image/kernel digest in the
structured scope before calibration.
