#!/usr/bin/env python3
"""Prove the pre-tokenized cache replays the inline path's exact token stream.

A throughput change to the dataloader is only a throughput change if the model
sees the same tokens in the same order with the same epoch boundaries. That is an
empirical claim, so measure it rather than assert it: build the dataloader both
ways from the same tokenizer and hash N batches of inputs+targets from each.

This is the same check that licensed the earlier prefetch change (12 batches
hashing to f4ab4f33...), applied to the cache. Run it on the box, where the corpus
lives, and run it BEFORE sealing a scope around the new prepare.py:

    .venv/bin/python tools/verify_token_cache.py --batches 12

Exit 0 means identical. Any other exit means do not seal.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "runs" / "code"))

import prepare


def stream_digest(tokenizer, *, batches: int, B: int, T: int, split: str, use_cache: bool):
    """Hash `batches` (inputs, targets) pairs from one dataloader configuration."""

    real_loader = prepare._cached_document_token_batches
    if not use_cache:
        # Force the inline path without editing prepare.py: the loader asks this
        # helper for a cache and falls back when it gets None.
        prepare._cached_document_token_batches = lambda split: None
    try:
        loader = prepare.make_dataloader(tokenizer, B, T, split)
        digest = hashlib.sha256()
        shapes = []
        for _ in range(batches):
            # The loader yields (inputs, targets, epoch). The epoch goes into the
            # hash too: epoch boundaries drive the training schedule, so replaying
            # the right tokens under the wrong epoch number is still a data change.
            inputs, targets, epoch = next(loader)
            shapes.append((tuple(inputs.shape), int(epoch)))
            for tensor in (inputs, targets):
                digest.update(tensor.to("cpu", torch.int64).contiguous().numpy().tobytes())
            digest.update(str(int(epoch)).encode())
        return digest.hexdigest(), shapes
    finally:
        prepare._cached_document_token_batches = real_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=prepare.MAX_SEQ_LEN)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    if prepare.load_token_cache(args.split) is None:
        raise SystemExit(
            f"no current token cache for split '{args.split}' -- build it first with "
            "`python prepare.py --build-token-cache`"
        )

    tokenizer = prepare.Tokenizer.from_directory()

    print(f"hashing {args.batches} batches of {args.batch_size}x{args.seq_len} ...")
    inline_digest, inline_shapes = stream_digest(
        tokenizer, batches=args.batches, B=args.batch_size, T=args.seq_len,
        split=args.split, use_cache=False,
    )
    print(f"  inline tokenization : {inline_digest}")
    cached_digest, cached_shapes = stream_digest(
        tokenizer, batches=args.batches, B=args.batch_size, T=args.seq_len,
        split=args.split, use_cache=True,
    )
    print(f"  pre-tokenized cache : {cached_digest}")

    if inline_shapes != cached_shapes:
        print("MISMATCH: batch shapes differ", file=sys.stderr)
        raise SystemExit(2)
    if inline_digest != cached_digest:
        print(
            "MISMATCH: the cache does not replay the inline token stream. "
            "Do NOT seal a scope around this prepare.py.",
            file=sys.stderr,
        )
        raise SystemExit(3)
    print("IDENTICAL — the cache is a throughput change only.")


if __name__ == "__main__":
    main()
