"""
One-time data preparation for autoresearch experiments.
Downloads data shards and trains a BPE tokenizer.

Usage:
    python prepare.py                  # full prep (download + tokenizer)
    python prepare.py --num-shards 8   # download only 8 shards (for testing)

Data and tokenizer are stored in ~/.cache/autoresearch/.
"""

import os
import hashlib
import json
import queue
import threading
import sys
import time
import math
import argparse
import pickle
from multiprocessing import Pool

import numpy as np
import requests
import pyarrow.parquet as pq
import rustbpe
import tiktoken
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 2048       # context length
TIME_BUDGET = 300        # training time budget in seconds (5 minutes)
EVAL_TOKENS = 40 * 524288  # number of tokens for val eval

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# OPHIS protocol adapter: prepare.py enumerates the training split by directory
# listing, so the *contents* of CACHE_DIR/data silently define dataset_split.  A
# shared cache that other campaigns write to is therefore not a frozen scope.  V2
# pins its own cache root so shards 00000-00009 + 06542 stay exact for the life of
# the campaign.  The value is a constant, not an environment lookup, so the sealed
# prepare.py digest fully determines which corpus a run consumed.
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch_v2")
DATA_DIR = os.path.join(CACHE_DIR, "data")
TOKENIZER_DIR = os.path.join(CACHE_DIR, "tokenizer")
BASE_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main"
MAX_SHARD = 6542 # the last datashard is shard_06542.parquet
VAL_SHARD = MAX_SHARD  # pinned validation shard (shard_06542)
VAL_FILENAME = f"shard_{VAL_SHARD:05d}.parquet"
VOCAB_SIZE = 8192

# BPE split pattern (GPT-4 style, with \p{N}{1,2} instead of {1,3})
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [f"<|reserved_{i}|>" for i in range(4)]
BOS_TOKEN = "<|reserved_0|>"

# OPHIS evaluator correction.  Upstream used decode([id]).encode("utf-8"),
# which turns invalid byte-fallback tokens into the three-byte replacement
# character.  Version the lookup so an existing upstream cache cannot silently
# keep the biased denominator.
TOKEN_BYTES_VERSION = 2
TOKEN_CACHE_VERSION = 1
TOKEN_CACHE_DIR = os.path.join(CACHE_DIR, "token_cache")
TOKEN_BYTES_VERSION_FILE = "token_bytes.version"

# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def download_single_shard(index):
    """Download one parquet shard with retries. Returns True on success."""
    filename = f"shard_{index:05d}.parquet"
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        return True

    url = f"{BASE_URL}/{filename}"
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filepath)
            print(f"  Downloaded {filename}")
            return True
        except (requests.RequestException, IOError) as e:
            print(f"  Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            if attempt < max_attempts:
                time.sleep(2 ** attempt)
    return False


def download_data(num_shards, download_workers=8):
    """Download training shards + pinned validation shard."""
    os.makedirs(DATA_DIR, exist_ok=True)
    num_train = min(num_shards, MAX_SHARD)
    ids = list(range(num_train))
    if VAL_SHARD not in ids:
        ids.append(VAL_SHARD)

    # Count what's already downloaded
    existing = sum(1 for i in ids if os.path.exists(os.path.join(DATA_DIR, f"shard_{i:05d}.parquet")))
    if existing == len(ids):
        print(f"Data: all {len(ids)} shards already downloaded at {DATA_DIR}")
        return

    needed = len(ids) - existing
    print(f"Data: downloading {needed} shards ({existing} already exist)...")

    workers = max(1, min(download_workers, needed))
    with Pool(processes=workers) as pool:
        results = pool.map(download_single_shard, ids)

    ok = sum(1 for r in results if r)
    print(f"Data: {ok}/{len(ids)} shards ready at {DATA_DIR}")

# ---------------------------------------------------------------------------
# Tokenizer training
# ---------------------------------------------------------------------------

def list_parquet_files():
    """Return sorted list of parquet file paths in the data directory."""
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


def text_iterator(max_chars=1_000_000_000, doc_cap=10_000):
    """Yield documents from training split (all shards except pinned val shard)."""
    parquet_paths = [p for p in list_parquet_files() if not p.endswith(VAL_FILENAME)]
    nchars = 0
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx)
            for text in rg.column("text").to_pylist():
                doc = text[:doc_cap] if len(text) > doc_cap else text
                nchars += len(doc)
                yield doc
                if nchars >= max_chars:
                    return


def _token_bytes_is_current(version_path):
    try:
        with open(version_path) as f:
            return int(f.read().strip()) == TOKEN_BYTES_VERSION
    except (OSError, ValueError):
        return False


def _write_token_bytes(enc, token_bytes_path, version_path):
    """Write the raw-byte BPB denominator and its explicit format version."""
    print("Tokenizer: building corrected raw token_bytes lookup...")
    special_set = set(SPECIAL_TOKENS)
    token_bytes_list = []
    for token_id in range(enc.n_vocab):
        token_str = enc.decode([token_id])
        if token_str in special_set:
            token_bytes_list.append(0)
        else:
            token_bytes_list.append(len(enc.decode_single_token_bytes(token_id)))
    torch.save(torch.tensor(token_bytes_list, dtype=torch.int32), token_bytes_path)
    temporary_version = version_path + ".tmp"
    with open(temporary_version, "w") as f:
        f.write(f"{TOKEN_BYTES_VERSION}\n")
    os.replace(temporary_version, version_path)
    print(
        f"Tokenizer: saved token_bytes v{TOKEN_BYTES_VERSION} to {token_bytes_path}"
    )


def train_tokenizer():
    """Train BPE tokenizer using rustbpe, save as tiktoken pickle."""
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    token_bytes_path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    version_path = os.path.join(TOKENIZER_DIR, TOKEN_BYTES_VERSION_FILE)

    if (
        os.path.exists(tokenizer_pkl)
        and os.path.exists(token_bytes_path)
        and _token_bytes_is_current(version_path)
    ):
        print(f"Tokenizer: already trained at {TOKENIZER_DIR}")
        return

    os.makedirs(TOKENIZER_DIR, exist_ok=True)

    if os.path.exists(tokenizer_pkl):
        # Migrate an upstream cache without retraining or changing the tokenizer.
        with open(tokenizer_pkl, "rb") as f:
            enc = pickle.load(f)
        print("Tokenizer: reusing tokenizer.pkl and correcting its byte-length table")
    else:
        parquet_files = list_parquet_files()
        if len(parquet_files) < 2:
            print(
                "Tokenizer: need at least 2 data shards "
                "(1 train + 1 val). Download more data first."
            )
            sys.exit(1)

        # --- Train with rustbpe ---
        print("Tokenizer: training BPE tokenizer...")
        t0 = time.time()

        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = VOCAB_SIZE - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(
            text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN
        )

        # Build tiktoken encoding from trained merges
        pattern = tokenizer.get_pattern()
        mergeable_ranks = {bytes(k): v for k, v in tokenizer.get_mergeable_ranks()}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )

        # Save tokenizer
        with open(tokenizer_pkl, "wb") as f:
            pickle.dump(enc, f)

        t1 = time.time()
        print(f"Tokenizer: trained in {t1 - t0:.1f}s, saved to {tokenizer_pkl}")

    _write_token_bytes(enc, token_bytes_path, version_path)

    # Sanity check
    test = "Hello world! Numbers: 123. Unicode: 你好"
    encoded = enc.encode_ordinary(test)
    decoded = enc.decode(encoded)
    assert decoded == test, f"Tokenizer roundtrip failed: {test!r} -> {decoded!r}"
    print(f"Tokenizer: sanity check passed (vocab_size={enc.n_vocab})")

# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

class Tokenizer:
    """Minimal tokenizer wrapper. Training is handled above."""

    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=TOKENIZER_DIR):
        with open(os.path.join(tokenizer_dir, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    version_path = os.path.join(TOKENIZER_DIR, TOKEN_BYTES_VERSION_FILE)
    if not _token_bytes_is_current(version_path):
        raise RuntimeError(
            "token_bytes is stale or unversioned; run the sealed prepare.py once "
            "to build the corrected raw-byte lookup"
        )
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    """Infinite iterator over document batches from parquet files."""
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, "No parquet files found. Run prepare.py first."
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        parquet_paths = [p for p in parquet_paths if p != val_path]
        assert len(parquet_paths) > 0, "No training shards found."
    else:
        parquet_paths = [val_path]
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


# ---------------------------------------------------------------------------
# OPHIS protocol adapter: pre-tokenized flat token cache
#
# Measured on this campaign's own controls: step time is bimodal.  The 64-batch
# prefetch queue fills for free during the 22-58s torch.compile, so steps 1-20 run
# at 526-684ms; once it drains, every remaining step sits at 959-1157ms, flat, on
# all six landed runs.  That 1.4-1.9x is the GPU waiting on the producer.
#
# The producer is not slow because BPE is slow -- tiktoken's
# `encode_ordinary_batch` already runs 8 threads.  It is slow because every step
# re-does work that is *identical on every run of the campaign*: read a parquet row
# group, materialize it with `.to_pylist()` into millions of Python strings, encode,
# and build a Python list per document.  With four arms sharing this host that is
# 32 encode threads plus four parquet readers competing for a box already at load
# ~325.
#
# So do it once.  The token stream a run consumes is a pure function of (shards,
# tokenizer, batching) -- all three frozen by the scope -- so it can be computed
# ahead of time and replayed.  The cache stores, for exactly one epoch:
#
#   <split>.tokens.u16   every document's ids, concatenated, in document order
#   <split>.docs.i64     document boundaries, n_docs + 1 offsets
#   <split>.batches.i32  documents per yielded batch, so the *grouping* is replayed
#   <split>.manifest.json version + corpus fingerprint + counts
#
# Replaying `batches.i32` matters and is easy to miss: the consumer refills
# `doc_buffer` in whole batches and then best-fit packs whatever is in the buffer,
# so a different grouping would change which documents are co-resident at packing
# time and therefore change the token stream.  Storing the grouping keeps the
# stream identical rather than merely equivalent.
#
# uint16 is exact here (VOCAB_SIZE 8192) and halves both the file and the page
# cache footprint.  The array is memory-mapped, so the four arms on this host share
# one set of pages instead of each holding its own decoded copy.
# ---------------------------------------------------------------------------


def _token_cache_paths(split):
    base = os.path.join(TOKEN_CACHE_DIR, split)
    return (
        base + ".tokens.u16",
        base + ".docs.i64",
        base + ".batches.i32",
        base + ".manifest.json",
    )


def _corpus_fingerprint(split):
    """Identify the exact corpus + tokenizer a cache was built from.

    Shard name and byte size rather than a content hash: the shards are large and
    immutable, and a rebuild is cheap enough that a false miss costs far less than
    hashing a gigabyte on every run.  The tokenizer digest is included because a
    retrained tokenizer changes every token id.
    """
    paths = list_parquet_files()
    val_path = os.path.join(DATA_DIR, VAL_FILENAME)
    if split == "train":
        paths = [p for p in paths if p != val_path]
    else:
        paths = [val_path]
    shards = [
        {"name": os.path.basename(p), "bytes": os.path.getsize(p)} for p in sorted(paths)
    ]
    tokenizer_pkl = os.path.join(TOKENIZER_DIR, "tokenizer.pkl")
    with open(tokenizer_pkl, "rb") as f:
        tokenizer_sha = hashlib.sha256(f.read()).hexdigest()
    return {
        "version": TOKEN_CACHE_VERSION,
        "split": split,
        "shards": shards,
        "tokenizer_sha256": tokenizer_sha,
        "tokenizer_batch_size": 128,
        "vocab_size": VOCAB_SIZE,
    }


def build_token_cache(tokenizer, split, tokenizer_batch_size=128):
    """Tokenize exactly one epoch of `split` and write the replayable cache.

    Walks the same `_document_batches` generator the live path walks and stops at
    the epoch boundary, so what is stored is by construction the first epoch the
    live path would have produced.
    """
    tokens_path, docs_path, batches_path, manifest_path = _token_cache_paths(split)
    os.makedirs(TOKEN_CACHE_DIR, exist_ok=True)
    fingerprint = _corpus_fingerprint(split)

    bos_token = tokenizer.get_bos_token_id()
    batches = _document_batches(split, tokenizer_batch_size=tokenizer_batch_size)

    token_chunks = []
    doc_lengths = []
    batch_sizes = []
    total_tokens = 0
    t0 = time.time()
    for doc_batch, batch_epoch in batches:
        if batch_epoch != 1:
            break  # one epoch is the whole cache; the loader cycles it
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        batch_sizes.append(len(token_lists))
        for ids in token_lists:
            doc_lengths.append(len(ids))
            total_tokens += len(ids)
        token_chunks.append(
            np.fromiter(
                (tid for ids in token_lists for tid in ids),
                dtype=np.uint16,
                count=sum(len(ids) for ids in token_lists),
            )
        )
        if len(batch_sizes) % 500 == 0:
            print(
                f"token cache [{split}]: {len(batch_sizes)} batches, "
                f"{total_tokens / 1e6:.1f}M tokens, {time.time() - t0:.0f}s",
                flush=True,
            )

    tokens = np.concatenate(token_chunks) if token_chunks else np.empty(0, np.uint16)
    offsets = np.zeros(len(doc_lengths) + 1, dtype=np.int64)
    np.cumsum(np.asarray(doc_lengths, dtype=np.int64), out=offsets[1:])
    sizes = np.asarray(batch_sizes, dtype=np.int32)

    # Write to temporaries and rename, so a killed build never leaves a cache that
    # looks complete.  The manifest lands last and is what readers gate on.
    for path, array in (
        (tokens_path, tokens),
        (docs_path, offsets),
        (batches_path, sizes),
    ):
        with open(path + ".tmp", "wb") as f:
            array.tofile(f)
        os.replace(path + ".tmp", path)

    fingerprint.update(
        {
            "num_documents": int(len(doc_lengths)),
            "num_tokens": int(total_tokens),
            "num_batches": int(len(batch_sizes)),
            "built_seconds": round(time.time() - t0, 1),
        }
    )
    with open(manifest_path + ".tmp", "w") as f:
        json.dump(fingerprint, f, indent=2, sort_keys=True)
    os.replace(manifest_path + ".tmp", manifest_path)
    print(
        f"token cache [{split}]: {len(doc_lengths):,} docs, {total_tokens / 1e6:.1f}M "
        f"tokens, {tokens.nbytes / 1e6:.0f}MB in {time.time() - t0:.0f}s",
        flush=True,
    )
    return manifest_path


def load_token_cache(split):
    """Return (tokens, doc_offsets, batch_sizes) or None when absent or stale."""
    tokens_path, docs_path, batches_path, manifest_path = _token_cache_paths(split)
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return None

    expected = _corpus_fingerprint(split)
    if any(manifest.get(k) != v for k, v in expected.items()):
        return None
    try:
        tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
        offsets = np.fromfile(docs_path, dtype=np.int64)
        sizes = np.fromfile(batches_path, dtype=np.int32)
    except OSError:
        return None
    if (
        len(offsets) != manifest["num_documents"] + 1
        or len(sizes) != manifest["num_batches"]
        or len(tokens) != manifest["num_tokens"]
        or int(offsets[-1]) != manifest["num_tokens"]
    ):
        return None
    return tokens, offsets, sizes


def _cached_document_token_batches(split):
    """Replay the live path's (token_lists, epoch) sequence from the cache.

    Yields the same batches, in the same order, with the same per-batch document
    counts and the same epoch numbering.  Documents come back as uint16 array views
    rather than Python lists: the consumer only takes `len(doc)`, slices it, and
    hands it to `torch.tensor`, all of which a numpy view supports, and avoiding the
    per-token Python objects is most of the point.
    """
    cache = load_token_cache(split)
    if cache is None:
        return None
    tokens, offsets, sizes = cache

    def generator():
        epoch = 1
        while True:
            doc = 0
            for size in sizes:
                yield [
                    tokens[offsets[doc + i] : offsets[doc + i + 1]] for i in range(size)
                ], epoch
                doc += int(size)
            epoch += 1

    return generator()


def make_dataloader(tokenizer, B, T, split, buffer_size=1000, prefetch_batches=64):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
    assert split in ["train", "val"]
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    # OPHIS protocol adapter: overlap tokenization with GPU compute.
    #
    # Upstream reads parquet and runs the BPE encoder *inline in the training
    # loop*, single-threaded, with no prefetch.  On an uncontended box the encoder
    # keeps ahead of the GPU and this is free.  On the shared host this campaign
    # runs on -- load average ~350, 20 tenants -- that one CPU thread is starved
    # and the GPU waits on it: measured 194 steps at 7.8% MFU where a
    # pre-tokenized trainer on the same box in the same hour reached 1455 steps at
    # 39.6% MFU.  The bottleneck is the pipeline, not the hardware or the model.
    #
    # This is a pure throughput change and deliberately NOT a data change.  One
    # producer thread walks `batches` in the original order and a bounded FIFO
    # queue hands the encoded documents back, so the consumer observes exactly the
    # token stream, document order, and epoch boundaries it observed before -- the
    # tokens are merely produced ahead of time instead of on demand.  pyarrow and
    # the Rust BPE encoder both release the GIL, so the producer runs during the
    # main thread's CUDA waits.
    # Prefer the pre-tokenized cache when it matches this corpus and tokenizer.
    # It emits the identical (token_lists, epoch) sequence, so the consumer below
    # is unchanged and unaware of which producer it is draining.  Falling back
    # rather than failing keeps a standalone `uv run train.py` working on a machine
    # that has never built a cache.
    cached_batches = _cached_document_token_batches(split)
    if cached_batches is not None:
        batches = cached_batches
        print(f"dataloader [{split}]: using pre-tokenized cache", flush=True)

        def _produce(item):
            return item  # already (token_lists, epoch)

    else:
        print(f"dataloader [{split}]: tokenizing inline (no cache)", flush=True)

        def _produce(item):
            doc_batch, batch_epoch = item
            return tokenizer.encode(doc_batch, prepend=bos_token), batch_epoch

    token_queue = queue.Queue(maxsize=prefetch_batches)
    producer_error = []

    def _producer():
        try:
            for item in batches:
                token_queue.put(_produce(item))
        except BaseException as exc:  # surfaced on the consumer thread, never swallowed
            producer_error.append(exc)
            token_queue.put(None)

    threading.Thread(target=_producer, name="ophis-tokenizer-prefetch", daemon=True).start()

    def refill_buffer():
        nonlocal epoch
        item = token_queue.get()
        if item is None:
            raise RuntimeError("tokenizer prefetch thread failed") from producer_error[0]
        token_lists, epoch = item
        doc_buffer.extend(token_lists)

    # Pre-allocate buffers: [inputs (B*T) | targets (B*T)]
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    # No doc fits — crop shortest to fill remaining
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=True)
        yield inputs, targets, epoch

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Bits per byte (BPB): vocab size-independent evaluation metric.
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte. Special tokens (byte length 0)
    are excluded from both sums.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    token_bytes = get_token_bytes(device="cuda")
    val_loader = make_dataloader(tokenizer, batch_size, MAX_SEQ_LEN, "val")
    steps = EVAL_TOKENS // (batch_size * MAX_SEQ_LEN)
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--num-shards", type=int, default=10, help="Number of training shards to download (-1 = all). Val shard is always pinned.")
    parser.add_argument("--download-workers", type=int, default=8, help="Number of parallel download workers")
    parser.add_argument(
        "--build-token-cache",
        action="store_true",
        help="tokenize one epoch of each split to the replayable flat cache",
    )
    args = parser.parse_args()

    num_shards = MAX_SHARD if args.num_shards == -1 else args.num_shards

    print(f"Cache directory: {CACHE_DIR}")
    print()

    # Step 1: Download data
    download_data(num_shards, download_workers=args.download_workers)
    print()

    # Step 2: Train tokenizer
    train_tokenizer()
    print()

    # Step 3 (optional, one-time): pre-tokenize.  Not run by default because the
    # training path falls back to inline tokenization and must stay usable on a
    # machine that has never built one.
    if args.build_token_cache:
        tokenizer = Tokenizer.from_directory()
        for split in ("train", "val"):
            if load_token_cache(split) is not None:
                print(f"token cache [{split}]: already current")
                continue
            build_token_cache(tokenizer, split)
        print()
    print("Done! Ready to train.")
