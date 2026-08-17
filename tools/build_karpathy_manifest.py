#!/usr/bin/env python3
"""Build the immutable data manifest for the Karpathy autoresearch baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

TRAIN_SHARDS = tuple(f"shard_{index:05d}.parquet" for index in range(10))
VALIDATION_SHARD = "shard_06542.parquet"
EXPECTED_SHARDS = (*TRAIN_SHARDS, VALIDATION_SHARD)
TOKENIZER_ARTIFACTS = ("token_bytes.pt", "token_bytes.version", "tokenizer.pkl")
TOKEN_BYTES_VERSION = 2
TOKENIZER_ARTIFACT_LIMIT = 64 * 1024 * 1024


class ManifestError(ValueError):
    """The cache cannot define the intended Karpathy benchmark scope."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_shards(cache_dir: Path) -> list[Path]:
    data_dir = cache_dir / "data"
    if not data_dir.is_dir():
        raise ManifestError(f"data directory does not exist: {data_dir}")
    # prepare.py enumerates by filename, not by file type. Count every parquet-named
    # directory entry so a broken symlink or directory cannot hide from the manifest
    # builder and then surprise pyarrow at runtime.
    actual = sorted(path.name for path in data_dir.iterdir() if path.name.endswith(".parquet"))
    missing = sorted(set(EXPECTED_SHARDS) - set(actual))
    extra = sorted(set(actual) - set(EXPECTED_SHARDS))
    if missing or extra:
        raise ManifestError(
            "Karpathy baseline requires exactly shards 00000-00009 and 06542; "
            f"missing={missing}, extra={extra}"
        )
    return [data_dir / name for name in EXPECTED_SHARDS]


def _require_tokenizer(cache_dir: Path) -> list[Path]:
    tokenizer_dir = cache_dir / "tokenizer"
    if not tokenizer_dir.is_dir():
        raise ManifestError(f"tokenizer directory does not exist: {tokenizer_dir}")
    paths = [tokenizer_dir / name for name in TOKENIZER_ARTIFACTS]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ManifestError(f"missing tokenizer artifacts: {missing}")
    version_path = tokenizer_dir / "token_bytes.version"
    try:
        version = version_path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read {version_path}: {exc}") from exc
    if version != str(TOKEN_BYTES_VERSION):
        raise ManifestError(
            f"token_bytes.version must be {TOKEN_BYTES_VERSION}, found {version!r}; "
            "rebuild token_bytes.pt with decode_single_token_bytes()"
        )
    oversized = [path.name for path in paths if path.stat().st_size > TOKENIZER_ARTIFACT_LIMIT]
    if oversized:
        raise ManifestError(
            f"tokenizer artifacts exceed {TOKENIZER_ARTIFACT_LIMIT} bytes: {oversized}"
        )
    return paths


def build_manifest(cache_dir: Path) -> dict[str, Any]:
    cache_dir = cache_dir.expanduser().resolve(strict=True)
    shards = _require_exact_shards(cache_dir)
    tokenizer = _require_tokenizer(cache_dir)
    files = []
    for path in (*shards, *tokenizer):
        relative = path.relative_to(cache_dir).as_posix()
        files.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    files.sort(key=lambda row: row["path"])
    return {
        "schema_version": 2,
        "benchmark": "karpathy/autoresearch",
        "cache_dir": str(cache_dir),
        "data_split": {
            "train": [f"data/{name}" for name in TRAIN_SHARDS],
            "validation": [f"data/{VALIDATION_SHARD}"],
        },
        "token_bytes_version": TOKEN_BYTES_VERSION,
        "files": files,
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        payload = build_manifest(args.cache_dir)
        atomic_write_json(args.output, payload)
    except (ManifestError, OSError) as exc:
        parser.error(str(exc))
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
