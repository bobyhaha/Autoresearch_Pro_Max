from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "tools" / "build_karpathy_manifest.py"


def _make_cache(tmp_path: Path) -> Path:
    cache = tmp_path / "cache"
    data = cache / "data"
    tokenizer = cache / "tokenizer"
    data.mkdir(parents=True)
    tokenizer.mkdir()
    for index in range(10):
        (data / f"shard_{index:05d}.parquet").write_bytes(f"train-{index}".encode())
    (data / "shard_06542.parquet").write_bytes(b"validation")
    (tokenizer / "token_bytes.pt").write_bytes(b"corrected-byte-table")
    (tokenizer / "token_bytes.version").write_text("2\n", encoding="utf-8")
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")
    return cache


def _run(cache: Path, output: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--cache-dir", str(cache), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_builder_writes_a_deterministic_schema_v2_manifest_atomically(tmp_path):
    cache = _make_cache(tmp_path)
    output = tmp_path / "nested" / "data_manifest.json"

    first = _run(cache, output)
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()
    second = _run(cache, output)
    assert second.returncode == 0, second.stderr
    assert output.read_bytes() == first_bytes
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))

    manifest = json.loads(first_bytes)
    assert manifest["schema_version"] == 2
    assert manifest["benchmark"] == "karpathy/autoresearch"
    assert manifest["token_bytes_version"] == 2
    assert manifest["data_split"]["train"] == [
        f"data/shard_{index:05d}.parquet" for index in range(10)
    ]
    assert manifest["data_split"]["validation"] == ["data/shard_06542.parquet"]
    assert len(manifest["files"]) == 14
    assert [row["path"] for row in manifest["files"]] == sorted(
        row["path"] for row in manifest["files"]
    )
    for row in manifest["files"]:
        path = cache / row["path"]
        assert row["bytes"] == path.stat().st_size
        assert row["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_version", "token_bytes.version must be 2"),
        ("extra_shard", "extra=['shard_00010.parquet']"),
        ("missing_shard", "missing=['shard_00009.parquet']"),
    ],
)
def test_builder_rejects_a_noncanonical_cache(tmp_path, mutation, message):
    cache = _make_cache(tmp_path)
    if mutation == "wrong_version":
        (cache / "tokenizer" / "token_bytes.version").write_text("1\n", encoding="utf-8")
    elif mutation == "extra_shard":
        (cache / "data" / "shard_00010.parquet").write_bytes(b"extra")
    else:
        (cache / "data" / "shard_00009.parquet").unlink()

    output = tmp_path / "data_manifest.json"
    completed = _run(cache, output)
    assert completed.returncode == 2
    assert message in completed.stderr
    assert not output.exists()
