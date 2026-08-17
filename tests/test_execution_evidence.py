from __future__ import annotations

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from conftest import approvals, execution_config, make_project, make_spec

import autoresearch.execution as execution_module
from autoresearch.evidence import EvidenceEngine
from autoresearch.execution import (
    Allocation,
    ExecutionService,
    NoPendingReplicate,
    build_runner_argv,
)
from autoresearch.knowledge import KnowledgeEngine
from autoresearch.records import make_record
from autoresearch.research import ResearchEngine
from autoresearch.sealing import SealingAuthority
from autoresearch.store import Store


def test_pilot_end_to_end_is_valid_but_claim_ineligible(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    decision = EvidenceEngine(store).judge(result["id"])
    assert decision["payload"]["measurement_verdict"] == "valid"
    assert decision["payload"]["claim_status"] == "ineligible"
    snapshot = KnowledgeEngine(store).synthesize()
    assert snapshot["beliefs"][0]["status"] == "exploratory"
    assert snapshot["beliefs"][0]["effect_mean"] == pytest.approx(-0.1)
    assert snapshot["sota"] == {}


def test_confirmation_aggregates_replicates_and_promotes_mean_not_best_seed(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(script, replicates=2, candidate_values=[0.9, 0.8])
    )
    manifest = SealingAuthority(store).seal(
        spec["id"], execution_config(script, data), approvals(spec)
    )
    results, unfinished = ExecutionService(store).execute_all(manifest["id"], workers=2)
    assert unfinished == []
    assert {row["payload"]["replicate_id"] for row in results} == {"seed_1", "seed_2"}
    decisions = EvidenceEngine(store).judge_all()
    assert len(decisions) == 2
    assert all(row["payload"]["measurement_verdict"] == "valid" for row in decisions)
    snapshot = KnowledgeEngine(store).synthesize()
    belief = snapshot["beliefs"][0]
    assert belief["status"] == "supported"
    assert belief["effect_mean"] == pytest.approx(-0.15)
    sota = snapshot["sota"]["test_fixed_budget_v1"]
    assert sota["value"] == pytest.approx(0.85)
    assert sorted(sota["replicate_values"]) == [0.8, 0.9]
    assert all((store.root / row["blob"]).is_file() for row in sota["code_snapshots"])
    assert store.validate()["valid"] is True


def test_extra_workers_do_not_turn_safe_slot_contention_into_failure(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(script, replicates=3, candidate_values=[0.9, 0.8, 0.7])
    )
    execution = execution_config(script, data)
    execution["runtime"]["resource_wait_seconds"] = 0
    manifest = SealingAuthority(store).seal(spec["id"], execution, approvals(spec))
    results, unfinished = ExecutionService(store).execute_all(manifest["id"], workers=4)
    assert unfinished == []
    assert {row["payload"]["replicate_id"] for row in results} == {
        "seed_1",
        "seed_2",
        "seed_3",
    }
    assert store.validate()["valid"] is True


def test_mutated_code_fails_preflight_and_is_preserved(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    script.write_text(script.read_text() + "\n# mutation\n", encoding="utf-8")
    result = ExecutionService(store).execute_next(manifest["id"])
    assert result["payload"]["status"] == "preflight_failed"
    decision = EvidenceEngine(store).judge(result["id"])
    assert decision["payload"]["measurement_verdict"] == "invalid"
    assert any("did not match" in reason for reason in decision["payload"]["reasons"])


def test_failed_control_stops_same_slot_sequence(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script,
        stage="pilot",
        replicates=1,
        candidate_values=[0.9],
        fail_control=True,
        sota_eligible=False,
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    assert result["payload"]["status"] == "failed"
    assert [row["name"] for row in result["payload"]["arms"]] == ["control"]
    decision = EvidenceEngine(store).judge(result["id"])
    assert decision["payload"]["measurement_verdict"] == "invalid"


def test_replicate_is_never_launched_twice(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    service = ExecutionService(store)
    service.execute_next(manifest["id"])
    with pytest.raises(NoPendingReplicate):
        service.execute_next(manifest["id"])


def test_internal_runner_crash_preserves_claim_and_blocks_replay(tmp_path, monkeypatch):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))

    def crash(*args, **kwargs):
        raise RuntimeError("simulated runner boundary failure")

    monkeypatch.setattr(execution_module, "run_arm", crash)
    service = ExecutionService(store)
    with pytest.raises(RuntimeError, match="simulated"):
        service.execute_next(manifest["id"])
    claims = list(store.inflight_dir.glob("*.json"))
    assert len(claims) == 1
    assert "uncertain_after_internal_failure" in claims[0].read_text(encoding="utf-8")
    assert store.list("result_bundle") == []
    with pytest.raises(NoPendingReplicate):
        service.execute_next(manifest["id"])
    validation = store.validate()
    assert validation["valid"] is True
    assert validation["inflight"] == 1


def test_remote_runner_has_gpu_uuid_and_independent_timeout_watchdog(tmp_path):
    allocation = Allocation(
        resource={
            "id": "remote_1",
            "backend": "ssh",
            "ssh_argv": ["ssh", "-o", "BatchMode=yes", "worker@example"],
            "workdir": "/srv/research project",
            "gpus": [3],
        },
        gpu=3,
        launch_telemetry={"uuid": "GPU-abc123"},
    )
    argv, environment = build_runner_argv(
        allocation,
        ["python3", "train.py", "--label", "contains space"],
        {"RUN_ID": "replicate 1"},
        timeout_seconds=10.2,
    )
    assert argv[:4] == ["ssh", "-o", "BatchMode=yes", "worker@example"]
    assert "CUDA_VISIBLE_DEVICES=GPU-abc123" in argv[-1]
    assert "timeout --signal=TERM --kill-after=5s 11s" in argv[-1]
    assert "'contains space'" in argv[-1]
    assert environment == {}

    sanitized, _ = build_runner_argv(
        allocation,
        [".venv/bin/python", "train.py"],
        {"AUTORESEARCH_SEED": "42"},
        timeout_seconds=10.2,
        sanitize_environment=True,
    )
    assert "env -u BASH_ENV" in sanitized[-1]
    assert "-u PYTHONPATH" in sanitized[-1]
    assert "AUTORESEARCH_SEED=42" in sanitized[-1]


def test_evidence_independently_rejects_argv_and_binding_coverage_drift(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    drifted_payload = copy.deepcopy(result["payload"])
    drifted_payload["arms"][0]["payload_argv"][-1] = "9999"
    drifted_payload["binding_checks"] = drifted_payload["binding_checks"][:-1]
    drifted = store.put(make_record("result_bundle", "result_adversarial_drift", drifted_payload))
    decision = EvidenceEngine(store).judge(drifted["id"])
    assert decision["payload"]["measurement_verdict"] == "invalid"
    assert decision["payload"]["claim_status"] == "blocked"
    reasons = decision["payload"]["reasons"]
    assert any("argv differs" in reason for reason in reasons)
    assert any("do not exactly cover" in reason for reason in reasons)


def test_gpu_draining_from_our_own_previous_arm_does_not_abort_the_next_arm(monkeypatch):
    """A paired replicate must not be killed by the tail of its own first arm.

    nvidia-smi reports non-zero utilization for a few hundred milliseconds after a
    process exits.  Paired arms run back-to-back on one leased GPU -- the design that
    makes a screen placement-free -- so the pre-arm gate re-checks while the previous
    arm is still draining.  Observed on exp_flr02_paired: 72 percent utilization with
    ZERO compute processes and ZERO memory, 378 ms after the control arm ended.
    """
    from autoresearch import execution

    resource = {"max_idle_memory_mb": 1024, "max_idle_utilization_percent": 5}
    draining = {
        "state": "available",
        "process_count": 0,
        "memory_used_mb": 0.0,
        "utilization_percent": 72.0,
    }
    idle = {
        "state": "available",
        "process_count": 0,
        "memory_used_mb": 0.0,
        "utilization_percent": 0.0,
    }
    samples = [draining, draining, idle]
    monkeypatch.setattr(execution, "probe", lambda *_args, **_kwargs: samples.pop(0))
    monkeypatch.setattr(execution.time, "sleep", lambda _seconds: None)

    settled = execution._await_idle(resource, 4)

    assert execution._is_available(resource, settled, require_gpu=True)
    assert not samples, "should have kept probing until the GPU read idle"


def test_a_real_co_tenant_still_fails_the_gate_after_the_settle_window(monkeypatch):
    """The settle window must not become a way to launch onto an occupied GPU.

    This is the failure mode the gate exists for: a neighbour arriving mid-run
    invalidated four of six replicates on exp_confirm_placement and flipped the sign of
    the apparent effect.  A real tenant is still there after the window; our own
    teardown is not.  That is the only distinction _await_idle is allowed to make.
    """
    from autoresearch import execution

    resource = {"max_idle_memory_mb": 1024, "max_idle_utilization_percent": 5}
    occupied = {
        "state": "available",
        "process_count": 1,
        "memory_used_mb": 40000.0,
        "utilization_percent": 99.0,
    }
    probes = {"n": 0}

    def _probe(*_args, **_kwargs):
        probes["n"] += 1
        return dict(occupied)

    monkeypatch.setattr(execution, "probe", _probe)
    monkeypatch.setattr(execution.time, "sleep", lambda _seconds: None)

    settled = execution._await_idle(resource, 4, attempts=3, delay=0.0)

    assert not execution._is_available(resource, settled, require_gpu=True)
    assert probes["n"] == 3, "should exhaust the settle window before giving up"


def test_sealed_bindings_are_staged_to_the_host_before_verification(tmp_path):
    """A newly written script must not cost a replicate just because nobody scp'd it.

    Before staging existed, a missing script failed preflight with "No such file or
    directory" and _claim_next then refused to re-run the replicate, because it already
    had a ResultBundle. Two spec pairs were burned that way in one session.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    (store_root / "blobs" / "sha256").mkdir(parents=True)
    workdir = tmp_path / "remote"
    workdir.mkdir()
    payload = b"print('hello')\n"
    digest = hashlib.sha256(payload).hexdigest()
    (store_root / "blobs" / "sha256" / digest).write_bytes(payload)

    resource = {"backend": "local", "workdir": str(workdir)}
    bindings = [
        {
            "execution_path": "train_new.py",
            "sha256": digest,
            "blob": f"blobs/sha256/{digest}",
        }
    ]

    staged = execution.stage_bindings(resource, bindings, store_root)
    assert staged[0]["action"] == "staged"
    assert (workdir / "train_new.py").read_bytes() == payload

    checks = execution.verify_bindings(resource, bindings)
    assert checks[0]["state"] == "verified"

    # Staging again is a no-op: the file is already the sealed bytes.
    assert execution.stage_bindings(resource, bindings, store_root)[0]["action"] == (
        "already_present"
    )


def test_staging_cannot_substitute_bytes_that_were_not_sealed(tmp_path):
    """Staging is addressed BY the recorded hash, so it can only place sealed content.

    If the blob for a binding is absent, staging must decline rather than invent
    something -- and verification must then fail the preflight, which is the outcome
    that carries the correct diagnostic.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    (store_root / "blobs" / "sha256").mkdir(parents=True)
    workdir = tmp_path / "remote"
    workdir.mkdir()
    digest = hashlib.sha256(b"never stored").hexdigest()

    resource = {"backend": "local", "workdir": str(workdir)}
    bindings = [{"execution_path": "ghost.py", "sha256": digest, "blob": f"blobs/sha256/{digest}"}]

    staged = execution.stage_bindings(resource, bindings, store_root)
    assert staged[0]["action"] == "unavailable"
    assert not (workdir / "ghost.py").exists()
    assert execution.verify_bindings(resource, bindings)[0]["state"] == "mismatch"


def test_staging_refuses_to_overwrite_content_this_store_never_sealed(tmp_path):
    """Staging fills a gap; it must not repair a discrepancy.

    A file that exists with UNKNOWN content is the only signal that the execution host
    is not what the operator believes. Silently replacing it would destroy that signal
    to make a run proceed -- the same class of mistake as an analysis adjustment that
    exists to make an inconvenient result disappear.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    (store_root / "blobs" / "sha256").mkdir(parents=True)
    workdir = tmp_path / "remote"
    workdir.mkdir()
    sealed = b"print('sealed')\n"
    digest = hashlib.sha256(sealed).hexdigest()
    (store_root / "blobs" / "sha256" / digest).write_bytes(sealed)

    foreign = b"print('who wrote this')\n"
    (workdir / "train.py").write_bytes(foreign)

    resource = {"backend": "local", "workdir": str(workdir)}
    bindings = [{"execution_path": "train.py", "sha256": digest, "blob": f"blobs/sha256/{digest}"}]

    staged = execution.stage_bindings(resource, bindings, store_root)
    assert staged[0]["action"] == "conflict"
    assert (workdir / "train.py").read_bytes() == foreign, "must not overwrite"
    assert execution.verify_bindings(resource, bindings)[0]["state"] == "mismatch"


def test_staging_advances_a_stale_version_of_our_own_sealed_file(tmp_path):
    """A previously sealed version of our own file is a known quantity, not a mutation.

    Refusing to advance it would make every code revision require a manual cleanup on
    the host, which is the friction that caused the missing-file problem in the first
    place.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    workdir = tmp_path / "remote"
    workdir.mkdir()

    old = b"print('v1')\n"
    new = b"print('v2')\n"
    old_digest = hashlib.sha256(old).hexdigest()
    new_digest = hashlib.sha256(new).hexdigest()
    (blobs / old_digest).write_bytes(old)  # v1 was sealed at some point
    (blobs / new_digest).write_bytes(new)
    (workdir / "train.py").write_bytes(old)

    resource = {"backend": "local", "workdir": str(workdir)}
    bindings = [
        {
            "execution_path": "train.py",
            "sha256": new_digest,
            "blob": f"blobs/sha256/{new_digest}",
        }
    ]

    staged = execution.stage_bindings(resource, bindings, store_root)
    assert staged[0]["action"] == "advanced"
    assert (workdir / "train.py").read_bytes() == new
    assert execution.verify_bindings(resource, bindings)[0]["state"] == "verified"


def test_a_replicate_with_existing_artifacts_is_never_claimed_twice(tmp_path):
    """Two workers must not both believe one replicate is free.

    The completed-set is derived from listing result_bundle records, which is a directory
    scan that can lag a concurrent worker. Artifacts are written earlier and are the
    durable trace. Observed on exp_confirm_attn_sink seed_42: claimed twice, then killed
    by a PermissionError on the read-only log its own first attempt had written.
    """
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(script, replicates=2, candidate_values=[0.9, 0.8])
    )
    manifest = SealingAuthority(store).seal(
        spec["id"], execution_config(script, data), approvals(spec)
    )
    service = ExecutionService(store)
    digest12 = manifest["digest"][:12]

    for replicate in manifest["payload"]["plan"]:
        directory = store.artifacts_dir / f"result_{digest12}_{replicate['replicate_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "control.stdout.log").write_text("already ran")

    with pytest.raises(NoPendingReplicate):
        service._claim_next(manifest, spec)

    assert not list(store.inflight_dir.glob("*.json")), (
        "a refused claim must not leave an inflight record behind"
    )


def _corpus_manifest(cache_dir, shards):
    return {
        "cache_dir": str(cache_dir),
        "files": [
            {"path": f"data/{name}", "bytes": size, "sha256": "0" * 64} for name, size in shards
        ],
    }


def _tokenizer_entries(cache_dir):
    entries = []
    for name in ("token_bytes.pt", "token_bytes.version", "tokenizer.pkl"):
        path = cache_dir / "tokenizer" / name
        payload = path.read_bytes()
        entries.append(
            {
                "path": f"tokenizer/{name}",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return entries


def test_an_extra_shard_on_the_host_fails_preflight(tmp_path):
    """prepare.py globs the data dir, so an unlisted shard silently changes training.

    The data_binding pins data_manifest.json -- a DESCRIPTION of the corpus -- while
    prepare.py builds its shard list with os.listdir(DATA_DIR). Dropping one parquet file
    into that directory changes the training data of every run while every binding check
    keeps passing and every record keeps claiming the same provenance.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    cache = tmp_path / "cache"
    data = cache / "data"
    data.mkdir(parents=True)
    for name in ("shard_00000.parquet", "shard_00001.parquet"):
        (data / name).write_bytes(b"x" * 100)

    described = _corpus_manifest(
        cache, [("shard_00000.parquet", 100), ("shard_00001.parquet", 100)]
    )
    blob_name = "corpus.json"
    (blobs / blob_name).write_text(json.dumps(described))
    bindings = [{"execution_path": "data_manifest.json", "blob": f"blobs/sha256/{blob_name}"}]
    resource = {"backend": "local", "workdir": str(tmp_path)}

    assert execution.verify_data_corpus(resource, bindings, store_root)[0]["state"] == ("verified")

    (data / "shard_00002.parquet").write_bytes(b"y" * 100)  # the silent corpus change
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "mismatch"
    assert row["extra"] == ["shard_00002.parquet"]


def test_a_truncated_or_missing_shard_also_fails_preflight(tmp_path):
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    cache = tmp_path / "cache"
    data = cache / "data"
    data.mkdir(parents=True)
    (data / "shard_00000.parquet").write_bytes(b"x" * 100)
    (data / "shard_00001.parquet").write_bytes(b"x" * 50)  # truncated

    described = _corpus_manifest(
        cache, [("shard_00000.parquet", 100), ("shard_00001.parquet", 100)]
    )
    (blobs / "corpus.json").write_text(json.dumps(described))
    bindings = [{"execution_path": "data_manifest.json", "blob": "blobs/sha256/corpus.json"}]
    resource = {"backend": "local", "workdir": str(tmp_path)}

    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "mismatch"
    assert row["resized"] == ["shard_00001.parquet"]

    (data / "shard_00001.parquet").unlink()
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["missing"] == ["shard_00001.parquet"]


def test_tokenizer_artifacts_are_verified_by_size_and_sha256(tmp_path):
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    cache = tmp_path / "cache"
    data = cache / "data"
    tokenizer = cache / "tokenizer"
    data.mkdir(parents=True)
    tokenizer.mkdir()
    (data / "shard_00000.parquet").write_bytes(b"shard")
    (tokenizer / "token_bytes.pt").write_bytes(b"correct-table")
    (tokenizer / "token_bytes.version").write_text("2\n", encoding="utf-8")
    (tokenizer / "tokenizer.pkl").write_bytes(b"tokenizer")

    described = _corpus_manifest(cache, [("shard_00000.parquet", 5)])
    described["files"].extend(_tokenizer_entries(cache))
    described["files"].append({"path": "notes/operator-only.txt"})
    (blobs / "corpus.json").write_text(json.dumps(described), encoding="utf-8")
    bindings = [{"execution_path": "data_manifest.json", "blob": "blobs/sha256/corpus.json"}]
    resource = {"backend": "local", "workdir": str(tmp_path)}

    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "verified"
    assert row["tokenizer_expected"] == row["tokenizer_found"] == 3

    # Same byte count is not enough for tokenizer state: unlike the ~1 GB corpus,
    # these artifacts are small enough to hash before every launch.
    (tokenizer / "token_bytes.pt").write_bytes(b"wrong--table-")
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "mismatch"
    assert row["tokenizer_hash_mismatch"] == ["tokenizer/token_bytes.pt"]

    (tokenizer / "token_bytes.pt").write_bytes(b"short")
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["tokenizer_resized"] == ["tokenizer/token_bytes.pt"]

    (tokenizer / "token_bytes.pt").write_bytes(b"correct-table")
    (tokenizer / "token_bytes.version").unlink()
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["tokenizer_missing"] == ["tokenizer/token_bytes.version"]


def test_remote_tokenizer_artifacts_are_verified_with_stat_and_sha256(tmp_path, monkeypatch):
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    cache_dir = "/srv/karpathy cache"
    payloads = {
        "tokenizer/token_bytes.pt": b"table",
        "tokenizer/token_bytes.version": b"2\n",
        "tokenizer/tokenizer.pkl": b"pickle",
    }
    described = {
        "cache_dir": cache_dir,
        "files": [
            {
                "path": "data/shard_00000.parquet",
                "bytes": 5,
                "sha256": "0" * 64,
            },
            *[
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                for relative, payload in payloads.items()
            ],
        ],
    }
    (blobs / "corpus.json").write_text(json.dumps(described), encoding="utf-8")
    bindings = [{"execution_path": "data_manifest.json", "blob": "blobs/sha256/corpus.json"}]
    resource = {
        "id": "remote",
        "backend": "ssh",
        "ssh_argv": ["ssh", "host"],
        "workdir": "/srv/work",
    }

    def inspect(_resource, command, timeout=15):
        del timeout
        if "*.parquet" in command:
            return SimpleNamespace(
                returncode=0,
                stdout="shard_00000.parquet 5\n",
                stderr="",
            )
        assert "stat -c" in command and "sha256sum" in command
        lines = []
        for relative, payload in payloads.items():
            remote_path = f"{cache_dir}/{relative}"
            lines.append(f"{len(payload)} {remote_path}")
            lines.append(f"{hashlib.sha256(payload).hexdigest()}  {remote_path}")
        return SimpleNamespace(returncode=0, stdout="\n".join(lines) + "\n", stderr="")

    monkeypatch.setattr(execution, "_run_control", inspect)
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "verified"
    assert row["tokenizer_found"] == 3

    payloads["tokenizer/token_bytes.pt"] = b"tAble"
    row = execution.verify_data_corpus(resource, bindings, store_root)[0]
    assert row["state"] == "mismatch"
    assert row["tokenizer_hash_mismatch"] == ["tokenizer/token_bytes.pt"]


def test_tokenizer_mismatch_stops_execution_at_preflight(tmp_path):
    script, data_manifest = make_project(tmp_path)
    cache = tmp_path / "cache"
    data_dir = cache / "data"
    tokenizer_dir = cache / "tokenizer"
    data_dir.mkdir(parents=True)
    tokenizer_dir.mkdir()
    (data_dir / "shard_00000.parquet").write_bytes(b"shard")
    (tokenizer_dir / "token_bytes.pt").write_bytes(b"correct-table")
    (tokenizer_dir / "token_bytes.version").write_text("2\n", encoding="utf-8")
    (tokenizer_dir / "tokenizer.pkl").write_bytes(b"tokenizer")
    described = _corpus_manifest(cache, [("shard_00000.parquet", 5)])
    described["files"].extend(_tokenizer_entries(cache))
    data_manifest.write_text(json.dumps(described), encoding="utf-8")

    store = Store(tmp_path / "state")
    proposal = make_spec(
        script, stage="pilot", replicates=1, candidate_values=[0.9], sota_eligible=False
    )
    spec = ResearchEngine(store).create(proposal)
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data_manifest))

    # Keep the size stable so only the tokenizer digest can catch the mutation.
    (tokenizer_dir / "token_bytes.pt").write_bytes(b"wrong--table-")
    result = ExecutionService(store).execute_next(manifest["id"])
    assert result["payload"]["status"] == "preflight_failed"
    assert result["payload"]["arms"] == []
    assert "tokenizer" in result["payload"]["failure"]


def test_staging_can_advance_a_file_it_previously_placed_read_only(tmp_path):
    """Staging must be able to update its own output, not just create it once.

    Blobs live in a content-addressed store and are kept read-only; copying propagates
    that mode, so a file staging placed earlier is UNWRITABLE. Advancing it to a new
    revision then fails with "Permission denied" -- observed in production as
    `scp: dest open ".../train_ngw.py": Permission denied` after a script was edited and
    re-sealed, which failed the preflight of two screens.

    The earlier advance test missed this because it created the destination itself with
    write_bytes (mode 644), never with the read-only mode staging actually produces.
    """
    from autoresearch import execution

    store_root = tmp_path / "state"
    blobs = store_root / "blobs" / "sha256"
    blobs.mkdir(parents=True)
    workdir = tmp_path / "remote"
    workdir.mkdir()

    v1, v2 = b"print('v1')\n", b"print('v2')\n"
    d1 = hashlib.sha256(v1).hexdigest()
    d2 = hashlib.sha256(v2).hexdigest()
    for digest, payload in ((d1, v1), (d2, v2)):
        blob = blobs / digest
        blob.write_bytes(payload)
        blob.chmod(0o444)  # the store keeps blobs read-only

    resource = {"backend": "local", "workdir": str(workdir)}
    b1 = [{"execution_path": "t.py", "sha256": d1, "blob": f"blobs/sha256/{d1}"}]
    b2 = [{"execution_path": "t.py", "sha256": d2, "blob": f"blobs/sha256/{d2}"}]

    assert execution.stage_bindings(resource, b1, store_root)[0]["action"] == "staged"
    assert execution.verify_bindings(resource, b1)[0]["state"] == "verified"

    row = execution.stage_bindings(resource, b2, store_root)[0]
    assert row["action"] == "advanced", row
    assert (workdir / "t.py").read_bytes() == v2
    assert execution.verify_bindings(resource, b2)[0]["state"] == "verified"


def test_a_contended_arm_is_rejected_against_its_own_argv_history():
    """Pairing cancels placement, not time.

    Arms of a paired replicate run sequentially, minutes apart, so a neighbour arriving
    between them hits one and not the other. Observed on exp_cxb_turn_1x2: the control got
    803 steps against its clean 975 while the candidate got 1936 against its clean 1845.
    The pair reported the OPPOSITE sign to the hypothesis under test, twice, because it
    compared two moments rather than two configurations. max_compute_processes stayed at 1
    throughout -- host contention leaves it there, so the existing co-tenancy gate is blind
    to this.
    """
    from autoresearch.evidence import THROUGHPUT_FLOOR, throughput_key, throughput_reason

    argv = [".venv/bin/python", "train.py", "--arm=control"]
    placement = {"host_id": "host-a", "gpu": 0, "gpu_uuid": "GPU-0"}
    key = throughput_key(argv, placement)

    # No baseline -> admissible. New configurations are never punished for lacking history.
    assert throughput_reason("control", 803, argv, None) is None
    assert throughput_reason("control", 803, argv, {}) is None

    # Established clean median of 975: 803 steps is 82% and must be rejected.
    reason = throughput_reason("control", 803, argv, {key: 975.0}, placement)
    assert reason and "host contention" in reason and "82%" in reason

    # A healthy arm at the same baseline passes.
    assert throughput_reason("control", 970, argv, {key: 975.0}, placement) is None
    assert 970 / 975.0 > THROUGHPUT_FLOOR

    # Missing or non-numeric step counts are handled by the existing minimum-steps rule.
    assert throughput_reason("control", None, argv, {key: 975.0}) is None
    assert throughput_reason("control", True, argv, {key: 975.0}) is None


def test_the_baseline_is_keyed_on_argv_so_a_slower_config_is_judged_against_itself():
    """A 600s arm legitimately runs ~1845 steps; a half-batch arm ~1950.

    Keying on the FULL argv is what stops the gate rejecting every deliberately-different
    configuration as contended -- which would have made the whole v(s) sweep inadmissible.
    """
    from autoresearch.evidence import throughput_key, throughput_reason

    fast = [".venv/bin/python", "train.py", "--arm=control"]
    slow = fast + ["--time-budget=600"]
    placement = {"host_id": "host-a", "gpu": 0, "gpu_uuid": "GPU-0"}
    baselines = {
        throughput_key(fast, placement): 975.0,
        throughput_key(slow, placement): 1845.0,
    }

    # 1845 steps is 189% of the fast baseline and exactly right for its own.
    assert throughput_reason("control", 1845, slow, baselines, placement) is None
    # And the slow config is still policed against ITS own history.
    assert throughput_reason("control", 1400, slow, baselines, placement) is not None


def _arm(name, steps, argv, maxproc=1):
    return {
        "name": name,
        "payload_argv": argv,
        "metrics": {"val_bpb": 0.96, "num_steps": steps},
        "telemetry": {"sample_count": 3, "max_compute_processes": maxproc},
        "status": "completed",
    }
