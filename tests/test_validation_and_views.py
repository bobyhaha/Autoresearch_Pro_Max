from __future__ import annotations

from pathlib import Path

import pytest
from conftest import approvals, execution_config, make_project, make_spec

from autoresearch.evidence import EvidenceEngine
from autoresearch.execution import ExecutionService
from autoresearch.knowledge import KnowledgeEngine
from autoresearch.records import RecordError, make_record
from autoresearch.research import ResearchEngine
from autoresearch.sealing import SealingAuthority
from autoresearch.store import Store


def test_unresolved_inflight_claim_warns_without_freezing_the_registry(tmp_path):
    """An orphan claim is a live-process fact, not a corrupt record.

    It must be visible and it must still block replay of its own replicate, but
    it must not stop synthesis for every unrelated spec in the registry.
    """
    store = Store(tmp_path / "state")
    store.init()
    store.write_operational(
        store.inflight_dir / "orphan.json",
        {"spec_id": "exp_missing", "replicate_id": "seed_1", "owner_pid": 999999},
    )
    result = store.validate()
    assert result["valid"] is True
    assert result["inflight"] == 1
    assert any("unresolved" in warning for warning in result["warnings"])
    KnowledgeEngine(store).synthesize()
    assert [row["token"] for row in store.claims()] == ["orphan"]
    released = store.release_claim("orphan")
    assert released["spec_id"] == "exp_missing"
    assert store.validate()["inflight"] == 0


def test_release_claim_rejects_path_traversal_without_touching_other_state(tmp_path):
    store = Store(tmp_path / "state")
    store.init()
    sentinel = store.root / "sentinel.json"
    sentinel.write_text('{"immutable":"keep"}\n', encoding="utf-8")

    with pytest.raises(RecordError, match="path-safe"):
        store.release_claim("../../sentinel")

    assert sentinel.read_text(encoding="utf-8") == '{"immutable":"keep"}\n'


def test_views_are_rebuildable_and_do_not_become_evidence(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(
            script,
            stage="pilot",
            replicates=1,
            candidate_values=[0.9],
            sota_eligible=False,
        )
    )
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    EvidenceEngine(store).judge(result["id"])
    first = KnowledgeEngine(store).synthesize()
    (store.views_dir / "BELIEFS.md").write_text("corrupt view", encoding="utf-8")
    second = KnowledgeEngine(store).synthesize()
    assert first["beliefs"] == second["beliefs"]
    assert (store.views_dir / "BELIEFS.md").read_text().startswith("# Beliefs")
    assert store.validate()["counts"]["evidence_decision"] == 1


def test_validator_detects_artifact_tampering(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(
            script,
            stage="pilot",
            replicates=1,
            candidate_values=[0.9],
            sota_eligible=False,
        )
    )
    manifest = SealingAuthority(store).seal(spec["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    stdout = result["payload"]["arms"][0]["artifacts"]["stdout"]
    assert not Path(stdout).is_absolute(), "artifact paths must be root-relative"
    artifact = store.root / stdout
    artifact.chmod(0o644)
    artifact.write_text("tampered\n", encoding="utf-8")
    validation = store.validate()
    assert validation["valid"] is False
    assert any("corrupt stdout artifact" in error for error in validation["errors"])


def test_validator_cross_checks_evidence_spec_against_result(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    first = ResearchEngine(store).create(
        make_spec(
            script,
            spec_id="exp_first",
            stage="pilot",
            replicates=1,
            candidate_values=[0.9],
            sota_eligible=False,
        )
    )
    second = ResearchEngine(store).create(
        make_spec(
            script,
            spec_id="exp_second",
            stage="pilot",
            replicates=1,
            candidate_values=[0.8],
            sota_eligible=False,
        )
    )
    manifest = SealingAuthority(store).seal(first["id"], execution_config(script, data))
    result = ExecutionService(store).execute_next(manifest["id"])
    store.put(
        make_record(
            "evidence_decision",
            "evidence_wrong_spec",
            {
                "result_id": result["id"],
                "result_digest": result["digest"],
                "spec_id": second["id"],
                "stage": "pilot",
                "measurement_verdict": "valid",
                "claim_status": "ineligible",
                "reasons": ["constructed adversarial record"],
                "measurements": {},
                "policy_version": "adversarial-v1",
            },
        )
    )
    validation = store.validate()
    assert validation["valid"] is False
    assert any("spec id disagrees with its result" in error for error in validation["errors"])


def test_paper_registration_binds_content_and_coverage(tmp_path):
    script, _ = make_project(tmp_path)
    store = Store(tmp_path / "state")
    spec = ResearchEngine(store).create(
        make_spec(
            script,
            stage="pilot",
            replicates=1,
            candidate_values=[0.9],
            sota_eligible=False,
        )
    )
    paper_source = tmp_path / "paper.pdf"
    paper_source.write_bytes(b"small immutable paper artifact")
    paper = ResearchEngine(store).register_paper(
        {
            "id": "paper_demo_001",
            "title": "Demo paper",
            "path": str(paper_source),
            "content_sha256": "",
            "spec_ids": [spec["id"]],
            "evidence_ids": [],
        }
    )
    assert paper["payload"]["content_sha256"]
    assert (store.root / paper["payload"]["blob"]).is_file()
    assert store.validate()["valid"] is True
    paper_source.write_bytes(b"changed")
    assert store.validate()["valid"] is True


def test_five_round_paper_gate_blocks_a_sixth_confirmation(tmp_path):
    script, data = make_project(tmp_path)
    store = Store(tmp_path / "state")
    research = ResearchEngine(store)
    sealing = SealingAuthority(store)
    execution = ExecutionService(store)
    evidence = EvidenceEngine(store)
    completed_specs = []
    evidence_ids = []
    for index in range(5):
        spec = research.create(
            make_spec(
                script,
                spec_id=f"exp_round_{index + 1}",
                replicates=1,
                candidate_values=[0.9],
                sota_eligible=False,
            )
        )
        manifest = sealing.seal(spec["id"], execution_config(script, data), approvals(spec))
        result = execution.execute_next(manifest["id"])
        decision = evidence.judge(result["id"])
        completed_specs.append(spec["id"])
        evidence_ids.append(decision["id"])

    sixth = research.create(
        make_spec(
            script,
            spec_id="exp_round_6",
            replicates=1,
            candidate_values=[0.9],
            sota_eligible=False,
        )
    )
    with pytest.raises(RecordError, match="paper gate"):
        sealing.seal(sixth["id"], execution_config(script, data), approvals(sixth))

    paper_source = tmp_path / "block_1.pdf"
    paper_source.write_bytes(b"block one analysis and next experiments")
    research.register_paper(
        {
            "id": "paper_block_1",
            "title": "Block one",
            "path": str(paper_source),
            "content_sha256": "",
            "spec_ids": completed_specs,
            "evidence_ids": evidence_ids,
        }
    )
    assert sealing.seal(sixth["id"], execution_config(script, data), approvals(sixth))
