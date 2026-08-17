from __future__ import annotations

import json
from copy import deepcopy

import pytest

from autoresearch.cli import main
from autoresearch.records import RecordError
from autoresearch.science import ScientificLibrary
from autoresearch.store import Store
from autoresearch.workflow import V2Workflow

NOW = "2026-08-16T12:00:00Z"


def agenda():
    return {
        "id": "agenda_memory",
        "name": "Memory search",
        "objective": "Find mechanisms that transfer to a short decoder run.",
        "scope": {"budget": "short"},
        "confidence_policy_version": "confidence-v1",
        "topics": [
            {
                "id": "topic_memory",
                "question": "When does memory help language models?",
                "queries": ["language model memory retrieval ablation"],
                "keywords": ["memory", "retrieval"],
                "refresh_days": 14,
                "minimum_sources": 1,
                "minimum_claims": 1,
            }
        ],
    }


def source(source_id: str, *, work_key: str, tier: str = "top", topic: str | None = None):
    return {
        "id": source_id,
        "work_key": work_key,
        "title": f"Study {source_id}",
        "authors": ["Ada Researcher"],
        "year": 2025,
        "venue": {
            "name": "International Conference on Machine Learning",
            "tier": tier,
            "peer_reviewed": "yes" if tier != "preprint" else "no",
        },
        "publication_type": "proceedings-article",
        "identifiers": {"doi": f"10.1/{source_id}"},
        "urls": [f"https://example.test/{source_id}"],
        "abstract": "A controlled result.",
        "topics": [topic] if topic else [],
        "retrieval": {"provider": "manual", "retrieved_at": NOW},
        "content": {"status": "abstract_only"},
        "citation_count": 0,
    }


def claim(
    claim_id: str,
    source_id: str,
    *,
    stance: str = "supports",
    reproduction: str = "not_attempted",
    topic: str | None = None,
):
    return {
        "id": claim_id,
        "belief_key": "belief_memory_improves_bpb",
        "statement": "A bounded memory mechanism improves validation BPB.",
        "claim_type": "empirical",
        "origin": "literature",
        "stance": stance,
        "scope": {"model": "decoder", "budget": "short"},
        "source_ids": [source_id],
        "evidence_ids": [],
        "derived_from_claim_ids": [],
        "topics": [topic] if topic else [],
        "locators": [{"source_id": source_id, "location": "section 4, table 2"}],
        "evidence": {
            "study_design": "controlled_benchmark",
            "artifact_status": "none",
            "reproduction_status": reproduction,
            "directness": "direct",
            "scope_match": "exact",
            "risk_of_bias": "low",
            "metrics": [{"name": "val_bpb", "result": "improved by 0.002"}],
        },
        "assessment": {
            "assessor": "agent:test",
            "assessed_at": NOW,
            "rationale": "The table directly compares a controlled intervention.",
            "confidence_policy_version": "confidence-v1",
        },
    }


def mechanism(claim_id: str):
    return {
        "id": "mech_memory_retrieval",
        "name": "Memory improves retrieval",
        "statement": "Memory exposes useful prior tokens without excessive compute.",
        "scope": {"model": "decoder", "budget": "short"},
        "nodes": [
            {"id": "memory", "label": "bounded memory"},
            {"id": "retrieval", "label": "useful retrieval"},
            {"id": "bpb", "label": "lower validation BPB"},
        ],
        "edges": [
            {
                "from": "memory",
                "to": "retrieval",
                "relation": "increases access to relevant history",
                "claim_ids": [claim_id],
            },
            {
                "from": "retrieval",
                "to": "bpb",
                "relation": "improves next-token prediction",
                "claim_ids": [claim_id],
            },
        ],
        "source_claim_ids": [claim_id],
        "assumptions": [],
        "alternatives": ["The change only alters optimization noise."],
        "predictions": ["The effect is larger when relevant context is distant."],
        "falsifiers": ["Matched compute removes the BPB advantage."],
    }


def hypothesis(claim_id: str):
    return {
        "id": "hyp_memory_short_budget",
        "belief_key": "belief_memory_improves_bpb",
        "statement": "A small bounded memory lowers BPB under the fixed short budget.",
        "rationale": "The cited mechanism should transfer when its overhead is bounded.",
        "scope": {"model": "decoder", "budget": "short"},
        "topics": ["topic_memory"],
        "mechanism_ids": ["mech_memory_retrieval"],
        "claim_ids": [claim_id],
        "prediction": {
            "metric": "val_bpb",
            "direction": "decrease",
            "statement": "Candidate-control val_bpb is below -0.000426.",
            "minimum_effect": 0.000426,
        },
        "falsifiers": ["The paired delta is at or above -0.000426."],
        "intervention": {
            "summary": "add a bounded memory path",
            "subsystem": "model.memory",
            "mutable_code_paths": ["train.py"],
            "diagnostics": ["memory hit rate", "steps completed"],
        },
        "competing_hypothesis_ids": [],
        "proposed_by": "agent:test",
    }


def test_confidence_is_cheap_transparent_and_not_venue_only(tmp_path):
    library = ScientificLibrary(Store(tmp_path / "state"), now=lambda: NOW)
    library.register_source(source("lit_top", work_key="work_top"))
    library.register_claim(claim("claim_top", "lit_top"))

    snapshot = library.synthesize()
    belief = snapshot["beliefs"][0]
    contribution = belief["contributions"][0]
    assert 0.5 < belief["confidence"] < 0.8
    assert contribution["components"]["venue"] == pytest.approx(0.9)
    assert contribution["components"]["reproduction"] == pytest.approx(0.42)
    assert contribution["venue_is_not_truth"] is True
    assert (tmp_path / "state" / "views" / "SCIENCE.json").is_file()


def test_independent_contradiction_is_contested_and_same_work_is_deduplicated(tmp_path):
    library = ScientificLibrary(Store(tmp_path / "state"), now=lambda: NOW)
    library.register_source(source("lit_a1", work_key="work_a"))
    library.register_source(source("lit_a2", work_key="work_a"))
    library.register_source(source("lit_b", work_key="work_b", tier="peer_reviewed"))
    library.register_claim(claim("claim_a1", "lit_a1"))
    stronger = claim("claim_a2", "lit_a2", reproduction="independent_success")
    library.register_claim(stronger)
    library.register_claim(
        claim("claim_b", "lit_b", stance="opposes", reproduction="independent_success")
    )

    belief = library.synthesize()["beliefs"][0]
    assert belief["independent_evidence_units"] == 2
    assert belief["state"] == "contested"
    assert belief["support_weight"] > 0
    assert belief["opposition_weight"] > 0


def test_mechanism_and_hypothesis_are_claim_bound_and_feed_experiment_chain(tmp_path):
    store = Store(tmp_path / "state")
    library = ScientificLibrary(store, now=lambda: NOW)
    library.register_agenda(agenda())
    library.register_source(source("lit_a", work_key="work_a"))
    library.register_claim(claim("claim_a", "lit_a"))
    library.register_mechanism(mechanism("claim_a"))
    library.register_hypothesis(hypothesis("claim_a"))

    assert store.validate()["valid"] is True
    with pytest.raises(RecordError, match="not research-ready"):
        V2Workflow(store)._scientific_context(["hyp_memory_short_budget"])
    context = V2Workflow(store)._scientific_context(
        ["hyp_memory_short_budget"], allow_weak_science=True
    )
    assert context is not None
    assert context["hypothesis"]["statement"].startswith("A small bounded memory")
    assert context["mechanism"]["chain"][0].startswith("bounded memory ->")
    projected = library.synthesize()
    assert projected["hypotheses"][0]["experiment_status"] == "untested"
    assert projected["idea_queue"][0]["hypothesis_id"] == "hyp_memory_short_budget"


def test_research_ready_hypothesis_requires_search_fulltext_and_independence(tmp_path):
    store = Store(tmp_path / "state")
    library = ScientificLibrary(store, now=lambda: NOW)
    library.register_agenda(agenda())
    paper = tmp_path / "paper.txt"
    paper.write_text("full methods and results", encoding="utf-8")
    source_ids = []
    claim_ids = []
    for suffix in ("a", "b"):
        raw_source = source(f"lit_{suffix}", work_key=f"work_{suffix}", topic="topic_memory")
        raw_source["retrieval"]["search_id"] = "search_memory_manual"
        raw_source["content"] = {
            "status": "fulltext_snapshot",
            "path": str(paper),
        }
        library.register_source(raw_source)
        library.register_claim(
            claim(
                f"claim_{suffix}",
                f"lit_{suffix}",
                reproduction="independent_success",
                topic="topic_memory",
            )
        )
        source_ids.append(f"lit_{suffix}")
        claim_ids.append(f"claim_{suffix}")
    library.register_search(
        {
            "id": "search_memory_manual",
            "provider": "manual",
            "query": "language model memory retrieval ablation",
            "searched_at": NOW,
            "agenda_id": "agenda_memory",
            "topic_id": "topic_memory",
            "filters": {},
            "result_source_ids": source_ids,
            "raw_result_count": 2,
        }
    )
    mech = mechanism("claim_a")
    mech["source_claim_ids"] = claim_ids
    for edge in mech["edges"]:
        edge["claim_ids"] = claim_ids
    library.register_mechanism(mech)
    hyp = hypothesis("claim_a")
    hyp["claim_ids"] = claim_ids
    library.register_hypothesis(hyp)

    context = V2Workflow(store)._scientific_context(["hyp_memory_short_budget"])
    assert context is not None
    projected = library.synthesize()
    assert projected["hypotheses"][0]["foundation_evidence_units"] == 2
    assert projected["research_gaps"][0]["search_due"] is False
    assert projected["research_gaps"][0]["analysis_due"] is False


def test_mechanism_cannot_treat_opposition_as_support(tmp_path):
    store = Store(tmp_path / "state")
    library = ScientificLibrary(store, now=lambda: NOW)
    library.register_source(source("lit_a", work_key="work_a"))
    library.register_claim(claim("claim_a", "lit_a", stance="opposes"))
    library.register_mechanism(mechanism("claim_a"))
    validation = store.validate()
    assert validation["valid"] is False
    assert any("opposing claim" in error for error in validation["errors"])


def test_agenda_separates_search_due_from_claim_analysis_due(tmp_path):
    library = ScientificLibrary(Store(tmp_path / "state"), now=lambda: NOW)
    library.register_agenda(agenda())
    initial = library.synthesize()["research_gaps"][0]
    assert initial["search_due"] is True
    assert library.synthesize()["research_tasks"][0]["kind"] == "search_literature"

    # A source removes the sparse-source reason, but claim extraction and full
    # text inspection remain CPU analysis work rather than repeated web queries.
    library.register_source(source("lit_a", work_key="work_a", topic="topic_memory"))
    gap = library.synthesize()["research_gaps"][0]
    assert gap["analysis_due"] is True
    assert "insufficient_extracted_claims" in gap["reasons"]


def test_openalex_search_preserves_query_and_reconstructs_abstract(tmp_path):
    library = ScientificLibrary(Store(tmp_path / "state"), now=lambda: NOW)

    def fake_fetch(url: str):
        assert "search=memory+retrieval" in url
        return {
            "results": [
                {
                    "id": "https://openalex.org/W1",
                    "doi": "https://doi.org/10.1/test",
                    "title": "Memory Test",
                    "publication_year": 2025,
                    "primary_location": {
                        "landing_page_url": "https://example.test/paper",
                        "source": {"display_name": "ICML"},
                    },
                    "type": "proceedings-article",
                    "authorships": [{"author": {"display_name": "A. Author"}}],
                    "abstract_inverted_index": {"Memory": [0], "helps": [1]},
                    "cited_by_count": 3,
                }
            ]
        }

    outcome = library.search_openalex("memory retrieval", fetch=fake_fetch)
    assert outcome["search"]["payload"]["query"] == "memory retrieval"
    assert outcome["sources"][0]["payload"]["abstract"] == "Memory helps"
    assert outcome["sources"][0]["payload"]["venue"]["tier"] == "top"
    assert library.store.validate()["valid"] is True


def test_scientific_claim_requires_real_provenance(tmp_path):
    declaration = claim("claim_bad", "lit_missing")
    declaration["source_ids"] = []
    with pytest.raises(RecordError, match="requires literature"):
        ScientificLibrary(Store(tmp_path / "state")).register_claim(declaration)

    declaration = deepcopy(claim("claim_bad", "lit_missing"))
    declaration["assessment"]["confidence_policy_version"] = "confidence-v999"
    with pytest.raises(RecordError, match="confidence-v1"):
        ScientificLibrary(Store(tmp_path / "state")).register_claim(declaration)


def test_science_cli_registers_and_rebuilds_library(tmp_path, capsys):
    root = tmp_path / "state"
    declarations = {
        "agenda": agenda(),
        "literature-source": source("lit_cli", work_key="work_cli", topic="topic_memory"),
        "scientific-claim": claim("claim_cli", "lit_cli", topic="topic_memory"),
        "mechanism": mechanism("claim_cli"),
        "hypothesis": hypothesis("claim_cli"),
    }
    for command, declaration in declarations.items():
        path = tmp_path / f"{command}.json"
        path.write_text(json.dumps(declaration), encoding="utf-8")
        assert main(["--root", str(root), command, str(path)]) == 0
    assert main(["--root", str(root), "science"]) == 0
    output = capsys.readouterr().out
    assert '"hypotheses": 1' in output
    assert (root / "views" / "IDEA_QUEUE.json").is_file()
    assert (root / "views" / "RESEARCH_TASKS.json").is_file()
