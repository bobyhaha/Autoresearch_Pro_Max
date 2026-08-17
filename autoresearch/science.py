"""A provenance-first scientific library for literature and experimental beliefs.

The library deliberately separates three things that are often conflated:

* retrieval records what was searched and what was found;
* claims record an attributed, scoped interpretation of a source;
* confidence is a rebuildable projection over independent evidence.

Venue prestige is one small confidence component.  It can never, by itself,
turn an abstract or a prestigious paper into a high-confidence belief.
"""

from __future__ import annotations

import copy
import json
import math
import mimetypes
import re
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .records import (
    ConflictError,
    RecordError,
    canonical_json,
    make_record,
    rule_matches,
    sha256_object,
    utc_now,
)
from .store import Store

CONFIDENCE_POLICY_VERSION = "confidence-v1"
OPENALEX_API = "https://api.openalex.org/works"

# This is intentionally a small, versioned classifier, not a universal claim
# that venue rank transfers across fields.  Unrecognized venues remain unknown.
TOP_ML_VENUES_V1 = {
    "aaai",
    "acl",
    "aaai conference on artificial intelligence",
    "annual meeting of the association for computational linguistics",
    "conference on computer vision and pattern recognition",
    "conference on empirical methods in natural language processing",
    "cvpr",
    "eccv",
    "emnlp",
    "european conference on computer vision",
    "international conference on computer vision",
    "international conference on learning representations",
    "international conference on machine learning",
    "international conference on machine learning and systems",
    "iccv",
    "iclr",
    "icml",
    "mlsys",
    "neural information processing systems",
    "neurips",
}

DESIGN_SCORES = {
    "meta_analysis": 0.95,
    "randomized_controlled": 0.92,
    "controlled_benchmark": 0.82,
    "ablation": 0.76,
    "observational": 0.58,
    "theoretical": 0.58,
    "case_study": 0.48,
    "anecdotal": 0.22,
    "unknown": 0.40,
}
VENUE_SCORES = {
    "top": 0.90,
    "selective": 0.82,
    "peer_reviewed": 0.72,
    "workshop": 0.60,
    "preprint": 0.46,
    "unknown": 0.42,
}
ARTIFACT_SCORES = {
    "verified": 0.95,
    "available": 0.76,
    "partial": 0.58,
    "none": 0.35,
    "unknown": 0.42,
}
REPRODUCTION_SCORES = {
    "independent_success": 0.98,
    "independent_failure": 0.05,
    "author_only": 0.65,
    "not_attempted": 0.42,
    "unknown": 0.42,
}
DIRECTNESS_SCORES = {"direct": 0.95, "indirect": 0.62, "speculative": 0.30}
SCOPE_SCORES = {"exact": 1.0, "close": 0.82, "partial": 0.58, "distant": 0.30, "unknown": 0.42}
BIAS_SCORES = {"low": 0.92, "medium": 0.64, "high": 0.28, "unknown": 0.46}
COMPONENT_WEIGHTS = {
    "venue": 0.06,
    "peer_review": 0.02,
    "study_design": 0.18,
    "content_depth": 0.08,
    "artifact": 0.12,
    "reproduction": 0.22,
    "directness": 0.14,
    "scope_match": 0.12,
    "risk_of_bias": 0.06,
}
PEER_REVIEW_SCORES = {"yes": 0.80, "no": 0.38, "unknown": 0.45}
CONTENT_SCORES = {"fulltext_snapshot": 0.95, "abstract_only": 0.55, "metadata_only": 0.25}


def _slug(text: str, *, fallback: str = "item") -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (value or fallback)[:80]


def _parse_time(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RecordError(f"invalid ISO timestamp: {value}") from exc


def _latest_decisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    latest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        result_id = str(row["payload"]["result_id"])
        previous = latest.get(result_id)
        if previous is None or str(row["created_at"]) > str(previous["created_at"]):
            latest[result_id] = row
    return latest


def _confidence_level(score: float) -> str:
    if score < 0.20:
        return "very_low"
    if score < 0.40:
        return "low"
    if score < 0.60:
        return "uncertain"
    if score < 0.75:
        return "moderate"
    if score < 0.90:
        return "high"
    return "very_high"


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


def _venue_classification(name: str, publication_type: str) -> dict[str, str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()
    is_top = any(venue in normalized or normalized in venue for venue in TOP_ML_VENUES_V1)
    if is_top:
        return {"name": name or "unknown", "tier": "top", "peer_reviewed": "yes"}
    if publication_type in {"article", "proceedings-article"} and name:
        # Bibliographic metadata alone does not prove the review process.
        return {"name": name, "tier": "unknown", "peer_reviewed": "unknown"}
    return {"name": name or "unknown", "tier": "preprint", "peer_reviewed": "no"}


def _abstract(inverted: Any) -> str:
    if not isinstance(inverted, Mapping):
        return ""
    positioned: list[tuple[int, str]] = []
    for word, raw_positions in inverted.items():
        if not isinstance(word, str) or not isinstance(raw_positions, list):
            continue
        for position in raw_positions:
            if isinstance(position, int) and not isinstance(position, bool):
                positioned.append((position, word))
    return " ".join(word for _, word in sorted(positioned))


def _fetch_json(url: str, *, timeout: float = 30.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "simplify-autoresearch-v2/0.3 literature-provenance",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecordError(f"literature provider request failed: {exc}") from exc


class ScientificLibrary:
    """Register immutable scientific objects and rebuild epistemic projections."""

    def __init__(self, store: Store, *, now: Callable[[], str] = utc_now) -> None:
        self.store = store
        self._now = now

    def _register(self, kind: str, declaration: Mapping[str, Any]) -> dict[str, Any]:
        raw = copy.deepcopy(dict(declaration))
        record_id = str(raw.pop("id", ""))
        if not record_id:
            raise RecordError(f"{kind} declaration requires id")
        path = self.store.record_path(kind, record_id)
        record = make_record(kind, record_id, raw)
        if path.exists():
            existing = self.store.get(kind, record_id)
            if existing["payload"] == record["payload"]:
                return existing
            raise ConflictError(f"{kind} {record_id} already exists with other content")
        return self.store.put(record)

    def register_agenda(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return self._register("research_agenda", declaration)

    def register_claim(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return self._register("scientific_claim", declaration)

    def register_search(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return self._register("literature_search", declaration)

    def register_mechanism(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return self._register("scientific_mechanism", declaration)

    def register_hypothesis(self, declaration: Mapping[str, Any]) -> dict[str, Any]:
        return self._register("scientific_hypothesis", declaration)

    def register_source(
        self, declaration: Mapping[str, Any], *, base: Path | None = None
    ) -> dict[str, Any]:
        raw = copy.deepcopy(dict(declaration))
        content = dict(raw.get("content", {}))
        source_path = content.pop("path", None)
        if source_path is not None:
            path = Path(str(source_path))
            if base is not None and not path.is_absolute():
                path = base / path
            if not path.is_file():
                raise RecordError(f"literature full text does not exist: {path}")
            digest, blob = self.store.add_blob(path)
            content.update(
                {
                    "status": "fulltext_snapshot",
                    "sha256": digest,
                    "blob": blob,
                    "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                    "source_name": path.name,
                }
            )
        raw["content"] = content
        if "work_key" not in raw:
            identity = raw.get("identifiers") or {"title": raw.get("title")}
            raw["work_key"] = f"work_{sha256_object(identity)[:20]}"
        return self._register("literature_source", raw)

    def search_openalex(
        self,
        query: str,
        *,
        limit: int = 25,
        agenda_id: str | None = None,
        topic_id: str | None = None,
        mailto: str | None = None,
        fetch: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Search OpenAlex and persist both the query and immutable result snapshots."""

        if not query.strip():
            raise RecordError("literature query must not be empty")
        if topic_id and not agenda_id:
            raise RecordError("a topic-bound literature search requires agenda_id")
        if agenda_id:
            agenda = self.store.get("research_agenda", agenda_id)
            known_topics = {row["id"] for row in agenda["payload"]["topics"]}
            if topic_id and topic_id not in known_topics:
                raise RecordError(f"agenda {agenda_id} has no topic {topic_id}")
        if limit < 1 or limit > 100:
            raise RecordError("OpenAlex limit must be between 1 and 100")
        searched_at = self._now()
        search_id = f"search_{searched_at.translate(str.maketrans('', '', '-:.TZ+'))[:14]}_{sha256_object({'q': query, 'topic': topic_id, 'at': searched_at})[:12]}"
        params: dict[str, Any] = {
            "search": query,
            "per-page": limit,
            "select": ",".join(
                (
                    "id",
                    "doi",
                    "title",
                    "publication_year",
                    "publication_date",
                    "primary_location",
                    "type",
                    "authorships",
                    "abstract_inverted_index",
                    "open_access",
                    "cited_by_count",
                )
            ),
        }
        if mailto:
            params["mailto"] = mailto
        url = f"{OPENALEX_API}?{urllib.parse.urlencode(params)}"
        response = dict((fetch or _fetch_json)(url))
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise RecordError("OpenAlex response has no results list")
        source_records: list[dict[str, Any]] = []
        for raw_result in raw_results:
            if not isinstance(raw_result, Mapping) or not raw_result.get("title"):
                continue
            source_payload = self._openalex_source(
                raw_result,
                query=query,
                search_id=search_id,
                searched_at=searched_at,
                topic_id=topic_id,
            )
            source_records.append(self.register_source(source_payload))
        search_payload: dict[str, Any] = {
            "id": search_id,
            "provider": "openalex",
            "query": query,
            "searched_at": searched_at,
            "filters": {"limit": limit},
            "result_source_ids": [row["id"] for row in source_records],
            "raw_result_count": len(raw_results),
            "request_url": url,
        }
        if agenda_id:
            search_payload["agenda_id"] = agenda_id
        if topic_id:
            search_payload["topic_id"] = topic_id
        search_record = self._register("literature_search", search_payload)
        return {"search": search_record, "sources": source_records}

    @staticmethod
    def _openalex_source(
        result: Mapping[str, Any],
        *,
        query: str,
        search_id: str,
        searched_at: str,
        topic_id: str | None,
    ) -> dict[str, Any]:
        openalex_id = str(result.get("id", ""))
        doi = str(result.get("doi") or "")
        identity = doi.lower() or openalex_id.lower() or str(result["title"]).lower()
        work_key = f"work_{sha256_object(identity)[:20]}"
        primary = result.get("primary_location")
        primary = primary if isinstance(primary, Mapping) else {}
        venue_source = primary.get("source")
        venue_source = venue_source if isinstance(venue_source, Mapping) else {}
        venue_name = str(venue_source.get("display_name") or "unknown")
        publication_type = str(result.get("type") or "unknown")
        abstract = _abstract(result.get("abstract_inverted_index"))
        # OpenAlex occasionally repeats an author across authorships (same person
        # credited under two affiliations, or a duplicated record upstream). The
        # record model requires unique authors, so an un-deduplicated list makes one
        # malformed result abort the entire search and lose every other source in
        # the page. Deduplicate here, preserving first-seen order so the author list
        # still reads as the byline does.
        authors: list[str] = []
        seen_authors: set[str] = set()
        for authorship in result.get("authorships", []):
            if not isinstance(authorship, Mapping):
                continue
            author = authorship.get("author")
            if isinstance(author, Mapping) and author.get("display_name"):
                name = str(author["display_name"])
                if name not in seen_authors:
                    seen_authors.add(name)
                    authors.append(name)
        urls = [value for value in (doi, openalex_id, primary.get("landing_page_url")) if value]
        snapshot_identity = {
            "work_key": work_key,
            "title": result.get("title"),
            "year": result.get("publication_year"),
            "venue": venue_name,
            "abstract": abstract,
            "search": search_id,
        }
        return {
            "id": f"lit_{work_key.removeprefix('work_')}_{sha256_object(snapshot_identity)[:10]}",
            "work_key": work_key,
            "title": str(result["title"]),
            "authors": authors,
            "year": result.get("publication_year"),
            "venue": _venue_classification(venue_name, publication_type),
            "publication_type": publication_type,
            "identifiers": {
                key: value for key, value in {"doi": doi, "openalex": openalex_id}.items() if value
            },
            "urls": list(dict.fromkeys(str(value) for value in urls)),
            "abstract": abstract,
            "topics": [topic_id] if topic_id else [],
            "retrieval": {
                "provider": "openalex",
                "query": query,
                "retrieved_at": searched_at,
                "search_id": search_id,
            },
            # A retrieved abstract is NOT read text. On 2026-08-17 this provider
            # returned correct titles, authors, years and DOIs while attaching the
            # abstract of an entirely different paper: "Scaling Laws for Neural
            # Language Models" (DOI 2001.08361) came back with an abstract about
            # "transport-validity theory for agentic AI interventions", and
            # FlashAttention (2205.14135) with one about a "Fused Memory-Compute
            # Tile". Verified against arXiv; the inverted-index decode is correct,
            # so the corruption is upstream of us.
            #
            # A claim extracted from that text would carry a real DOI and real
            # authors attached to a fabricated statement -- provenance that looks
            # impeccable and is worthless. So search results are scored as
            # metadata_only regardless of whether an abstract came back: discovery
            # is trustworthy here, content is not. To earn abstract or fulltext
            # credit, fetch the canonical source and register it explicitly with
            # `literature-source`, which is what "reading a paper" means.
            "content": {
                "status": "metadata_only",
                "retrieved_abstract_unverified": bool(abstract),
            },
            "citation_count": int(result.get("cited_by_count") or 0),
        }

    def refresh_agenda(
        self,
        agenda_id: str,
        *,
        limit_per_query: int = 25,
        mailto: str | None = None,
        fetch: Callable[[str], Mapping[str, Any]] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        agenda = self.store.get("research_agenda", agenda_id)
        current = self.synthesize(write=False)
        due = {
            row["topic_id"]
            for row in current["research_gaps"]
            if row["agenda_id"] == agenda_id and row["search_due"]
        }
        searches: list[dict[str, Any]] = []
        for topic in agenda["payload"]["topics"]:
            if not force and topic["id"] not in due:
                continue
            for query in topic["queries"]:
                searches.append(
                    self.search_openalex(
                        query,
                        limit=limit_per_query,
                        agenda_id=agenda_id,
                        topic_id=topic["id"],
                        mailto=mailto,
                        fetch=fetch,
                    )
                )
        snapshot = self.synthesize()
        return {"agenda_id": agenda_id, "searches": searches, "science": snapshot["summary"]}

    def synthesize(self, *, write: bool = True) -> dict[str, Any]:
        sources = {row["id"]: row for row in self.store.list("literature_source")}
        searches = self.store.list("literature_search")
        claims = self.store.list("scientific_claim")
        mechanisms = self.store.list("scientific_mechanism")
        hypotheses = self.store.list("scientific_hypothesis")
        agendas = self.store.list("research_agenda")
        specs = {row["id"]: row for row in self.store.list("experiment_spec")}
        decisions = _latest_decisions(self.store.list("evidence_decision"))
        results = {row["id"]: row for row in self.store.list("result_bundle")}

        beliefs = self._beliefs(claims, sources, decisions, specs)
        belief_by_key = {row["belief_key"]: row for row in beliefs}
        mechanism_rows = self._mechanisms(mechanisms, claims, belief_by_key)
        mechanism_by_id = {row["mechanism_id"]: row for row in mechanism_rows}
        hypothesis_rows = self._hypotheses(
            hypotheses,
            claims,
            belief_by_key,
            mechanism_by_id,
            specs,
            decisions,
            results,
        )
        gaps = self._research_gaps(agendas, sources, claims, searches, beliefs)
        idea_queue = sorted(
            (
                {
                    "hypothesis_id": row["hypothesis_id"],
                    "statement": row["statement"],
                    "readiness": row["readiness"],
                    "confidence": row["confidence"],
                    "experiment_status": row["experiment_status"],
                    "unresolved": row["unresolved"],
                }
                for row in hypothesis_rows
                if row["experiment_status"] in {"untested", "preliminary", "inconclusive"}
            ),
            key=lambda row: (-row["readiness"], -row["confidence"], row["hypothesis_id"]),
        )
        research_tasks = self._research_tasks(
            gaps,
            beliefs,
            mechanism_rows,
            hypothesis_rows,
            claims,
        )
        summary = {
            "sources": len(sources),
            "fulltext_sources": sum(
                row["payload"]["content"]["status"] == "fulltext_snapshot"
                for row in sources.values()
            ),
            "searches": len(searches),
            "claims": len(claims),
            "beliefs": len(beliefs),
            "contested_beliefs": sum(row["state"] == "contested" for row in beliefs),
            "mechanisms": len(mechanism_rows),
            "hypotheses": len(hypothesis_rows),
            "untested_hypotheses": sum(
                row["experiment_status"] == "untested" for row in hypothesis_rows
            ),
            "search_due_topics": sum(row["search_due"] for row in gaps),
            "open_research_tasks": len(research_tasks),
        }
        snapshot = {
            "generated_at": self._now(),
            "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            "confidence_interpretation": (
                "heuristic evidence confidence, not a statistically calibrated posterior; "
                "inspect component scores and provenance before acting"
            ),
            "summary": summary,
            "beliefs": beliefs,
            "mechanisms": mechanism_rows,
            "hypotheses": hypothesis_rows,
            "research_gaps": gaps,
            "idea_queue": idea_queue,
            "research_tasks": research_tasks,
        }
        if write:
            self._write_views(snapshot)
        return snapshot

    def _beliefs(
        self,
        claims: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        decisions: Mapping[str, Mapping[str, Any]],
        specs: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for claim in claims:
            grouped[str(claim["payload"]["belief_key"])].append(claim)
        rows: list[dict[str, Any]] = []
        for belief_key, belief_claims in sorted(grouped.items()):
            contributions: list[dict[str, Any]] = []
            # Repeated extractions from the same work are correlated; only the
            # strongest contribution per work and stance enters confidence.
            per_work: dict[tuple[str, str], dict[str, Any]] = {}
            for claim in belief_claims:
                payload = claim["payload"]
                if payload["origin"] == "literature":
                    for source_id in payload["source_ids"]:
                        source = sources.get(source_id)
                        if source is None:
                            continue
                        contribution = self._literature_contribution(claim, source)
                        key = (source["payload"]["work_key"], payload["stance"])
                        previous = per_work.get(key)
                        if (
                            previous is None
                            or contribution["reliability"] > previous["reliability"]
                        ):
                            per_work[key] = contribution
                elif payload["origin"] == "experiment":
                    contribution = self._experiment_contribution(claim, decisions, specs)
                    if contribution is not None:
                        contributions.append(contribution)
            contributions.extend(per_work.values())
            log_odds = sum(float(row["log_odds"]) for row in contributions)
            confidence = _sigmoid(log_odds)
            positive = sum(row["log_odds"] for row in contributions if row["log_odds"] > 0)
            negative = -sum(row["log_odds"] for row in contributions if row["log_odds"] < 0)
            contested = positive >= 0.45 and negative >= 0.45
            if contested:
                state = "contested"
            elif confidence >= 0.70:
                state = "supported"
            elif confidence <= 0.30:
                state = "rejected"
            else:
                state = "uncertain"
            rows.append(
                {
                    "belief_key": belief_key,
                    "statement": belief_claims[0]["payload"]["statement"],
                    "claim_ids": [row["id"] for row in belief_claims],
                    "confidence": round(confidence, 6),
                    "confidence_level": _confidence_level(confidence),
                    "state": state,
                    "support_weight": round(positive, 6),
                    "opposition_weight": round(negative, 6),
                    "independent_evidence_units": len(contributions),
                    "contributions": sorted(
                        contributions,
                        key=lambda row: (row["kind"], row["provenance_id"]),
                    ),
                }
            )
        return rows

    @staticmethod
    def _research_tasks(
        gaps: Sequence[Mapping[str, Any]],
        beliefs: Sequence[Mapping[str, Any]],
        mechanisms: Sequence[Mapping[str, Any]],
        hypotheses: Sequence[Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        tasks: list[dict[str, Any]] = []
        for gap in gaps:
            key = f"{gap['agenda_id']}:{gap['topic_id']}"
            if gap["search_due"]:
                tasks.append(
                    {
                        "task_id": f"task_search_{sha256_object(key)[:16]}",
                        "kind": "search_literature",
                        "priority": 90,
                        "target_id": key,
                        "reason": ", ".join(gap["reasons"]),
                        "queries": gap["queries"],
                    }
                )
            if gap["analysis_due"]:
                tasks.append(
                    {
                        "task_id": f"task_analyze_{sha256_object(key)[:16]}",
                        "kind": "read_fulltext_and_extract_claims",
                        "priority": 85,
                        "target_id": key,
                        "reason": ", ".join(gap["reasons"]),
                    }
                )
        for belief in beliefs:
            if belief["state"] == "contested":
                tasks.append(
                    {
                        "task_id": f"task_conflict_{sha256_object(belief['belief_key'])[:16]}",
                        "kind": "resolve_claim_conflict",
                        "priority": 80,
                        "target_id": belief["belief_key"],
                        "reason": "independent supporting and opposing evidence both have weight",
                    }
                )
        for mechanism in mechanisms:
            if mechanism["confidence"] < 0.70:
                tasks.append(
                    {
                        "task_id": f"task_mechanism_{sha256_object(mechanism['mechanism_id'])[:16]}",
                        "kind": "strengthen_or_discriminate_mechanism",
                        "priority": 70,
                        "target_id": mechanism["mechanism_id"],
                        "reason": "one or more causal edges remain weak",
                        "weakest_edges": mechanism["weakest_edges"],
                    }
                )
        experiment_claim_keys = {
            row["payload"]["belief_key"]
            for row in claims
            if row["payload"]["origin"] == "experiment"
        }
        for hypothesis in hypotheses:
            status = hypothesis["experiment_status"]
            if status == "untested" and hypothesis["readiness"] > 0:
                tasks.append(
                    {
                        "task_id": f"task_experiment_{sha256_object(hypothesis['hypothesis_id'])[:16]}",
                        "kind": "stage_hypothesis_experiment",
                        "priority": 65,
                        "target_id": hypothesis["hypothesis_id"],
                        "reason": "untested hypothesis has a scored scientific foundation",
                        "readiness": hypothesis["readiness"],
                    }
                )
            elif status in {"preliminary", "inconclusive", "contested"}:
                tasks.append(
                    {
                        "task_id": f"task_refine_{sha256_object(hypothesis['hypothesis_id'])[:16]}",
                        "kind": "refine_hypothesis_or_discriminator",
                        "priority": 75,
                        "target_id": hypothesis["hypothesis_id"],
                        "reason": f"experimental state is {status}",
                        "unresolved": hypothesis["unresolved"],
                    }
                )
            elif (
                status in {"supported", "refuted"}
                and hypothesis["belief_key"] not in experiment_claim_keys
            ):
                tasks.append(
                    {
                        "task_id": f"task_conclude_{sha256_object(hypothesis['hypothesis_id'])[:16]}",
                        "kind": "materialize_experiment_claim",
                        "priority": 95,
                        "target_id": hypothesis["hypothesis_id"],
                        "reason": f"decisive {status} confirmation has not entered the claim ledger",
                    }
                )
        return sorted(tasks, key=lambda row: (-row["priority"], row["task_id"]))

    @staticmethod
    def _literature_contribution(
        claim: Mapping[str, Any], source: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload = claim["payload"]
        evidence = payload["evidence"]
        source_payload = source["payload"]
        components = {
            "venue": VENUE_SCORES[source_payload["venue"]["tier"]],
            "peer_review": PEER_REVIEW_SCORES[source_payload["venue"]["peer_reviewed"]],
            "study_design": DESIGN_SCORES[evidence["study_design"]],
            "content_depth": CONTENT_SCORES[source_payload["content"]["status"]],
            "artifact": ARTIFACT_SCORES[evidence["artifact_status"]],
            "reproduction": REPRODUCTION_SCORES[evidence["reproduction_status"]],
            "directness": DIRECTNESS_SCORES[evidence["directness"]],
            "scope_match": SCOPE_SCORES[evidence["scope_match"]],
            "risk_of_bias": BIAS_SCORES[evidence["risk_of_bias"]],
        }
        reliability = sum(components[key] * COMPONENT_WEIGHTS[key] for key in components)
        # A top venue without artifacts, reproduction, or direct applicability
        # remains moderate. Meta-analyses receive more evidential mass.
        mass = 1.9 if evidence["study_design"] == "meta_analysis" else 1.45
        sign = 1.0 if payload["stance"] == "supports" else -1.0
        if evidence["reproduction_status"] == "independent_failure":
            sign *= -1.0
        return {
            "kind": "literature",
            "provenance_id": source["id"],
            "work_key": source_payload["work_key"],
            "claim_id": claim["id"],
            "stance": "supports" if sign > 0 else "opposes",
            "reliability": round(reliability, 6),
            "log_odds": round(sign * reliability * mass, 6),
            "components": {key: round(value, 6) for key, value in components.items()},
            "venue_is_not_truth": True,
        }

    @staticmethod
    def _experiment_contribution(
        claim: Mapping[str, Any],
        decisions: Mapping[str, Mapping[str, Any]],
        specs: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any] | None:
        payload = claim["payload"]
        by_spec: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        decision_by_id = {row["id"]: row for row in decisions.values()}
        for evidence_id in payload["evidence_ids"]:
            decision = decision_by_id.get(evidence_id)
            if decision is None:
                continue
            evidence_payload = decision["payload"]
            if (
                evidence_payload.get("measurement_verdict") == "valid"
                and evidence_payload.get("claim_status") == "eligible"
                and isinstance(
                    evidence_payload.get("measurements", {}).get("effect_value"), (int, float)
                )
            ):
                by_spec[str(evidence_payload["spec_id"])].append(decision)
        if not by_spec:
            return None
        # One scientific claim should cite one preregistered experiment. If an
        # imported claim cites several, use the least favorable valid outcome.
        outcomes: list[tuple[float, float, str, int]] = []
        for spec_id, rows in by_spec.items():
            spec = specs.get(spec_id)
            if spec is None:
                continue
            spec_payload = spec["payload"]
            values = [float(row["payload"]["measurements"]["effect_value"]) for row in rows]
            aggregate = sum(values) / len(values)
            required = int(spec_payload["analysis"]["minimum_valid_replicates"])
            stage = spec_payload["stage"]
            complete = len(values) >= required
            if not complete:
                outcome_sign = 0.0
            elif rule_matches(aggregate, spec_payload["analysis"]["success_rule"]):
                outcome_sign = 1.0
            elif rule_matches(aggregate, spec_payload["analysis"]["falsifier_rule"]):
                outcome_sign = -1.0
            else:
                outcome_sign = 0.0
            reliability = (0.90 + 0.02 * min(5, len(values))) if stage == "confirmation" else 0.30
            if stage == "confirmation" and not complete:
                reliability = 0.25
            outcomes.append((outcome_sign, min(0.98, reliability), spec_id, len(values)))
        informative = [row for row in outcomes if row[0] != 0]
        if not informative:
            return None
        outcome_sign, reliability, spec_id, count = min(informative, key=lambda row: row[0])
        declared_sign = 1.0 if payload["stance"] == "supports" else -1.0
        # The preregistered result, not the author's label, determines direction.
        # A mismatched declaration remains visible instead of reversing evidence.
        sign = outcome_sign
        return {
            "kind": "experiment",
            "provenance_id": spec_id,
            "claim_id": claim["id"],
            "stance": "supports" if sign > 0 else "opposes",
            "reliability": round(reliability, 6),
            "log_odds": round(sign * reliability * 2.2, 6),
            "valid_replicates": count,
            "declared_stance_mismatch": declared_sign != outcome_sign,
        }

    @staticmethod
    def _mechanisms(
        mechanisms: Sequence[Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
        beliefs: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        claim_map = {row["id"]: row for row in claims}
        rows: list[dict[str, Any]] = []
        for mechanism in mechanisms:
            payload = mechanism["payload"]
            edges: list[dict[str, Any]] = []
            for edge in payload["edges"]:
                confidences = []
                for claim_id in edge["claim_ids"]:
                    claim = claim_map.get(claim_id)
                    if claim is None:
                        continue
                    belief = beliefs.get(str(claim["payload"]["belief_key"]))
                    if belief:
                        confidences.append(float(belief["confidence"]))
                confidence = min(confidences) if confidences else 0.5
                edges.append({**edge, "confidence": round(confidence, 6)})
            confidence = min((row["confidence"] for row in edges), default=0.5)
            rows.append(
                {
                    "mechanism_id": mechanism["id"],
                    "name": payload["name"],
                    "statement": payload["statement"],
                    "confidence": round(confidence, 6),
                    "confidence_level": _confidence_level(confidence),
                    "weakest_edges": [
                        row for row in edges if math.isclose(row["confidence"], confidence)
                    ],
                    "edges": edges,
                    "alternatives": payload["alternatives"],
                    "falsifiers": payload["falsifiers"],
                }
            )
        return rows

    @staticmethod
    def _experiment_status(
        hypothesis_id: str,
        specs: Mapping[str, Mapping[str, Any]],
        decisions: Mapping[str, Mapping[str, Any]],
        results: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        linked = [
            spec
            for spec in specs.values()
            if hypothesis_id in spec["payload"].get("knowledge", {}).get("source_ids", [])
        ]
        summaries: list[dict[str, Any]] = []
        for spec in linked:
            payload = spec["payload"]
            spec_decisions = [
                row
                for row in decisions.values()
                if row["payload"].get("spec_id") == spec["id"]
                and row["payload"].get("measurement_verdict") == "valid"
                and isinstance(
                    row["payload"].get("measurements", {}).get("effect_value"), (int, float)
                )
            ]
            values = [
                float(row["payload"]["measurements"]["effect_value"]) for row in spec_decisions
            ]
            aggregate = sum(values) / len(values) if values else None
            required = int(payload["analysis"]["minimum_valid_replicates"])
            planned = {str(row["replicate_id"]) for row in payload["plan"]}
            terminal = {
                str(row["payload"]["replicate_id"])
                for row in results.values()
                if row["payload"].get("spec_id") == spec["id"] and row["id"] in decisions
            }
            complete = planned == terminal
            if not values:
                status = "untested"
            elif payload["stage"] == "confirmation" and (not complete or len(values) < required):
                status = "preliminary"
            elif rule_matches(float(aggregate), payload["analysis"]["success_rule"]):
                status = (
                    "supported" if payload["stage"] == "confirmation" else "exploratory_support"
                )
            elif rule_matches(float(aggregate), payload["analysis"]["falsifier_rule"]):
                status = (
                    "refuted" if payload["stage"] == "confirmation" else "exploratory_refutation"
                )
            else:
                status = "inconclusive"
            summaries.append(
                {
                    "spec_id": spec["id"],
                    "stage": payload["stage"],
                    "status": status,
                    "effect_mean": aggregate,
                    "valid_replicates": len(values),
                    "required_replicates": required,
                    "plan_complete": complete,
                    "evidence_ids": [row["id"] for row in spec_decisions],
                }
            )
        precedence = {
            "refuted": 7,
            "supported": 6,
            "inconclusive": 5,
            "preliminary": 4,
            "exploratory_refutation": 3,
            "exploratory_support": 2,
            "untested": 1,
        }
        statuses = {row["status"] for row in summaries}
        if {"supported", "refuted"}.issubset(statuses):
            status = "contested"
        else:
            decisive = (
                max(summaries, key=lambda row: precedence[row["status"]]) if summaries else None
            )
            status = decisive["status"] if decisive else "untested"
        return {"status": status, "experiments": summaries}

    def _hypotheses(
        self,
        hypotheses: Sequence[Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
        beliefs: Mapping[str, Mapping[str, Any]],
        mechanisms: Mapping[str, Mapping[str, Any]],
        specs: Mapping[str, Mapping[str, Any]],
        decisions: Mapping[str, Mapping[str, Any]],
        results: Mapping[str, Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        claim_map = {row["id"]: row for row in claims}
        rows: list[dict[str, Any]] = []
        for hypothesis in hypotheses:
            payload = hypothesis["payload"]
            foundation = []
            foundation_units: set[tuple[str, str]] = set()
            unresolved: list[str] = []
            for claim_id in payload["claim_ids"]:
                claim = claim_map.get(claim_id)
                belief = beliefs.get(str(claim["payload"]["belief_key"])) if claim else None
                if belief:
                    foundation.append(float(belief["confidence"]))
                    for contribution in belief["contributions"]:
                        unit = str(contribution.get("work_key", contribution["provenance_id"]))
                        foundation_units.add((str(contribution["kind"]), unit))
                    if belief["state"] in {"contested", "uncertain", "rejected"}:
                        unresolved.append(
                            f"claim foundation {belief['belief_key']} is {belief['state']}"
                        )
            for mechanism_id in payload["mechanism_ids"]:
                mechanism = mechanisms.get(mechanism_id)
                if mechanism:
                    foundation.append(float(mechanism["confidence"]))
                    if mechanism["confidence"] < 0.60:
                        unresolved.append(f"mechanism {mechanism_id} has weak edges")
            foundation_confidence = sum(foundation) / len(foundation) if foundation else 0.5
            if len(foundation_units) < 2:
                unresolved.append(
                    "fewer than two independent evidence units support the foundation"
                )
            experiment = self._experiment_status(hypothesis["id"], specs, decisions, results)
            status = experiment["status"]
            log_odds = math.log(
                max(1e-6, foundation_confidence) / max(1e-6, 1 - foundation_confidence)
            )
            if status == "supported":
                log_odds += 2.4
            elif status == "refuted":
                log_odds -= 2.4
            elif status == "exploratory_support":
                log_odds += 0.35
            elif status == "exploratory_refutation":
                log_odds -= 0.35
            elif status == "contested":
                unresolved.append("independent confirmation results conflict")
            confidence = _sigmoid(log_odds)
            if status in {"supported", "refuted"}:
                readiness = 0.0
            else:
                readiness = foundation_confidence
                if unresolved:
                    readiness *= 0.7
            rows.append(
                {
                    "hypothesis_id": hypothesis["id"],
                    "belief_key": payload["belief_key"],
                    "topics": payload["topics"],
                    "statement": payload["statement"],
                    "rationale": payload["rationale"],
                    "prediction": payload["prediction"],
                    "falsifiers": payload["falsifiers"],
                    "mechanism_ids": payload["mechanism_ids"],
                    "claim_ids": payload["claim_ids"],
                    "foundation_confidence": round(foundation_confidence, 6),
                    "foundation_evidence_units": len(foundation_units),
                    "confidence": round(confidence, 6),
                    "confidence_level": _confidence_level(confidence),
                    "experiment_status": status,
                    "experiments": experiment["experiments"],
                    "readiness": round(readiness, 6),
                    "unresolved": sorted(set(unresolved)),
                    "intervention": payload["intervention"],
                }
            )
        return rows

    def _research_gaps(
        self,
        agendas: Sequence[Mapping[str, Any]],
        sources: Mapping[str, Mapping[str, Any]],
        claims: Sequence[Mapping[str, Any]],
        searches: Sequence[Mapping[str, Any]],
        beliefs: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        now = _parse_time(self._now())
        contested = {row["belief_key"] for row in beliefs if row["state"] == "contested"}
        rows: list[dict[str, Any]] = []
        for agenda in agendas:
            for topic in agenda["payload"]["topics"]:
                topic_id = topic["id"]
                topic_sources = [
                    source for source in sources.values() if topic_id in source["payload"]["topics"]
                ]
                topic_work_keys = {source["payload"]["work_key"] for source in topic_sources}
                topic_claims = [
                    claim for claim in claims if topic_id in claim["payload"].get("topics", [])
                ]
                topic_searches = [
                    search
                    for search in searches
                    if search["payload"].get("agenda_id") == agenda["id"]
                    and search["payload"].get("topic_id") == topic_id
                ]
                last_search = max(
                    (_parse_time(str(row["payload"]["searched_at"])) for row in topic_searches),
                    default=None,
                )
                reasons = []
                if len(topic_work_keys) < int(topic["minimum_sources"]):
                    reasons.append("insufficient_sources")
                if len(topic_claims) < int(topic["minimum_claims"]):
                    reasons.append("insufficient_extracted_claims")
                if any(claim["payload"]["belief_key"] in contested for claim in topic_claims):
                    reasons.append("contested_claims")
                if last_search is None:
                    reasons.append("never_searched")
                elif now - last_search > timedelta(days=int(topic["refresh_days"])):
                    reasons.append("search_stale")
                if topic_sources and not any(
                    row["payload"]["content"]["status"] == "fulltext_snapshot"
                    for row in topic_sources
                ):
                    reasons.append("no_fulltext_snapshot")
                search_reasons = {
                    "insufficient_sources",
                    "never_searched",
                    "search_stale",
                }
                contested_refresh_due = "contested_claims" in reasons and (
                    last_search is None
                    or now - last_search > timedelta(days=max(1, int(topic["refresh_days"]) // 2))
                )
                rows.append(
                    {
                        "agenda_id": agenda["id"],
                        "topic_id": topic_id,
                        "question": topic["question"],
                        "queries": topic["queries"],
                        "source_count": len(topic_work_keys),
                        "claim_count": len(topic_claims),
                        "last_searched_at": last_search.isoformat().replace("+00:00", "Z")
                        if last_search
                        else None,
                        "search_due": bool(search_reasons & set(reasons)) or contested_refresh_due,
                        "analysis_due": any(
                            reason in {"insufficient_extracted_claims", "no_fulltext_snapshot"}
                            for reason in reasons
                        ),
                        "reasons": sorted(set(reasons)),
                    }
                )
        return rows

    def conclusion_template(self, hypothesis_id: str, spec_id: str) -> dict[str, Any]:
        hypothesis = self.store.get("scientific_hypothesis", hypothesis_id)
        snapshot = self.synthesize(write=False)
        row = next(
            (item for item in snapshot["hypotheses"] if item["hypothesis_id"] == hypothesis_id),
            None,
        )
        if row is None:
            raise RecordError(f"hypothesis projection is missing: {hypothesis_id}")
        experiment = next((item for item in row["experiments"] if item["spec_id"] == spec_id), None)
        if experiment is None or experiment["status"] not in {"supported", "refuted"}:
            raise RecordError("a scientific conclusion requires a completed decisive confirmation")
        stance = "supports" if experiment["status"] == "supported" else "opposes"
        return {
            "id": f"claim_{_slug(hypothesis_id)}_{_slug(spec_id)}_{stance}",
            "belief_key": hypothesis["payload"]["belief_key"],
            "statement": hypothesis["payload"]["statement"],
            "claim_type": "empirical",
            "origin": "experiment",
            "stance": stance,
            "scope": copy.deepcopy(hypothesis["payload"]["scope"]),
            "source_ids": [],
            "evidence_ids": experiment["evidence_ids"],
            "derived_from_claim_ids": hypothesis["payload"]["claim_ids"],
            "topics": [],
            "locators": [],
            "evidence": {
                "study_design": "controlled_benchmark",
                "artifact_status": "verified",
                "reproduction_status": "author_only",
                "directness": "direct",
                "scope_match": "exact",
                "risk_of_bias": "low",
                "metrics": [
                    {
                        "name": hypothesis["payload"]["prediction"]["metric"],
                        "result": f"effect_mean={experiment['effect_mean']}",
                    }
                ],
            },
            "assessment": {
                "assessor": "autoresearch-evidence-linker",
                "assessed_at": self._now(),
                "rationale": (
                    f"{experiment['status']} by preregistered confirmation {spec_id} with "
                    f"{experiment['valid_replicates']} valid replicates"
                ),
                "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
            },
        }

    def _write_views(self, snapshot: Mapping[str, Any]) -> None:
        self.store.write_view("SCIENCE.json", canonical_json(snapshot) + "\n")
        for name, key in (
            ("SCIENTIFIC_BELIEFS.json", "beliefs"),
            ("MECHANISMS.json", "mechanisms"),
            ("HYPOTHESES.json", "hypotheses"),
            ("RESEARCH_GAPS.json", "research_gaps"),
            ("IDEA_QUEUE.json", "idea_queue"),
            ("RESEARCH_TASKS.json", "research_tasks"),
        ):
            self.store.write_view(name, canonical_json(snapshot[key]) + "\n")
        self.store.write_view("SCIENCE.md", _science_markdown(snapshot))


def _science_markdown(snapshot: Mapping[str, Any]) -> str:
    lines = [
        "# Scientific library",
        "",
        f"Generated: `{snapshot['generated_at']}`",
        "",
        "> Confidence is a transparent heuristic evidence score, not a calibrated posterior.",
        "> Inspect provenance and component scores before treating a belief as established.",
        "",
        "## Beliefs",
        "",
    ]
    for row in snapshot["beliefs"]:
        lines.extend(
            [
                f"### `{row['belief_key']}` — {row['state']}",
                "",
                row["statement"],
                "",
                f"Confidence: **{row['confidence']:.3f}** ({row['confidence_level']}); "
                f"{row['independent_evidence_units']} independent evidence unit(s).",
                "",
            ]
        )
    lines.extend(["## Hypotheses", ""])
    for row in snapshot["hypotheses"]:
        lines.extend(
            [
                f"- `{row['hypothesis_id']}`: {row['statement']} — "
                f"{row['experiment_status']}, confidence {row['confidence']:.3f}",
                "",
            ]
        )
    lines.extend(["## Research due", ""])
    due = [row for row in snapshot["research_gaps"] if row["search_due"]]
    if not due:
        lines.extend(["No agenda topic is currently due.", ""])
    for row in due:
        lines.extend(
            [
                f"- `{row['agenda_id']}/{row['topic_id']}`: {', '.join(row['reasons'])}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
