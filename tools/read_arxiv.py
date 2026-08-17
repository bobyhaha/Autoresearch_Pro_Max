#!/usr/bin/env python3
"""Read papers from arXiv itself, because the metadata provider cannot be trusted.

On 2026-08-17 the OpenAlex-backed `literature-search` returned correct titles,
authors, years and DOIs with the abstract of an entirely different paper attached:
"Scaling Laws for Neural Language Models" (2001.08361) came back describing a
"transport-validity theory for agentic AI interventions", FlashAttention
(2205.14135) describing a "Fused Memory-Compute Tile". Verified against arXiv. A
claim extracted from that text would carry a real DOI and real authors attached to
a fabricated statement -- provenance that looks impeccable and is worthless.

So discovery may come from anywhere, but *content* comes from the publisher. This
queries the arXiv Atom API directly and stores what arXiv itself returns, with the
query and retrieval time recorded so a reader can re-run it and get the same thing.

    uv run python tools/read_arxiv.py --query "efficient language model pretraining" \
        --from-year 2025 --limit 20 --topic topic_throughput_efficiency
    uv run python tools/read_arxiv.py --plan   # run the campaign's standing reading list

Writes one LiteratureSource declaration per paper into runs/science/sources/ and
prints a registration command. Nothing is registered automatically: what enters the
immutable store stays a deliberate act.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "runs" / "science" / "sources"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# The campaign's standing reading list. Each entry is (topic_id, query). Topics
# match runs/science/agendas/, so extracted claims land under a real agenda rather
# than floating free.
PLAN: list[tuple[str, str]] = [
    ("topic_throughput_efficiency",
     'cat:cs.LG AND abs:"language model" AND abs:"training" AND abs:"throughput"'),
    ("topic_throughput_efficiency",
     'cat:cs.LG AND abs:"data loading" AND abs:"GPU"'),
    ("topic_kernel_efficiency",
     'cat:cs.LG AND abs:"kernel fusion" AND abs:"GPU"'),
    ("topic_kernel_efficiency",
     'cat:cs.LG AND abs:"attention" AND abs:"memory bandwidth" AND abs:"efficient"'),
    ("topic_optimizer_short_horizon",
     'cat:cs.LG AND abs:"optimizer" AND abs:"large language model" AND abs:"pretraining"'),
    ("topic_optimizer_short_horizon",
     'cat:cs.LG AND abs:"learning rate schedule" AND abs:"language model"'),
    ("topic_architecture_small_budget",
     'cat:cs.CL AND abs:"sliding window attention" AND abs:"transformer"'),
    ("topic_architecture_small_budget",
     'cat:cs.LG AND abs:"normalization" AND abs:"transformer" AND abs:"training"'),
    ("topic_architecture_small_budget",
     'cat:cs.CL AND abs:"small language model" AND abs:"architecture"'),
    ("topic_token_exposure_exchange_rate",
     'cat:cs.LG AND abs:"scaling law" AND abs:"language model"'),
    ("topic_token_exposure_exchange_rate",
     'cat:cs.LG AND abs:"compute-optimal" AND abs:"training"'),
    ("topic_data_quality",
     'cat:cs.CL AND abs:"data curation" AND abs:"pretraining"'),
]


def fetch(query: str, limit: int, from_year: int, retries: int = 3) -> list[dict]:
    """One arXiv Atom query, newest first, filtered to >= from_year."""
    url = (
        "https://export.arxiv.org/api/query?"
        + urllib.parse.urlencode(
            {
                # A query containing a field prefix (abs:, ti:, cat:) is passed
                # through verbatim. Bare text gets all:, which is OR-ish and, sorted
                # by date, happily returns a MIMO detection paper for a query about
                # transformer pretraining. Prefer explicit fields.
                "search_query": query if re.search(r"\b(abs|ti|cat|au):", query) else f"all:{query}",
                "start": 0,
                "max_results": limit * 3,  # over-fetch; the year filter discards most
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
        )
    )
    for attempt in range(retries):
        done = subprocess.run(
            ["curl", "-sS", "--max-time", "45", url],
            capture_output=True, text=True, check=False,
        )
        if done.returncode == 0 and done.stdout.strip().startswith("<?xml"):
            break
        time.sleep(3 * (attempt + 1))
    else:
        return []

    try:
        root = ET.fromstring(done.stdout)
    except ET.ParseError:
        return []

    papers: list[dict] = []
    for entry in root.findall(f"{ATOM}entry"):
        published = (entry.findtext(f"{ATOM}published") or "")[:10]
        if not published or int(published[:4]) < from_year:
            continue
        raw_id = entry.findtext(f"{ATOM}id") or ""
        arxiv_id = raw_id.rsplit("/", 1)[-1]
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        summary = " ".join((entry.findtext(f"{ATOM}summary") or "").split())
        if not title or not summary:
            continue
        authors, seen = [], set()
        for a in entry.findall(f"{ATOM}author"):
            name = (a.findtext(f"{ATOM}name") or "").strip()
            if name and name not in seen:  # the record model requires unique authors
                seen.add(name)
                authors.append(name)
        cats = [c.get("term") for c in entry.findall(f"{ATOM}category") if c.get("term")]
        papers.append(
            {
                "arxiv_id": arxiv_id,
                "title": title,
                "abstract": summary,
                "authors": authors,
                "published": published,
                "year": int(published[:4]),
                "categories": cats,
                "comment": " ".join((entry.findtext(f"{ARXIV_NS}comment") or "").split()),
            }
        )
        if len(papers) >= limit:
            break
    return papers


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", s)[:70]


def declaration(paper: dict, topic: str, query: str) -> dict:
    base = slug(paper["title"]) or hashlib.sha256(paper["arxiv_id"].encode()).hexdigest()[:16]
    return {
        "id": f"lit_{base}",
        "work_key": f"work_arxiv_{paper['arxiv_id'].replace('.', '_').replace('v', '_v')}",
        "title": paper["title"],
        "authors": paper["authors"],
        "year": paper["year"],
        "venue": {"name": "arXiv", "tier": "preprint", "peer_reviewed": "unknown"},
        "publication_type": "preprint",
        "identifiers": {"arxiv": paper["arxiv_id"]},
        "urls": [f"https://arxiv.org/abs/{paper['arxiv_id']}"],
        "abstract": paper["abstract"],
        "topics": [topic],
        "retrieval": {
            "provider": "arxiv_atom_api",
            "query": query,
            "retrieved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        # abstract_only, not fulltext_snapshot: this is the abstract as arXiv itself
        # served it, which is verified provenance but is still not the paper. Claims
        # about method detail need the full text and must say so.
        "content": {"status": "abstract_only", "source": "arxiv_atom_api_verbatim"},
        "citation_count": 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--query")
    ap.add_argument("--topic", default="topic_throughput_efficiency")
    ap.add_argument("--from-year", type=int, default=2025)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--plan", action="store_true", help="run the standing reading list")
    args = ap.parse_args()

    jobs = PLAN if args.plan else [(args.topic, args.query)]
    if not args.plan and not args.query:
        raise SystemExit("give --query or --plan")

    OUT.mkdir(parents=True, exist_ok=True)
    written, skipped = [], 0
    seen_ids = {p.stem for p in OUT.glob("*.json")}
    for topic, query in jobs:
        papers = fetch(query, args.limit, args.from_year)
        print(f"{len(papers):3d} papers  [{topic}]  {query}")
        for paper in papers:
            decl = declaration(paper, topic, query)
            if decl["id"] in seen_ids:
                skipped += 1
                continue
            seen_ids.add(decl["id"])
            path = OUT / f"{decl['id']}.json"
            path.write_text(json.dumps(decl, indent=2, sort_keys=True) + "\n")
            written.append(path)
        time.sleep(3)  # arXiv asks for one request every 3 seconds

    print(f"\nwrote {len(written)} new declarations, skipped {skipped} duplicates")
    print("register with:")
    print("  for f in runs/science/sources/lit_*.json; do "
          "uv run autoresearch --root .autoresearch literature-source \"$f\"; done")


if __name__ == "__main__":
    main()
