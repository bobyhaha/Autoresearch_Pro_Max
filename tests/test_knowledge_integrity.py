"""The knowledge base must not be able to cite something that does not exist.

This test exists because it happened. `kb_scaling_embeddings` was listed in
`knowledge/INDEX.md` and cited in a paper for about five hours while the file had
never been written -- a `cd X && cat > file` chain failed on the `cd`, so the first
heredoc was skipped while the later ones succeeded, and the resulting file listing
was misread as confirmation.

An index entry is not evidence of a file. These checks make that a test failure
rather than something a reader has to notice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE = ROOT / "knowledge"
CARD_LINK = re.compile(r"\]\((papers/[\w.\-]+\.md|mechanisms/[\w.\-]+\.md)\)")
WIKILINK = re.compile(r"\[\[([^\]|]+)\]\]")


def card_ids() -> set[str]:
    if not KNOWLEDGE.exists():
        return set()
    return {
        path.stem
        for folder in ("papers", "mechanisms")
        for path in (KNOWLEDGE / folder).glob("*.md")
    }


@pytest.mark.skipif(not KNOWLEDGE.exists(), reason="no knowledge base in this checkout")
def test_index_links_all_resolve():
    index = KNOWLEDGE / "INDEX.md"
    missing = [
        target
        for target in CARD_LINK.findall(index.read_text(encoding="utf-8"))
        if not (KNOWLEDGE / target).is_file()
    ]
    assert not missing, f"INDEX.md links cards that do not exist: {missing}"


@pytest.mark.skipif(not KNOWLEDGE.exists(), reason="no knowledge base in this checkout")
def test_every_card_declares_a_verification_status():
    allowed = {"discovery", "partially_verified", "fulltext_verified", "synthesis_ready"}
    for folder in ("papers", "mechanisms"):
        for path in (KNOWLEDGE / folder).glob("*.md"):
            text = path.read_text(encoding="utf-8")
            found = re.search(r"^status:\s*(\S+)", text, re.MULTILINE)
            assert found, f"{path.name} has no status: field"
            assert found.group(1) in allowed, (
                f"{path.name} has unknown status {found.group(1)!r}; "
                "the status gate is what decides whether a card may support an experiment"
            )


@pytest.mark.skipif(not KNOWLEDGE.exists(), reason="no knowledge base in this checkout")
def test_cross_card_wikilinks_resolve():
    """A [[link]] to a card that was never written is the failure this test is named for."""
    known = card_ids()
    dangling: list[str] = []
    for folder in ("papers", "mechanisms"):
        for path in (KNOWLEDGE / folder).glob("*.md"):
            for raw in WIKILINK.findall(path.read_text(encoding="utf-8")):
                target = Path(raw.strip()).name
                if target and target not in known:
                    dangling.append(f"{path.name} -> {raw.strip()}")
    assert not dangling, f"dangling card references: {dangling}"


@pytest.mark.skipif(
    not (ROOT / "runs" / "state" / "records" / "experiment_spec").exists(),
    reason="no run state in this checkout",
)
def test_every_cited_source_id_resolves_to_a_card():
    """An ExperimentSpec's knowledge.source_ids is its provenance. A spec citing a
    card that does not exist has no provenance, only the appearance of some."""
    known = card_ids()
    specs = ROOT / "runs" / "state" / "records" / "experiment_spec"
    missing: list[str] = []
    for path in specs.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))["payload"]
        for source in payload["knowledge"]["source_ids"]:
            if source.startswith(("kb_", "mech_")) and source not in known:
                missing.append(f"{path.stem} cites {source}")
    assert not missing, f"specs cite non-existent knowledge cards: {missing}"
