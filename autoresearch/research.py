"""Research Engine: turn a reviewed scientific design into an ExperimentSpec."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .records import ConflictError, RecordError, make_record, read_json, sha256_file
from .store import Store


class ResearchEngine:
    def __init__(self, store: Store) -> None:
        self.store = store

    def create(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        raw = dict(proposal)
        record_id = str(raw.pop("id", ""))
        if not record_id:
            raise RecordError("an ExperimentSpec proposal requires id")
        path = self.store.record_path("experiment_spec", record_id)
        if path.exists():
            existing = self.store.get("experiment_spec", record_id)
            if existing["payload"] == raw:
                return existing
            raise ConflictError(f"ExperimentSpec {record_id} already exists with other content")
        record = make_record("experiment_spec", record_id, raw)
        return self.store.put(record)

    def create_from_file(self, path: str | Path) -> dict[str, Any]:
        return self.create(read_json(Path(path)))

    def register_paper(self, declaration: Mapping[str, Any], *, base: Path | None = None) -> dict:
        raw = dict(declaration)
        record_id = str(raw.pop("id", ""))
        path = Path(str(raw.get("path", "")))
        if base is not None and not path.is_absolute():
            path = base / path
        if not path.is_file():
            raise RecordError(f"paper source does not exist: {path}")
        raw["path"] = str(path.resolve())
        actual_digest = sha256_file(path)
        declared = raw.get("content_sha256")
        if declared not in {None, "", actual_digest}:
            raise RecordError("paper content_sha256 does not match the file")
        raw["content_sha256"] = actual_digest
        blob_digest, blob = self.store.add_blob(path)
        if blob_digest != actual_digest:
            raise RecordError("paper blob digest disagrees with source digest")
        raw["blob"] = blob
        path = self.store.record_path("paper", record_id)
        if path.exists():
            existing = self.store.get("paper", record_id)
            if existing["payload"] == raw:
                return existing
            raise ConflictError(f"paper {record_id} already exists with other content")
        return self.store.put(make_record("paper", record_id, raw))
