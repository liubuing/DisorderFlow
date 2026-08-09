import json
import hashlib
from datetime import datetime, timezone

import pytest

from scripts.build.acquire_rcsb_successor_v3_snapshot import (
    QUERY_SPEC,
    acquire,
    query_sha256,
)
from scripts.build.discover_rcsb_successor_v3_snapshot import discover


class Response:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return self.payload


def test_rcsb_query_is_frozen_and_write_once(tmp_path):
    assert QUERY_SPEC["query"]["nodes"][1]["parameters"]["value"] == "2026-07-14"
    assert QUERY_SPEC["request_options"]["sort"] == [
        {"sort_by": "rcsb_id", "direction": "asc"}]
    payload = {"query_id": "q", "total_count": 2, "result_set": ["1AAA", "2BBB"]}
    output = tmp_path / "snapshot"
    report = acquire(
        output, opener=lambda *_args, **_kwargs: Response(payload),
        now=datetime(2026, 8, 7, tzinfo=timezone.utc))
    assert report["query_sha256"] == query_sha256()
    assert report["entry_ids"] == ["1AAA", "2BBB"]
    with pytest.raises(FileExistsError):
        acquire(output, opener=lambda *_args, **_kwargs: pytest.fail("network called"))


def test_rcsb_exact_discovery_excludes_sabdab_and_reference(tmp_path):
    acquisition = tmp_path / "acquisition.json"
    acquisition.write_text(json.dumps({
        "classification": "metadata_entry_ids_only",
        "entry_ids": ["1AAA", "2BBB", "3CCC"],
    }), encoding="ascii")
    summary = tmp_path / "summary.csv"
    summary.write_text("PDB\npdb_00001aaa\n", encoding="ascii")
    reference = tmp_path / "reference_union_manifest_v3.json"
    reference.write_text(json.dumps({
        "exact_exposed_pdb_ids": ["2bbb"],
        "records": [{"pdb_id": "2bbb"}],
    }), encoding="ascii")
    output = tmp_path / "discovery.json"
    reference_sha256 = hashlib.sha256(reference.read_bytes()).hexdigest()
    report = discover(
        acquisition, summary, reference, output,
        expected_reference_sha256=reference_sha256)
    assert report["candidate_entry_ids"] == ["3CCC"]
    assert report["metadata_feasibility_passed"] is False
    with pytest.raises(FileExistsError):
        discover(
            acquisition, summary, reference, output,
            expected_reference_sha256=reference_sha256)
