import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "filter_viral_candidates.py"
SPEC = importlib.util.spec_from_file_location("filter_viral", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_filter_viral_keeps_only_viral_candidates(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "classification": "metadata_only",
        "schema_version": "v1",
        "snapshot_dir": "data/snap",
        "snapshot_manifest_sha256": "abc",
        "entries": [
            {"pdb_id": "1ABC", "status": "candidate", "viral_antigen": True},
            {"pdb_id": "2DEF", "status": "candidate", "viral_antigen": False},
            {"pdb_id": "3GHI", "status": "excluded", "viral_antigen": True},
            {"pdb_id": "4JKL", "status": "deferred_viral_antigen"},
        ],
    }), encoding="utf-8")
    artifact = MODULE.filter_viral(discovery, tmp_path / "out.json")
    assert artifact["counts"]["viral_antigen_candidates"] == 1
    assert artifact["entries"][0]["pdb_id"] == "1ABC"


def test_filter_viral_refuses_overwrite(tmp_path):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({"classification": "metadata_only", "entries": []}))
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.filter_viral(discovery, out)
