import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "build_candidate_interface_extension_reference_union.py"
SPEC = importlib.util.spec_from_file_location("ext_union", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalize_pdb_variants():
    assert MODULE.normalize_pdb("1ABC") == "1abc"
    assert MODULE.normalize_pdb("pdb_1abc_0") == "1abc"
    assert MODULE.normalize_pdb("0001abc") == "1abc"


def test_snapshot_entry_ids_normalizes_unique_ids(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "search_response.json").write_text(json.dumps({
        "result_set": ["1ABC", "2DEF", {"identifier": "3ghi"}],
    }), encoding="ascii")
    ids = MODULE.snapshot_entry_ids(snapshot)
    assert ids == ["1abc", "2def", "3ghi"]


def test_snapshot_entry_ids_rejects_duplicates(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "search_response.json").write_text(json.dumps({
        "result_set": ["1ABC", {"identifier": "1abc"}],
    }), encoding="ascii")
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.snapshot_entry_ids(snapshot)


def test_reference_records_builds_valid_records(tmp_path):
    structural = tmp_path / "structural_manifest.json"
    structural.write_text(json.dumps({
        "records": [{
            "instance": "33CI_E_D_C",
            "pdb_id": "33CI",
            "vh_sequence": "QVQLVQ",
            "vl_sequence": "DIQMTQ",
            "paired_cdr_sequence": "SSAAAAGGGGLLL",
            "cdr_h3_sequence": "AAAA",
            "antigen_sequence": "KVAELVHFL",
        }],
    }), encoding="utf-8")
    records = MODULE.reference_records(structural)
    assert records[0]["reference_id"] == "EXTR00001"
    assert records[0]["pdb_id"] == "33ci"
    assert records[0]["vh_sequence"] == "QVQLVQ"


def test_build_rejects_non_union_base(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"status": "something_else", "records": []}))
    with pytest.raises(ValueError, match="not a frozen successor-v3 reference union"):
        MODULE.build(base, tmp_path / "s", tmp_path / "m", tmp_path / "out.json")


def test_build_refuses_overwrite(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"status": MODULE.UNION_STATUS, "records": [],
                                "exact_exposed_pdb_ids": []}))
    with pytest.raises(FileExistsError):
        MODULE.build(base, tmp_path / "s", tmp_path / "m", out)
