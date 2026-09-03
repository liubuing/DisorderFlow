import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "merge_structural_manifests.py"
SPEC = importlib.util.spec_from_file_location("merge_manifests", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_manifest(path, instances):
    records = []
    for instance in instances:
        records.append({
            "instance": instance,
            "pdb_id": instance.split("_")[0],
            "vh_sequence": "QVQLVQ",
            "vl_sequence": "DIQMTQ",
            "paired_cdr_sequence": "SSAAAAGGGGLLL",
            "cdr_h3_sequence": "AAAA",
            "antigen_sequence": "KVAELVHFL",
            "n_contacting_h3_positions": 1,
            "n_h3_antigen_residue_contacts": 2,
            "resolution": 2.0,
        })
    path.write_text(json.dumps({"records": records}), encoding="utf-8")


def test_merge_combines_unique_records(tmp_path):
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    make_manifest(m2, ["2DEF_H_L_P"])
    merged = MODULE.merge([m1, m2], tmp_path / "out.json")
    assert merged["counts"]["total_records"] == 2
    assert sorted(r["instance"] for r in merged["records"]) == ["1ABC_H_L_P", "2DEF_H_L_P"]


def test_merge_rejects_duplicate_instances(tmp_path):
    m1 = tmp_path / "m1.json"
    m2 = tmp_path / "m2.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    make_manifest(m2, ["1ABC_H_L_P"])
    with pytest.raises(ValueError, match="Duplicate instance"):
        MODULE.merge([m1, m2], tmp_path / "out.json")


def test_merge_refuses_overwrite(tmp_path):
    m1 = tmp_path / "m1.json"
    make_manifest(m1, ["1ABC_H_L_P"])
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.merge([m1], out)
