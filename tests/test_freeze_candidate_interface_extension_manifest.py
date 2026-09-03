import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "build" / "freeze_candidate_interface_extension_manifest.py"
SPEC = importlib.util.spec_from_file_location("freeze_ext", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_record(instance):
    return {
        "instance": instance,
        "pdb_id": instance.split("_")[0],
        "viral_antigen": False,
        "resolution": 2.0,
        "antigen_sequence": "KVAELVHFL",
        "cdr_h3_sequence": "AAAA",
        "vh_sequence": "QVQLVQ",
        "vl_sequence": "DIQMTQ",
        "paired_cdr_sequence": "SSAAAAGGGGLLL",
        "n_contacting_h3_positions": 1,
        "n_h3_antigen_residue_contacts": 2,
        "pdb_path": "data/x.pdb",
        "pdb_sha256": "abc",
    }


def make_audit(tmp_path, components):
    return (tmp_path / "audit.json").write_text(json.dumps({
        "status": "frozen_model_free_successor_v3_isolation_audit",
        "gate_passed": True,
        "components": components,
    }), encoding="utf-8")


def make_merged(tmp_path, instances):
    (tmp_path / "merged.json").write_text(json.dumps({
        "records": [make_record(i) for i in instances],
    }), encoding="utf-8")


def test_freeze_builds_membership(tmp_path):
    instances = ["1ABC_H_L_P", "2DEF_H_L_P", "3GHI_H_L_P", "4JKL_H_L_P", "5MNO_H_L_P"]
    make_merged(tmp_path, instances)
    make_audit(tmp_path, [
        {"component_id": "EXT001", "representative_id": "1ABC_H_L_P", "members": ["1ABC_H_L_P"]},
        {"component_id": "EXT002", "representative_id": "2DEF_H_L_P", "members": ["2DEF_H_L_P"]},
        {"component_id": "EXT003", "representative_id": "3GHI_H_L_P", "members": ["3GHI_H_L_P"]},
        {"component_id": "EXT004", "representative_id": "4JKL_H_L_P", "members": ["4JKL_H_L_P"]},
        {"component_id": "EXT005", "representative_id": "5MNO_H_L_P", "members": ["5MNO_H_L_P"]},
    ])
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({"frozen_split_manifest_sha256": "x"}), encoding="utf-8")
    manifest = MODULE.freeze(tmp_path / "audit.json", tmp_path / "merged.json",
                             contract, tmp_path / "out.json")
    assert manifest["membership"]["train"] == []
    assert len(manifest["membership"]["calibration"]) == 5
    assert manifest["membership"]["test"] == []


def test_freeze_rejects_failed_gate(tmp_path):
    make_merged(tmp_path, ["1ABC_H_L_P"])
    (tmp_path / "audit.json").write_text(json.dumps({
        "status": "frozen_model_free_feasibility_failure",
        "gate_passed": False,
        "components": [],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="gate did not pass"):
        MODULE.freeze(tmp_path / "audit.json", tmp_path / "merged.json",
                      tmp_path / "c.json", tmp_path / "out.json")


def test_freeze_refuses_overwrite(tmp_path):
    out = tmp_path / "out.json"
    out.write_text("sentinel")
    with pytest.raises(FileExistsError):
        MODULE.freeze(tmp_path / "a.json", tmp_path / "m.json",
                      tmp_path / "c.json", out)
