import json

import pytest

from scripts.audit_successor_v3_isolation import (
    connected_components,
    exact_pdb_exclusion,
    select_donor,
    select_representative,
    write_json_once,
)


def candidate(instance, pdb_id, antigen, positions=1, pairs=1, resolution=2.0):
    return {
        "instance": instance,
        "pdb_id": pdb_id,
        "antigen_sequence": antigen,
        "n_contacting_h3_positions": positions,
        "n_h3_antigen_residue_contacts": pairs,
        "resolution": resolution,
    }


def test_exact_exclusion_normalizes_pdb_ids_and_orders_deterministically():
    rows = [candidate("z", "PDB_00001ABC", "AAAA"),
            candidate("a", "2def", "AAAA")]
    eligible, excluded = exact_pdb_exclusion(rows, ["1abc"])
    assert [row["instance"] for row in eligible] == ["a"]
    assert [row["instance"] for row in excluded] == ["z"]


def test_connected_components_join_hits_transitively_across_axes():
    hits = {
        "vh": [{"query": "b", "target": "a", "identity": 91.0}],
        "vl": [],
        "paired_cdr": [{"query": "c", "target": "b", "identity": 0.71}],
        "h3": [],
        "antigen": [{"query": "d", "target": "e", "identity": 31.0}],
    }
    assert connected_components(["e", "c", "a", "d", "b"], hits) == [
        ["a", "b", "c"], ["d", "e"]]


def test_representative_ranking_is_deterministic_in_required_order():
    rows = [
        candidate("a", "1aaa", "AAAA", positions=3, pairs=9, resolution=1.5),
        candidate("b", "1aab", "AAAA", positions=4, pairs=2, resolution=3.0),
        candidate("c", "1aac", "AAAA", positions=4, pairs=8, resolution=2.0),
        candidate("d", "1aad", "AAAA", positions=4, pairs=8, resolution=1.8),
        candidate("e", "1aae", "AAAA", positions=4, pairs=8, resolution=1.8),
    ]
    assert select_representative(list(reversed(rows)))["instance"] == "d"


def test_donor_uses_other_component_then_length_delta_then_id():
    focal = candidate("focal", "1aaa", "AAAAAA")
    same = candidate("same", "1aab", "AAAAAA")
    donor_z = candidate("z", "1aac", "AAAAA")
    donor_a = candidate("a", "1aad", "AAAAAAA")
    components = {"focal": "C1", "same": "C1", "z": "C2", "a": "C3"}
    assert select_donor(
        focal, [same, donor_z, donor_a], components)["instance"] == "a"


def test_write_once_refuses_to_replace_existing_output(tmp_path):
    path = tmp_path / "frozen.json"
    write_json_once(path, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_once(path, {"value": 2})
    assert json.loads(path.read_text(encoding="ascii")) == {"value": 1}


def test_audit_source_has_no_model_framework_or_results_imports():
    source = (__import__("pathlib").Path(__file__).parents[1]
              / "scripts/audit_successor_v3_isolation.py").read_text(encoding="utf-8")
    assert "import torch" not in source.casefold()
    assert "disorderflow.models" not in source
    assert "results" not in [line.strip().split()[1]
                             for line in source.splitlines()
                             if line.strip().startswith("import ")]
