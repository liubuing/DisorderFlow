import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_candidate_interface_calibration_extension.py"
SPEC = importlib.util.spec_from_file_location("ext_pair_evidence", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_pair_outcomes_counts_only_reliable_deltas():
    rows = [
        {"target_iptm": 0.80, "noise_iptm": 0.01},
        {"target_iptm": 0.70, "noise_iptm": 0.01},
        {"target_iptm": 0.805, "noise_iptm": 0.01},
    ]
    outcomes = MODULE.pair_outcomes(rows, "iptm")
    assert len(outcomes) == 2


def test_pair_statistics_gate_fields():
    rows = [
        {"scaffold_family": "S1", "target_iptm": 0.8, "noise_iptm": 0.01},
        {"scaffold_family": "S1", "target_iptm": 0.7, "noise_iptm": 0.01},
        {"scaffold_family": "S1", "target_iptm": 0.6, "noise_iptm": 0.01},
        {"scaffold_family": "S2", "target_iptm": 0.5, "noise_iptm": 0.01},
        {"scaffold_family": "S2", "target_iptm": 0.3, "noise_iptm": 0.01},
    ]
    stats = MODULE.pair_statistics(rows, "iptm")
    assert stats["reliable_pairs"] == 4
    assert stats["contributing_scaffolds"] == 2
    assert stats["max_scaffold_pair_fraction"] == pytest.approx(3 / 4)


def test_aggregate_conditions_means_replicates():
    rows = [
        {"construct_id": "C1", "scaffold_family": "S1",
         "target_iptm": 0.8, "noise_iptm": 0.02, "target_plddt": 0.9, "noise_plddt": 0.03,
         "target_pae": 0.1, "noise_pae": 0.01},
        {"construct_id": "C1", "scaffold_family": "S1",
         "target_iptm": 0.6, "noise_iptm": 0.02, "target_plddt": 0.7, "noise_plddt": 0.03,
         "target_pae": 0.3, "noise_pae": 0.01},
    ]
    aggregated = MODULE.aggregate_conditions(rows)
    assert len(aggregated) == 1
    assert aggregated[0]["target_iptm"] == pytest.approx(0.7)
    assert aggregated[0]["noise_iptm"] == pytest.approx(0.02)
