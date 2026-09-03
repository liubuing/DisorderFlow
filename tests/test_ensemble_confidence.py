import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "modules" / "ensemble_confidence.py"
SPEC = importlib.util.spec_from_file_location("ensemble_confidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_robust_p25():
    summary = MODULE.robust_interface_pae([10.0, 20.0, 30.0, 40.0], "p25")
    assert summary["robust"] == pytest.approx(17.5)
    assert summary["min"] == 10.0
    assert summary["max"] == 40.0


def test_robust_min_and_median():
    values = [10.0, 20.0, 30.0]
    assert MODULE.robust_interface_pae(values, "min")["robust"] == 10.0
    assert MODULE.robust_interface_pae(values, "median")["robust"] == 20.0


def test_robust_handles_missing():
    summary = MODULE.robust_interface_pae([10.0, None, 30.0], "p25")
    assert summary["robust"] == pytest.approx(15.0)
    assert summary["n_missing"] == 1
    assert summary["n_conformations"] == 3


def test_robust_all_missing():
    summary = MODULE.robust_interface_pae([None, None])
    assert summary["robust"] is None


def test_rank_designs_ensemble_prefers_lower_robust_pae():
    designs = [
        {"sequence": "A", "per_conf_pae": [30.0, 40.0, 50.0]},
        {"sequence": "B", "per_conf_pae": [5.0, 6.0, 7.0]},
        {"sequence": "C", "per_conf_pae": [15.0, 16.0, 17.0]},
    ]
    ranked = MODULE.rank_designs_ensemble(designs)
    assert [r["sequence"] for r in ranked] == ["B", "C", "A"]


def test_rank_designs_ensemble_flags_abstentions():
    ranked = MODULE.rank_designs_ensemble(
        [{"sequence": "A", "per_conf_pae": [10.0, 20.0]}])
    entry = ranked[0]
    assert entry["confidence_policy"] == "pae_only"
    assert entry["confidence_abstentions"] == ["plddt", "iptm"]


def test_rank_designs_ensemble_missing_sorts_last():
    ranked = MODULE.rank_designs_ensemble([
        {"sequence": "missing", "per_conf_pae": [None]},
        {"sequence": "good", "per_conf_pae": [8.0]},
    ])
    assert ranked[0]["sequence"] == "good"
    assert ranked[1]["sequence"] == "missing"
