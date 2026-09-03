import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "modules" / "honest_confidence.py"
SPEC = importlib.util.spec_from_file_location("honest_confidence", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_design(af2_pae=None, pae=None, ppl=10.0, entropy=1.0):
    return {
        "sequence": "AAAA",
        "af2_interface_pae": af2_pae,
        "pae": pae,
        "plddt": 0.8,
        "iptm": 0.5,
        "ppl": ppl,
        "entropy": entropy,
    }


def test_deployment_pae_prefers_af2_interface_pae():
    assert MODULE.deployment_pae(make_design(af2_pae=5.0, pae=20.0)) == 5.0


def test_deployment_pae_falls_back_to_bfn_pae():
    assert MODULE.deployment_pae(make_design(af2_pae=None, pae=20.0)) == 20.0


def test_deployment_pae_returns_none_when_absent():
    assert MODULE.deployment_pae(make_design(af2_pae=None, pae=None)) is None


def test_rank_designs_honest_sorts_by_pae_ascending():
    designs = [{
        "designs": [
            make_design(af2_pae=30.0),
            make_design(af2_pae=5.0),
            make_design(af2_pae=15.0),
        ]
    }]
    ranked = MODULE.rank_designs_honest(designs)
    assert [r["interface_pae"] for r in ranked] == [5.0, 15.0, 30.0]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def test_rank_designs_honest_flags_abstentions():
    ranked = MODULE.rank_designs_honest([{"designs": [make_design(af2_pae=5.0)]}])
    entry = ranked[0]
    assert entry["confidence_policy"] == "pae_only"
    assert entry["confidence_abstentions"] == ["plddt", "iptm"]
    assert entry["plddt_reported_not_deployed"] == 0.8
    assert entry["iptm_reported_not_deployed"] == 0.5


def test_rank_designs_honest_missing_pae_sorts_last():
    designs = [{
        "designs": [
            make_design(af2_pae=None, pae=None),
            make_design(af2_pae=8.0),
        ]
    }]
    ranked = MODULE.rank_designs_honest(designs)
    assert ranked[0]["interface_pae"] == 8.0
    assert ranked[1]["interface_pae"] is None
