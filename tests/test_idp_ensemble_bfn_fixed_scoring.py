from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.score_idp_ensemble_candidates_bfn import score_value

ROOT = Path(__file__).resolve().parents[1]


def test_bfn_fixed_scoring_contract_freezes_primary_endpoint():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_bfn_fixed_scoring_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["primary"]["endpoint"] == "bfn_fixed_candidate_iptm"
    assert config["bfn"]["fixed_t"] == 0.5
    assert config["bfn"]["antigen_chain"] == "P"
    assert config["statistics"]["inference_unit"] == "independent_antibody_reference_component"
    assert "not an ensemble-aware BFN generator" in config["claim_boundary"]


def test_score_value_uses_generated_region_for_local_metrics():
    import torch

    batch = {"generate_flag": torch.tensor([[False, True, True]])}
    output = {
        "iptm": torch.tensor([0.4]),
        "plddt": torch.tensor([[10.0, 20.0, 40.0]]),
        "pae": torch.tensor([[[9.0, 9.0, 9.0], [9.0, 1.0, 3.0], [9.0, 5.0, 7.0]]]),
        "state_compatibility": torch.tensor([0.2]),
    }
    row = score_value(output, batch)
    assert row["iptm"] == pytest.approx(0.4)
    assert row["generated_region_plddt"] == pytest.approx(30.0)
    assert row["generated_region_pae"] == pytest.approx(np.mean([1.0, 3.0, 5.0, 7.0]))
