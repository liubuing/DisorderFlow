import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.score_idp_ensemble_matched_candidates import (
    candidate_pools,
    leave_one_out,
    robust_value,
)

ROOT = Path(__file__).resolve().parents[1]


def test_scoring_contract_uses_component_level_inference():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_matched_scoring_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["scoring"]["selection_protocol"] == "leave_one_conformer_out"
    assert config["statistics"]["inference_unit"] == "independent_antibody_reference_component"
    assert config["development_gates"]["minimum_valid_components_per_generator"] == 6
    assert config["scoring"]["minimum_sidechain_contacts_per_conformer"] == 1
    assert "not an ensemble-aware BFN generator" in config["claim_boundary"]


def test_candidate_pools_deduplicate_within_component_method():
    raw = {"slots": [
        {"status": "success", "component_id": "c", "method": "m", "candidates": [
            {"sequence": "AAAA"}, {"sequence": "AAAA"}, {"sequence": "CCCC"}
        ]}
    ]}
    assert candidate_pools(raw, ["m"])[("c", "m")] == ["AAAA", "CCCC"]


def test_robust_value_penalizes_a_low_outlier():
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    assert robust_value([1, 1, 1, 1], aggregation) > robust_value([0, 1, 1, 1], aggregation)


def test_leave_one_out_never_uses_heldout_for_ensemble_selection():
    aggregation = {
        "p25_weight": 0.45,
        "minimum_weight": 0.25,
        "mean_weight": 0.20,
        "standard_deviation_penalty": 0.10,
    }
    matrix = np.asarray([
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 100.0],
    ])
    rows = leave_one_out("c", "m", ["AAAA", "CCCC"], matrix, np.zeros(3), aggregation)
    assert rows[2]["ensemble_candidate"] == "AAAA"
    assert rows[2]["ensemble_score"] == pytest.approx(1.0)


def test_generation_artifact_is_complete_before_scoring():
    raw = json.loads((
        ROOT / "reviewer_outputs/idp_ensemble_matched_generation_dev_v1/raw_generation.json"
    ).read_text(encoding="utf-8"))
    assert raw["status"] == "complete"
    assert raw["completed_slots"] == 54
    assert raw["failures"] == []
