import json
from pathlib import Path

import pytest
import yaml

from scripts.analyze_idp_ensemble_multimethod_dev import (
    component_method_means,
    exact_sign_flip_p,
    paired_summary,
)

ROOT = Path(__file__).resolve().parents[1]


def test_development_contract_does_not_claim_confirmation():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["classification"] == "retrospective_development_benchmark_not_confirmation"
    assert config["gates"]["minimum_valid_components"] == 12
    assert config["primary"]["inference_unit"] == "independent_antibody_reference_component"
    assert config["gates"]["require_complete_matched_arms"] == [
        "single_conformer_bfn", "proteinmpnn", "esm_if"
    ]
    assert "no BFN superiority" in config["claim_boundary"]


def test_manifest_has_six_unique_exposed_components_and_three_sources():
    manifest = json.loads((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1_components.json"
    ).read_text(encoding="utf-8"))
    components = manifest["components"]
    assert len(components) == 6
    assert len({row["component_id"] for row in components}) == 6
    assert len({row["ensemble_source"] for row in components}) == 3
    assert all(row["prior_outcome_exposure"] for row in components)
    assert all(row["pose_provenance"] == "template_transferred_computational_pose" for row in components)


def test_component_aggregation_occurs_before_paired_inference():
    folds = [
        {"component_id": "a", "fold": "0", "method": "ensemble", "score": 3.0},
        {"component_id": "a", "fold": "1", "method": "ensemble", "score": 1.0},
        {"component_id": "a", "fold": "0", "method": "single", "score": 1.0},
        {"component_id": "a", "fold": "1", "method": "single", "score": 1.0},
        {"component_id": "b", "fold": "0", "method": "ensemble", "score": 2.0},
        {"component_id": "b", "fold": "0", "method": "single", "score": 3.0},
    ]
    scores = component_method_means(folds)
    summary = paired_summary(scores, "ensemble", "single", ["a", "b"], 1000, 17)
    assert summary["valid_components"] == 2
    assert summary["mean_delta"] == pytest.approx(0.0)
    assert summary["positive_component_fraction"] == 0.5


def test_missing_baseline_component_is_not_imputed():
    scores = {("a", "ensemble"): 2.0, ("a", "single"): 1.0, ("b", "ensemble"): 4.0}
    summary = paired_summary(scores, "ensemble", "single", ["a", "b"], 100, 3)
    assert summary["valid_components"] == 1
    assert summary["rows"][1]["valid"] is False
    assert summary["rows"][1]["delta"] is None


def test_exact_sign_flip_is_component_level_and_two_sided():
    assert exact_sign_flip_p([1.0, 1.0, 1.0]) == 0.25
