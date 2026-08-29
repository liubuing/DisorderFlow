import json
from pathlib import Path

import pytest
import yaml

from scripts.generate_idp_ensemble_matched_baselines import (
    canonicalize_component,
    validate_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def test_generation_contract_freezes_matched_budget_and_methods():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_matched_generation_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["generation"]["seeds"] == [4101, 4111, 4121]
    assert config["generation"]["candidates_per_seed"] == 8
    assert config["generation"]["mutable_region"] == "heavy_chain_h3_only"
    assert all(config["generation"][method]["enabled"] for method in (
        "proteinmpnn", "single_conformer_bfn", "esm_if"
    ))


def test_all_components_canonicalize_and_match_native_h3(tmp_path):
    manifest = json.loads((
        ROOT / "configs/benchmarks/idp_ensemble_multimethod_dev_v1_components.json"
    ).read_text(encoding="utf-8"))
    for component in manifest["components"]:
        output = tmp_path / component["component_id"] / "canonical.pdb"
        result = canonicalize_component(component, output)
        assert set(result["chains"]) == {"H", "L", "P"}
        assert result["native_h3"] == component["native_h3"]
        assert output.is_file()


def test_vhh_component_canonicalizes_as_heavy_and_antigen_only(tmp_path):
    expansion = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    component = next(
        row for row in expansion["components"] if row["component_id"] == "ASYN_2X6M"
    )
    component = {**component, "reference_path": component["reference_pdb"]}
    output = tmp_path / "vhh.pdb"
    result = canonicalize_component(component, output)
    assert set(result["chains"]) == {"H", "P"}
    assert result["native_h3"] == component["native_h3"]


def test_candidate_slot_requires_exact_count_and_valid_length():
    component = {"native_h3": "ABCDE"}
    valid = [{"sequence": "ACDEF"} for _ in range(8)]
    validate_rows(valid, component, 8)
    with pytest.raises(ValueError, match="Expected 8"):
        validate_rows(valid[:-1], component, 8)
    with pytest.raises(ValueError, match="Invalid generated H3"):
        validate_rows([{"sequence": "ACDXF"} for _ in range(8)], component, 8)
