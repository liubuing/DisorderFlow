from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_expansion_freezes_six_distinct_antibody_lineages_before_scoring():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    components = config["components"]
    assert len(components) == 6
    assert len({row["component_id"] for row in components}) == 6
    assert {row["component_id"] for row in components} == {
        "ABETA_3U0T", "ABETA_7BXV", "TAU_6LRA",
        "ASYN_2X6M", "ASYN_6CT7", "TAU_6PXR",
    }
    assert config["status"] == "frozen_after_6pxr_input_topology_fix_before_replica_generation"
    assert config["statistics"]["primary_total_components_after_merge"] == 12
    assert config["statistics"]["sensitivity_excluding_synthetic_component"] == 11


def test_only_one_expansion_component_uses_synthetic_local_ensemble():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    synthetic = [
        row for row in config["components"]
        if row["ensemble"]["type"] == "synthetic_local_openmm"
    ]
    assert [row["component_id"] for row in synthetic] == ["TAU_6PXR"]
    assert len(synthetic[0]["ensemble"]["seeds"]) == 8
    assert synthetic[0]["ensemble"]["sampling_mode"] == (
        "peptide_only_then_fixed_antibody_template_transfer"
    )


def test_vhh_component_explicitly_has_no_light_chain():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_expansion_6ref_v1.yml"
    ).read_text(encoding="utf-8"))
    vhh = next(row for row in config["components"] if row["component_id"] == "ASYN_2X6M")
    assert vhh["light_chain"] is None
    assert vhh["epitope"] == "DYEPEA"
    assert vhh["nominal_epitope"] == "GYQDYEPEA"
