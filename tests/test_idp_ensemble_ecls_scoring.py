from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts.benchmark_h3_epitope_delta import MPNN_ALPHABET
from scripts.score_idp_ensemble_candidates_ecls import pose_rows_from_manifest, write_apo

ROOT = Path(__file__).resolve().parents[1]


def test_ecls_contract_freezes_sequence_sensitive_endpoint():
    config = yaml.safe_load((
        ROOT / "configs/benchmarks/idp_ensemble_ecls_scoring_dev_v1.yml"
    ).read_text(encoding="utf-8"))
    assert config["primary"]["endpoint"] == "epitope_conditioning_gain"
    assert config["primary"]["formula"] == "apo_h3_nll_minus_complex_h3_nll"
    assert config["proteinmpnn"]["conditional_probs_only_backbone"] is True
    assert config["statistics"]["inference_unit"] == "independent_antibody_reference_component"


def test_write_apo_removes_only_antigen_atoms(tmp_path):
    complex_path = tmp_path / "complex.pdb"
    apo_path = tmp_path / "apo.pdb"
    complex_path.write_text(
        "ATOM      1  CA  ALA H   1       0.000   0.000   0.000\n"
        "ATOM      2  CA  GLY P   1       1.000   0.000   0.000\nEND\n",
        encoding="ascii",
    )
    write_apo(complex_path, apo_path)
    text = apo_path.read_text(encoding="ascii")
    assert " ALA H" in text
    assert " GLY P" not in text


def test_epitope_conditioning_gain_is_higher_when_complex_nll_is_lower():
    apo = np.full((2, len(MPNN_ALPHABET)), -2.0)
    complex_logp = np.full((2, len(MPNN_ALPHABET)), -2.0)
    complex_logp[0, MPNN_ALPHABET.index("A")] = -1.0
    from scripts.benchmark_h3_epitope_delta import sequence_nll

    gain = sequence_nll(apo, "A", [0]) - sequence_nll(complex_logp, "A", [0])
    assert gain == pytest.approx(1.0)


def test_expansion_manifest_pose_rows_require_exactly_five():
    component = {
        "component_id": "TEST",
        "pose_paths": [f"pose_{index}.pdb" for index in range(5)],
    }
    rows = pose_rows_from_manifest([component])
    assert [row["conformer"] for row in rows["TEST"]] == list(range(5))
    component["pose_paths"].pop()
    with pytest.raises(ValueError, match="exactly five"):
        pose_rows_from_manifest([component])


def test_sensitivity_manifest_allows_three_pose_rows():
    component = {
        "component_id": "SENSITIVITY",
        "sensitivity_only": True,
        "pose_paths": [f"pose_{index}.pdb" for index in range(3)],
    }
    rows = pose_rows_from_manifest([component])
    assert [row["conformer"] for row in rows["SENSITIVITY"]] == [0, 1, 2]
