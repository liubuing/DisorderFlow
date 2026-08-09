import pytest

from scripts.ablation_disorder_conditioning import AA_LETTERS
from scripts.benchmark_vs_mpnn import sequence_recovery
from scripts.t21_statistics import aggregate


def test_ablation_uses_canonical_amino_acid_order():
    assert AA_LETTERS == "ACDEFGHIKLMNPQRSTVWY"


def test_sequence_recovery_rejects_length_mismatch():
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        sequence_recovery("ACD", "ACDE")


def test_t21_reports_overall_and_post_tier_validity_separately():
    results = [
        {
            "id": "a",
            "torsion_hit_tier": True,
            "valid_t2_1": True,
            "metrics": {
                "mean_held_out_contact_recovery": 0.5,
                "mean_rmsd_recovery_angstrom": 0.1,
                "random_restraints_held_out_contact": 0.7,
            },
        },
        {"id": "b", "torsion_hit_tier": True, "valid_t2_1": False},
        {"id": "c", "torsion_hit_tier": False, "valid_t2_1": False},
    ]
    config = {
        "statistics": {"bootstrap_trials": 100, "bootstrap_seed": 1},
        "development_gates": {
            "minimum_valid_structure_fraction": 0.3,
            "minimum_fraction_structures_positive_held_out_recovery": 0.5,
        },
    }
    observed = aggregate(results, config, {"a": "cluster-1"})
    assert observed["overall_valid_structure_fraction"] == pytest.approx(1 / 3)
    assert observed["post_tier_qc_fraction"] == pytest.approx(1 / 2)
    contrast = observed["paired_arm_contrasts"]["supplied_minus_random_restraints"]
    assert contrast["mean"] == pytest.approx(-0.2)
    assert contrast["inference_unit"] == "official_antigen_cluster"
