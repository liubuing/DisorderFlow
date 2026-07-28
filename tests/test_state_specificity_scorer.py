"""Tests for state-specific IDP antibody design MVP."""

from state_specificity_scorer import (
    DesignSpec,
    complementarity_score,
    generate_paratope_candidates,
    load_preset,
    rank_candidates,
    sequence_complexity,
)


def test_complementarity_rewards_amyloid_aromatic_charge_pattern():
    epitope = "KLVFFAED"
    designed = "DYWYSRNYQHG"
    weak = "GGGGGGGGGGG"
    assert complementarity_score(designed, epitope) > complementarity_score(weak, epitope)


def test_rank_candidates_prefers_positive_over_negative_specificity():
    spec = DesignSpec(
        target="abeta42",
        epitope_id="core_16_24",
        epitope_seq="KLVFFAED",
        positive_state="oligomer_core",
        negative_epitopes=["DAEFRHDSGY", "GGGGGGGG"],
    )
    ranked = rank_candidates([
        {"candidate_id": "good", "cdr_h3": "DYWYSRNYQHG"},
        {"candidate_id": "bad", "cdr_h3": "GGGGGGGGGGG"},
    ], spec)
    assert ranked[0]["candidate_id"] == "good"
    assert ranked[0]["specificity_gap"] > ranked[1]["specificity_gap"]


def test_generator_is_deterministic_and_outputs_rankable_candidates():
    spec = load_preset("abeta_core")
    a = generate_paratope_candidates(spec, n=12, length=11, seed=7)
    b = generate_paratope_candidates(spec, n=12, length=11, seed=7)
    assert [x["cdr_h3"] for x in a] == [x["cdr_h3"] for x in b]
    ranked = rank_candidates(a, spec)
    assert len(ranked) == 12
    assert ranked[0]["rank"] == 1
    assert "state_specific_score" in ranked[0]


def test_sequence_complexity_penalizes_homopolymers():
    assert sequence_complexity("DYWYSRNYQHG") > sequence_complexity("TTTTTTTTTTT")
