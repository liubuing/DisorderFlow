"""Tests for contact-aware state specificity scorer."""

from state_contact_scorer import (
    Contact,
    abeta_fragment_info,
    benchmark_native_vs_decoys,
    conservative_mutant,
    generate_contact_guided_variants,
    benchmark_native_vs_scrambled,
    hotspot_breaking_mutant,
    score_contact_guided_variants,
    score_sequence_on_contact_map,
    select_abeta_like_chain,
)


def _toy_contact_map():
    return {
        "paratope_sequence": "DYW",
        "peptide_sequence": "KFF",
        "contacts": [
            Contact(0, "H", 31, "D", 0, "P", 1, "K", 3.8),
            Contact(1, "H", 32, "Y", 1, "P", 2, "F", 4.0),
            Contact(2, "H", 33, "W", 2, "P", 3, "F", 4.2),
        ],
    }


def test_native_scores_above_scrambled_hotspot_layout():
    cmap = _toy_contact_map()
    native = score_sequence_on_contact_map("DYW", cmap)
    scrambled = score_sequence_on_contact_map("WYD", cmap)
    assert native["state_contact_score"] > scrambled["state_contact_score"]
    assert native["hotspot_score"] > scrambled["hotspot_score"]


def test_benchmark_reports_native_percentile():
    bench = benchmark_native_vs_scrambled(_toy_contact_map(), n_scramble=4, seed=1)
    assert bench["summary"]["native_score"] >= bench["summary"]["scramble_mean_score"]
    assert 0.0 <= bench["summary"]["native_percentile"] <= 1.0


def test_abeta_fragment_info_detects_contiguous_fragment():
    info = abeta_fragment_info("DAEFRHDSGY")
    assert info["is_abeta_like"] is True
    assert info["abeta_start"] == 1
    assert info["abeta_end"] == 10


def test_select_abeta_like_chain_prefers_abeta_over_short_non_abeta():
    chains = {
        "X": [{"aa": a} for a in "EDQIDW"],
        "P": [{"aa": a} for a in "DAEFRHDSGY"],
        "H": [{"aa": "A"} for _ in range(120)],
    }
    selected = select_abeta_like_chain(chains)
    assert selected["chosen_chain"] == "P"


def test_hotspot_breaking_mutant_lowers_toy_score():
    cmap = _toy_contact_map()
    broken = hotspot_breaking_mutant(cmap)
    assert score_sequence_on_contact_map("DYW", cmap)["state_contact_score"] > score_sequence_on_contact_map(broken, cmap)["state_contact_score"]


def test_benchmark_native_vs_decoys_includes_multiple_decoy_types():
    bench = benchmark_native_vs_decoys(_toy_contact_map(), n_scramble=4, seed=2)
    types = {d["decoy_type"] for d in bench["decoys"]}
    assert {"scrambled", "alanine", "glycine", "hotspot_breaking", "conservative"}.issubset(types)
    assert conservative_mutant("DYW") == "EFY"


def test_contact_guided_variants_are_rankable():
    cmap = _toy_contact_map()
    variants = generate_contact_guided_variants(cmap, n=5, max_mutations=2, seed=3)
    ranked = score_contact_guided_variants(cmap, variants)
    assert len(ranked) == 5
    assert ranked[0]["rank"] == 1
    assert "candidate_score" in ranked[0]
