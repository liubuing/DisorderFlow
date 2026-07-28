from idp_state_ensemble import build_negative_states, positive_state
from state_contact_scorer import Contact, score_sequence_on_contact_map
from state_contrast_scorer import rank_state_contrast_candidates, score_state_contrast
from statecontrast_workflow import integration_claim_level, readiness_requirements, validate_reference_count, workflow_stages
from statecontrast_real_ensembles import real_ensemble_mode_status
from contact_variable_attribution import (
    chemistry_class,
    compile_guidance_rules,
    generate_attribution_guided_variants,
    residue_matches_classes,
    volume_class,
    compare_high_low_gap,
)


def toy_contact_map():
    return {
        "paratope_sequence": "YWK",
        "peptide_sequence": "FD",
        "paratope_residues": [
            {"chain": "H", "resid": 31, "aa": "Y"},
            {"chain": "H", "resid": 32, "aa": "W"},
            {"chain": "H", "resid": 33, "aa": "K"},
        ],
        "contacts": [
            Contact(0, "H", 31, "Y", 0, "E", 1, "F", 4.0),
            Contact(1, "H", 32, "W", 0, "E", 1, "F", 4.5),
            Contact(2, "H", 33, "K", 1, "E", 2, "D", 4.0),
        ],
    }


def test_negative_states_are_scoreable():
    cmap = toy_contact_map()
    neg = build_negative_states(cmap, "toy")
    assert len(neg) == 3
    for state in neg:
        score = score_sequence_on_contact_map("YWK", state.contact_map)
        assert "state_contact_score" in score


def test_state_contrast_scores_gap():
    cmap = toy_contact_map()
    pos = [positive_state("toy_pos", cmap)]
    neg = build_negative_states(cmap, "toy")
    score = score_state_contrast("YWK", pos, neg)
    assert score["positive_score"] > 0
    assert "specificity_gap" in score


def test_rank_state_contrast_candidates():
    cmap = toy_contact_map()
    pos = [positive_state("toy_pos", cmap)]
    neg = build_negative_states(cmap, "toy")
    variants = [
        {"candidate_id": "native", "sequence": "YWK"},
        {"candidate_id": "weak", "sequence": "AAA"},
    ]
    ranked = rank_state_contrast_candidates(variants, pos, neg)
    assert ranked[0]["statecontrast_rank"] == 1
    assert len(ranked) == 2


def test_contact_variable_attribution_classes():
    assert volume_class("G") == "small"
    assert volume_class("W") == "large"
    assert chemistry_class("Y") == "aromatic"
    assert chemistry_class("K") == "positive"


def test_compare_high_low_gap_outputs_features():
    cmap = toy_contact_map()
    rows = [
        {"candidate_id": "a", "sequence": "YWK", "specificity_gap": 1.0},
        {"candidate_id": "b", "sequence": "AAA", "specificity_gap": -1.0},
    ]
    out = compare_high_low_gap(rows, cmap, top_fraction=0.5)
    assert out["n_high"] == 1
    assert out["n_low"] == 1
    assert out["feature_rows"]


def test_compile_guidance_rules_uses_only_threshold_signals():
    rows = [
        {
            "reference_pdb": "toy",
            "paratope_index": 1,
            "chain": "H",
            "resid": 31,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.25,
        },
        {
            "reference_pdb": "toy",
            "paratope_index": 2,
            "chain": "H",
            "resid": 32,
            "volume_class": "small",
            "chemistry_class": "flexible",
            "fraction_delta": -0.19,
        },
    ]
    rules = compile_guidance_rules(rows, min_abs_delta=0.20)
    assert list(rules["toy"].keys()) == [0]
    assert rules["toy"][0]["preferred"][0]["chemistry_class"] == "aromatic"


def test_attribution_guided_variants_repair_preferred_class():
    cmap = toy_contact_map()
    rules = {
        0: {
            "preferred": [{"volume_class": "large", "chemistry_class": "aromatic", "fraction_delta": 0.3}],
            "avoid": [],
        }
    }
    parents = [{"candidate_id": "bad_parent", "sequence": "AWK"}]
    variants = generate_attribution_guided_variants(cmap, parents, rules, n=4, max_mutations=2, seed=1)
    assert variants
    assert all(residue_matches_classes(v["sequence"][0], "large", "aromatic") for v in variants)


def test_statecontrast_workflow_contract_marks_cpu_and_gpu_stages():
    stages = workflow_stages()
    assert stages[0]["stage"] == "domain_intake"
    assert any(s["stage"] == "fold_sidecheck" and s["gpu_required"] for s in stages)
    req = readiness_requirements()
    assert req["cpu_only_supported_for_statecontrast_benchmark"] is True
    assert "parent_only" in req["required_controls"]


def test_integration_claim_level_is_conservative():
    assert integration_claim_level(2, False, False) == "proof_of_concept_position_discovery"
    assert integration_claim_level(8, True, True) == "validated_domain_workflow"


def test_verified_reference_count_excludes_unreviewed_entries():
    refs = [
        {"pdb": "GOOD", "reliability": "reliable", "provenance_verified": True},
        {"pdb": "UNREVIEWED", "reliability": "reliable"},
        {"pdb": "BAD", "reliability": "questionable", "provenance_verified": True},
    ]
    result = validate_reference_count(refs, minimum=2, require_verified=True)
    assert result["n_declared_references"] == 3
    assert result["n_references"] == 1
    assert result["status"] == "fail"


def test_verified_reference_count_accepts_reviewed_entries():
    refs = [
        {"pdb": str(index), "reliability": "reliable", "provenance_verified": True}
        for index in range(4)
    ]
    result = validate_reference_count(refs, minimum=4, require_verified=True)
    assert result["n_references"] == 4
    assert result["status"] == "pass"


def test_real_ensemble_status_reports_missing_and_ready_entries(tmp_path):
    ready = tmp_path / "ready.pdb"
    ready.write_text(
        "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  \n",
        encoding="utf-8",
    )
    cfg = {
        "negative_states": {
            "mode": "structure_ensemble",
            "ensembles": [
                {"name": "ready", "pdb": "ready.pdb", "chain": "A", "state_type": "fibril"},
                {"name": "missing", "pdb": "missing.pdb", "chain": "A", "state_type": "oligomer"},
            ],
        }
    }
    status = real_ensemble_mode_status(cfg, tmp_path)
    assert status["ready"] == 1
    assert status["missing"] == 1
    assert status["status"] == "blocked_missing_or_invalid_ensemble_pdbs"
    assert status["entries"][0]["contact_scoring_status"] == "blocked_without_pose"
