import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / "pipeline" / f"{name}.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variable_region_boundary_uses_j_motif():
    mod = load_script("analyze_shortlist_variable_regions")
    seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSNYAMSWVRQAPGKGLEWVSA" \
          "INASGTRTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGKGYVRYFDV" \
          "WGQGTLVTVSSASTKGPSVF"
    cut = mod.find_variable_region(seq, "heavy")
    assert cut["boundary_status"] == "j_motif_found"
    assert cut["variable_seq"].endswith("WGQGTLVTVSS")
    assert cut["trailing_seq"].startswith("ASTK")


def test_nglyco_flag_localizes_to_variable_region():
    mod = load_script("analyze_shortlist_variable_regions")
    variable = "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAINASGTRTYYADSV"
    trailing = "ASTKGPSVF"
    region = mod.flag_region("n_glycosylation_motif", variable + trailing, variable, trailing)
    assert region == "variable"


def test_v2_status_blocks_variable_region_review():
    mod = load_script("build_shortlist_v2_side_evidence")
    dev = {"developability_status": "developability_pass", "introduced_or_candidate_specific_flags": ""}
    var = {"variable_region_status": "variable_region_review"}
    assert mod.evidence_status(dev, var) == "variable_region_developability_review"


def test_v2_status_allows_clean_candidate_for_fold_queue():
    mod = load_script("build_shortlist_v2_side_evidence")
    dev = {"developability_status": "developability_pass", "introduced_or_candidate_specific_flags": ""}
    var = {"variable_region_status": "inherited_flags_outside_variable_region"}
    assert mod.evidence_status(dev, var) == "ready_for_fv_fold_sidecheck"


def test_export_prefers_contacting_duplicate_antibody_chains():
    mod = load_script("export_full_chain_constructs")
    cmap = {"paratope_residues": [{"chain": "A", "resid": 52}, {"chain": "B", "resid": 93}]}
    roles = {"H": "H", "L": "L", "A": "H", "B": "L"}
    assert mod.contact_chain_roles(cmap, roles) == {"H": "A", "L": "B"}


def test_5csz_rescue_detects_a54_ngt_context():
    mod = load_script("rescue_5csz_nglyco_candidates")
    seq = "SYAAHANRYGKGYVRTIYNMPI"
    idx_by_residue = {("A", 54): 6}
    assert mod.has_nglyco_motif(seq, idx_by_residue)


def test_statecontrast_variable_deltas_compare_high_and_low_gap():
    mod = load_script("summarize_statecontrast_variables")
    rows = [
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "group": "high_gap",
            "fraction": "0.8",
        },
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "group": "low_gap",
            "fraction": "0.3",
        },
    ]
    deltas = mod.compute_deltas(rows)
    assert deltas == [
        {
            "reference_pdb": "toy",
            "paratope_index": "1",
            "chain": "H",
            "resid": "31",
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "high_fraction": 0.8,
            "low_fraction": 0.3,
            "fraction_delta": 0.5,
            "direction": "high_gap_enriched",
        }
    ]
    assert mod.aggregate_by_volume(deltas) == [
        {
            "reference_pdb": "toy",
            "volume_class": "large",
            "mean_delta": 0.5,
            "max_abs_delta": 0.5,
            "n_positions": 1,
        }
    ]


def test_overfit_control_shuffles_single_rule_to_new_position():
    mod = load_script("run_statecontrast_ab_overfit_controls")
    rng = __import__("random").Random(1)
    rules = {2: {"preferred": [{"volume_class": "large", "chemistry_class": "aromatic", "fraction_delta": 0.2}], "avoid": []}}
    shuffled = mod.shuffled_position_rules(rules, sequence_len=5, rng=rng)
    assert list(shuffled.keys()) != [2]
    assert list(shuffled.values())[0]["preferred"][0]["chemistry_class"] == "aromatic"


def test_replicate_aggregation_requires_beating_both_controls():
    mod = load_script("run_statecontrast_ab_overfit_replicates")
    rows = []
    for seed, guided, parent, random_rule, shuffled in [
        (1, 0.5, 0.2, 0.4, 0.3),
        (2, 0.5, 0.2, 0.6, 0.3),
        (3, 0.5, 0.2, 0.4, 0.6),
    ]:
        rows.extend([
            {"seed": seed, "reference_pdb": "toy", "arm": "guided_v3_heldout_eval", "top10_mean_gap_delta": guided},
            {"seed": seed, "reference_pdb": "toy", "arm": "top_parent_only_heldout_eval", "top10_mean_gap_delta": parent},
            {"seed": seed, "reference_pdb": "toy", "arm": "random_rule_v3_heldout_eval", "top10_mean_gap_delta": random_rule},
            {"seed": seed, "reference_pdb": "toy", "arm": "shuffled_rule_v3_heldout_eval", "top10_mean_gap_delta": shuffled},
            {"seed": seed, "reference_pdb": "toy", "arm": "v2_discovery_pool_heldout_eval", "top10_mean_gap_delta": 0.1},
        ])
    summary = mod.summarize_metric("toy", rows, "top10_mean_gap_delta")
    assert summary["guided_beats_parent_random_and_shuffled_rate"] == 0.3333
    assert summary["verdict"] == "mixed_or_overfit_risk"


def test_stable_attribution_requires_recurrent_same_direction_signal():
    mod = load_script("run_statecontrast_ab_stable_attribution")
    rows = [
        {
            "seed": 1,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.25,
        },
        {
            "seed": 2,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": 0.21,
        },
        {
            "seed": 3,
            "reference_pdb": "toy",
            "paratope_index": 3,
            "chain": "H",
            "resid": 52,
            "volume_class": "large",
            "chemistry_class": "aromatic",
            "fraction_delta": -0.30,
        },
    ]
    stable = mod.stable_signal_rows(rows, "toy", min_recurrence=2)
    assert len(stable) == 1
    assert stable[0]["direction"] == "high_gap_enriched"
    assert stable[0]["recurrence"] == 2


def test_mutation_effects_compare_present_vs_absent_candidates():
    mod = load_script("analyze_statecontrast_mutation_effects")
    rows = [
        {"reference_pdb": "toy", "candidate_id": "a", "mutations": "A1Y", "specificity_gap_delta": "0.4"},
        {"reference_pdb": "toy", "candidate_id": "b", "mutations": "A1Y", "specificity_gap_delta": "0.2"},
        {"reference_pdb": "toy", "candidate_id": "c", "mutations": "", "specificity_gap_delta": "0.0"},
        {"reference_pdb": "toy", "candidate_id": "d", "mutations": "G2A", "specificity_gap_delta": "0.0"},
    ]
    mutation_rows, position_rows = mod.mutation_effects(rows, min_support=2)
    top = mutation_rows[0]
    assert top["effect_key"] == "A:1:Y"
    assert top["present_mean"] == 0.3
    assert top["absent_mean"] == 0.0
    assert position_rows[0]["paratope_index"] == 1


def test_effect_guided_v4_recurrent_mutation_rules_and_generation():
    mod = load_script("run_statecontrast_ab_effect_guided_v4")
    rows = []
    for seed in (1, 2):
        rows.extend([
            {"seed": seed, "candidate_id": f"hit_{seed}_1", "mutations": "A1Y", "specificity_gap_delta": "0.4"},
            {"seed": seed, "candidate_id": f"hit_{seed}_2", "mutations": "A1Y", "specificity_gap_delta": "0.3"},
            {"seed": seed, "candidate_id": f"miss_{seed}_1", "mutations": "", "specificity_gap_delta": "0.0"},
            {"seed": seed, "candidate_id": f"miss_{seed}_2", "mutations": "G2A", "specificity_gap_delta": "0.0"},
        ])
    rules = mod.stable_effect_rules(rows, "toy", min_support=2, min_effect=0.1, min_recurrence=2)
    assert rules[0]["rule_type"] == "mutation"
    assert rules[0]["paratope_index"] == 1
    assert rules[0]["new_aa"] == "Y"
    cmap = {"paratope_sequence": "AG", "contacts": [], "paratope_residues": []}
    variants = mod.generate_effect_guided_variants(cmap, [{"candidate_id": "p", "sequence": "AG"}], rules, n=3, max_mutations=2, seed=1, reference_pdb="toy")
    assert variants
    assert all(v["sequence"][0] == "Y" for v in variants)


def test_effect_guided_v4_caps_active_rules_to_mutation_budget():
    mod = load_script("run_statecontrast_ab_effect_guided_v4")
    cmap = {"paratope_sequence": "AGST", "contacts": [], "paratope_residues": []}
    rules = [
        {"rule_type": "mutation", "paratope_index": 1, "new_aa": "Y"},
        {"rule_type": "mutation", "paratope_index": 2, "new_aa": "Y"},
        {"rule_type": "mutation", "paratope_index": 3, "new_aa": "Y"},
    ]
    variants = mod.generate_effect_guided_variants(cmap, [{"candidate_id": "p", "sequence": "AGST"}], rules, n=3, max_mutations=2, seed=1, reference_pdb="toy")
    assert variants
    assert all(int(v["n_mutations"]) <= 2 for v in variants)


def test_final_report_summarizes_position_rules_sorted_by_reference():
    mod = load_script("build_statecontrast_ab_final_report")
    rows = [
        {"reference_pdb": "5CSZ", "paratope_index": "12"},
        {"reference_pdb": "5CSZ", "paratope_index": "9"},
        {"reference_pdb": "4HIX", "paratope_index": "19"},
        {"reference_pdb": "4HIX", "paratope_index": "3"},
    ]
    assert mod.summarize_rules(rows) == {"5CSZ": ["9", "12"], "4HIX": ["3", "19"]}


def test_gpu_readiness_distinguishes_driver_from_python_runtime():
    mod = load_script("check_gpu_readiness")
    checks = [
        {"name": "nvidia_smi", "required": False, "status": "pass"},
        {"name": "torch_cuda", "required": False, "status": "pass", "stdout_tail": "2.12.0+cpu\nFalse\n0"},
        {"name": "colabfold_executable", "required": False, "status": "pass"},
        {"name": "project_file:x", "required": True, "status": "pass"},
    ]
    summary = mod.summarize_checks(checks)
    assert summary["overall_status"] == "gpu_driver_ready_python_runtime_not_ready"
    assert summary["gpu_driver_detected"] is True
    assert summary["python_gpu_runtime_detected"] is False


def test_domain_workflow_metadata_keeps_abeta_claim_conservative():
    mod = load_script("run_statecontrast_domain_workflow")
    cfg = mod.load_config(str(ROOT / "configs" / "statecontrast" / "abeta_position_effect_v1.yml"))
    metadata = mod.workflow_metadata(cfg)
    assert metadata["domain"] == "idp_antibody"
    assert metadata["reference_check"]["n_declared_references"] == 2
    assert metadata["reference_check"]["n_references"] == 0
    assert metadata["reference_check"]["status"] == "fail"
    assert metadata["claim_level"] == "exploratory_only"
    assert "parent_only" in metadata["controls"]
