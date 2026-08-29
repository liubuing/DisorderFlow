import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "scripts" / "binder_confidence_v1.py"
    spec = importlib.util.spec_from_file_location("binder_confidence_v1", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_empty_measurements_abstain_without_probabilities():
    mod = load_script()
    config = {
        "labels": {
            "expression_pass": {"required_batches": 2, "minimum_yield_mg_per_l": 1.0,
                                "minimum_sec_monomer_fraction": 0.8},
            "binding_screen_hit": {"target_analyte": "target", "minimum_technical_replicates": 3,
                                   "maximum_replicate_cv": 0.2,
                                   "minimum_signal_ratio_vs_external_negative": 3.0},
            "specificity_pass": {"control_analytes": ["scramble", "BSA"],
                                 "maximum_control_fraction_of_target": 0.2},
        },
        "assay_validity": {"external_negative_blind_id": "NEG", "mock_blind_id": "MOCK",
                           "native_control_id": "native"},
        "training_gates": {"minimum_qc_candidates": 50, "minimum_positive_binding_hits": 20,
                           "minimum_scaffolds": 3, "minimum_targets": 3,
                           "panel_only_model_forbidden": True},
    }
    features = [{"construct_id": "candidate", "scaffold_id": "3D6", "target_id": "abeta"}]
    result = mod.analyze_measurements(config, features, Path("missing"), Path("missing"), Path("missing"))
    assert result["status"] == "abstain_no_experimental_labels"
    assert result["complete_candidate_labels"] == 0
    assert result["training_readiness"]["eligible"] is False


def test_expression_requires_two_complete_passing_batches():
    mod = load_script()
    config = {"labels": {"expression_pass": {"required_batches": 2,
        "minimum_yield_mg_per_l": 1.0, "minimum_sec_monomer_fraction": 0.8}}}
    rows = [
        {"construct_id": "x", "expression_batch": "1", "yield_mg_per_l": "2",
         "sec_monomer_fraction": "0.9", "qc_pass": "True"},
        {"construct_id": "x", "expression_batch": "2", "yield_mg_per_l": "2",
         "sec_monomer_fraction": "0.9", "qc_pass": "True"},
    ]
    labels, audit = mod.expression_labels(config, rows)
    assert labels["x"] is True
    assert audit["x"]["complete"] is True


def test_training_gate_rejects_single_scaffold_panel():
    mod = load_script()
    config = {"training_gates": {"minimum_qc_candidates": 1,
        "minimum_positive_binding_hits": 1, "minimum_scaffolds": 3,
        "minimum_targets": 3, "panel_only_model_forbidden": True}}
    rows = [{"expression_pass": True, "binding_screen_hit": True,
             "scaffold_id": "3D6", "target_id": "abeta"}]
    result = mod.training_readiness(config, rows)
    assert result["eligible"] is False
    assert result["checks"]["minimum_scaffolds"] is False
    assert result["checks"]["minimum_targets"] is False


def test_complete_valid_assay_labels_panel_but_still_rejects_training(tmp_path):
    mod = load_script()
    config = {
        "labels": {
            "expression_pass": {"required_batches": 2, "minimum_yield_mg_per_l": 1.0,
                                "minimum_sec_monomer_fraction": 0.8},
            "binding_screen_hit": {"target_analyte": "target", "minimum_technical_replicates": 3,
                                   "maximum_replicate_cv": 0.2,
                                   "minimum_signal_ratio_vs_external_negative": 3.0},
            "specificity_pass": {"control_analytes": ["scramble", "BSA"],
                                 "maximum_control_fraction_of_target": 0.2},
        },
        "assay_validity": {"external_negative_blind_id": "NEG", "mock_blind_id": "MOCK",
                           "native_control_id": "native"},
        "training_gates": {"minimum_qc_candidates": 1, "minimum_positive_binding_hits": 1,
                           "minimum_scaffolds": 3, "minimum_targets": 3,
                           "panel_only_model_forbidden": True},
    }
    features = [{"construct_id": "candidate", "scaffold_id": "3D6", "target_id": "abeta"}]
    expression = tmp_path / "expression.csv"
    expression.write_text(
        "construct_id,expression_batch,yield_mg_per_l,sec_monomer_fraction,qc_pass\n"
        "candidate,1,2,0.9,True\n"
        "candidate,2,2,0.9,True\n"
        "native,1,2,0.9,True\n"
        "native,2,2,0.9,True\n", encoding="ascii")
    key = tmp_path / "key.csv"
    key.write_text(
        "blind_sample_id,construct_id\nCAND,candidate\nNATIVE,native\nNEG,negative\nMOCK,mock\n",
        encoding="ascii")
    bli = tmp_path / "bli.csv"
    rows = ["blind_sample_id,analyte,blank_subtracted_response,fit_quality,operator_exclusion_reason"]
    means = {
        ("CAND", "target"): 10.0, ("CAND", "scramble"): 1.0, ("CAND", "BSA"): 1.0,
        ("NATIVE", "target"): 8.0, ("NATIVE", "scramble"): 1.0, ("NATIVE", "BSA"): 1.0,
        ("NEG", "target"): 1.0, ("NEG", "scramble"): 1.0, ("NEG", "BSA"): 1.0,
        ("MOCK", "target"): 0.5, ("MOCK", "scramble"): 0.5, ("MOCK", "BSA"): 0.5,
    }
    for (blind, analyte), mean in means.items():
        for value in (mean * 0.95, mean, mean * 1.05):
            rows.append(f"{blind},{analyte},{value},pass,")
    bli.write_text("\n".join(rows) + "\n", encoding="ascii")
    result = mod.analyze_measurements(config, features, expression, bli, key)
    assert result["status"] == "panel_labels_complete_model_training_still_gated"
    assert result["assay_validity"]["native_target_signal_pass"] is True
    assert result["assay_validity"]["native_expression_pass"] is True
    assert result["labeled_rows"][0]["overall_screen_pass"] is True
    assert result["training_readiness"]["eligible"] is False
