import hashlib
import json
import pickle
from pathlib import Path

import pytest

from scripts.benchmark_h3_candidate_reranking import (
    build_esmif_partial_sequence,
    validate_esmif_sample,
)

ROOT = Path(__file__).resolve().parents[1]


def test_followup_evidence_manifest_hashes_match_files():
    manifest = json.loads(
        (ROOT / "publication/followup_evidence_manifest.json").read_text())
    for item in manifest["files"]:
        path = ROOT / item["path"]
        assert path.is_file(), item["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_t21_current_mechanism_is_terminal_negative():
    decision = json.loads(
        (ROOT / "publication/t21_mechanism_terminal_decision.json").read_text())
    contrast = decision["primary_contrast"]
    assert decision["mechanism_supported"] is False
    assert decision["terminal_for_current_mechanism"] is True
    assert decision["rerun_existing_final_permitted"] is False
    assert contrast["mean"] < 0
    assert contrast["ci95"][1] < 0


def test_esmif_partial_sequence_masks_only_design_positions():
    partial = build_esmif_partial_sequence("ACDEFG", [1, 4], 9)
    assert partial == ["A", "<mask>", "D", "E", "<mask>", "G",
                       "<pad>", "<pad>", "<pad>"]


def test_esmif_rejects_changes_to_fixed_heavy_positions():
    validate_esmif_sample("AYDEFQ", "ACDEFG", [1, 5])
    with pytest.raises(ValueError, match="fixed heavy-chain positions"):
        validate_esmif_sample("AYXEFQ", "ACDEFG", [1, 5])


def test_mutation_panel_does_not_promote_censored_values_to_kd():
    panel = json.loads(
        (ROOT / "publication/experimental_mutations_v1.json").read_text())
    assert panel["records"]
    for record in panel["records"]:
        assert record["extraction_tier"].startswith("exact_")
        if "censoring" in record:
            assert "value" not in record
            assert record["endpoint"] != "KD_M"


def test_multiscaffold_v2_holdout_is_frozen_from_independent_components():
    holdout = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/holdout_manifest.json"
    ).read_text())
    feasibility = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/feasibility_audit.json"
    ).read_text())
    representative_hash = hashlib.sha256(
        "\n".join(sorted(holdout["representative_ids"])).encode()).hexdigest()
    proposed = feasibility["schemes"][feasibility["proposed_scheme"]]
    assert holdout["classification"] == (
        "prospective_internal_holdout; not external confirmation")
    assert holdout["n_all_axis_components"] == 20
    assert holdout["n_representatives"] == 20
    assert representative_hash == holdout["representative_ids_sha256"]
    assert set(holdout["representative_ids"]) <= set(proposed["independent_ids"])


def test_multiscaffold_v2_training_lineage_is_isolated():
    contract = json.loads((
        ROOT / "results/ablation/multiscaffold_v2_training_contract.json"
    ).read_text())
    lookup_audit = json.loads((
        ROOT / "data/multiscaffold_confirmatory_v2/disorder_lookup_audit.json"
    ).read_text())
    lookup = pickle.loads((
        ROOT / "data/disorder_supervision/train_afdb_multiscaffold_v2.pkl"
    ).read_bytes())
    assert contract["status"] == "two_stage_training_contract_verified"
    assert contract["stage_a"]["initialization"] == "random"
    assert contract["stage_a"]["freeze_backbone"] is False
    assert contract["stage_a"]["torch_load_guard_smoke_passed"] is True
    assert contract["stage_a"]["one_step_forward_backward_smoke_passed"] is True
    assert contract["stage_b"]["only_permitted_initializer"] == "stage_a_checkpoint"
    assert len(lookup["profiles"]) == 827
    assert lookup_audit["n_excluded_records"] == 1
    assert not set(lookup_audit["excluded_ids"]) & set(lookup["profiles"])


def test_multiscaffold_v2_structural_evaluation_is_complete_but_primary_gate_failed():
    af2 = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/af2/results.json"
    ).read_text())
    prodigy = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/prodigy/results.json"
    ).read_text())
    analysis = json.loads((
        ROOT / "results/multiscaffold_confirmatory_v2/analysis.json"
    ).read_text())
    assert af2["summary"] == {
        "expected_prediction_slots": 960,
        "recorded_prediction_slots": 960,
        "successes": 960,
        "failures": 0,
    }
    assert prodigy["summary"]["recorded_slots"] == 960
    assert prodigy["summary"]["successes"] == 959
    assert prodigy["summary"]["failures"] == 1
    assert analysis["status"] == "primary_gate_failed"
    assert analysis["primary"]["passed"] is False
    assert analysis["primary"]["valid_components"] == 7
    assert analysis["primary"]["mean_positive_delta"] < 0
    assert analysis["primary"]["component_bootstrap_ci95"][1] < 0
    assert not any(analysis["primary"]["gate_results"].values())
