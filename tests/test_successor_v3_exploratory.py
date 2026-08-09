import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from scripts.build.discover_successor_v3_exploratory import discover
from scripts.check_successor_v3_future_readiness import check
from scripts.audit_successor_v3_exploratory import wsl_path
from scripts.analyze_successor_v3_contact import binary_metrics
from scripts.build.extend_successor_v3_reference_union import extend
from scripts.build.build_successor_v3_contact_dev_dataset import (
    assign_folds,
    attach_hard_mismatch,
)


ROOT = Path(__file__).resolve().parents[1]


def write_summary(path):
    fields = [
        "INSTANCE", "PDB", "SABDAB_ID", "Hchain", "Lchain", "antigen_chain",
        "antigen_type", "antigen_name", "SABDABdepo_date", "SABDABupdate_date",
        "resolution",
    ]
    rows = [
        ["pdb_1aaa-H-L", "pdb_1aaa", "a", "H", "L", "P", "PEPTIDE", "x",
         "20260101", "20260102", "2.0"],
        ["pdb_2bbb-H-L", "pdb_2bbb", "b", "H", "L", "P", "PEPTIDE", "y",
         "20260201", "20260202", "3.0"],
        ["pdb_2bbb-A-B", "pdb_2bbb", "c", "A", "B", "C", "PEPTIDE", "y",
         "20260201", "20260202", "1.5"],
        ["pdb_3ccc-H-L", "pdb_3ccc", "d", "H", "L", "P", "PROTEIN", "z",
         "20260301", "20260302", "1.0"],
    ]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def test_exploratory_discovery_is_deterministic_and_excludes_reference(tmp_path):
    summary = tmp_path / "summary.csv"
    write_summary(summary)
    reference = tmp_path / "reference.json"
    reference.write_text(json.dumps({
        "exact_exposed_pdb_ids": ["1aaa"],
        "records": [{"pdb_id": "1aaa"}],
    }), encoding="ascii")
    output = tmp_path / "discovery.json"
    report = discover(summary, reference, output, max_pdbs=2)
    assert [row["instance"] for row in report["candidates"]] == ["pdb_2bbb-A-B"]
    assert report["claim_boundary"].startswith("retrospective exploratory")
    with pytest.raises(FileExistsError):
        discover(summary, reference, output)


@pytest.mark.parametrize("count,ready", [(11, False), (12, True)])
def test_future_readiness_gate_is_exact_and_write_once(tmp_path, count, ready):
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "classification": "model_free_exact_ID_discovery",
        "candidate_entry_ids": [f"{index:04d}" for index in range(count)],
    }), encoding="ascii")
    output = tmp_path / "readiness.json"
    report = check(
        discovery, output, now=datetime(2026, 8, 7, tzinfo=timezone.utc))
    assert report["ready"] is ready
    assert report["observed_exact_new_entries"] == count
    with pytest.raises(FileExistsError):
        check(discovery, output)


def test_wsl_path_maps_windows_drive_without_changing_posix_paths():
    mapped = wsl_path("C:/Users/test/cohort.fasta")
    assert mapped == "/mnt/c/Users/test/cohort.fasta"


def test_sealed_exploratory_result_has_valid_hashes_and_claim_boundary():
    manifest = json.loads((
        ROOT / "publication/successor_v3_exploratory_result_manifest.json"
    ).read_text(encoding="ascii"))
    for artifact in manifest["artifacts"].values():
        assert hashlib.sha256((ROOT / artifact["path"]).read_bytes()).hexdigest() == (
            artifact["sha256"])
    assert manifest["cohort_flow"]["independent_components"] == 39
    assert manifest["descriptive_reference_threshold_comparison"] == {
        "classification": "descriptive reuse of frozen thresholds; not a confirmatory gate",
        "minimum_valid_components": True,
        "minimum_valid_fraction": True,
        "maximum_mean_factual_minus_mismatched_nll": True,
        "maximum_component_bootstrap_ci95_upper": True,
        "minimum_negative_component_fraction": True,
        "maximum_successor_minus_stage_a_factual_nll": True,
        "minimum_contact_auroc": False,
        "all_thresholds_passed": False,
    }
    assert manifest["future_confirmatory_status"]["ready"] is False


def test_contact_metrics_detect_score_direction_and_calibration():
    metrics = binary_metrics([0, 0, 1, 1], [-2.0, -1.0, 1.0, 2.0])
    assert metrics["auroc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["mean_positive_logit"] > metrics["mean_negative_logit"]
    assert 0.0 <= metrics["brier_score"] <= 1.0


def test_reference_extension_marks_all_viewed_ids_and_adds_sequences(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({
        "status": "frozen_successor_v3_confirmatory_reference_union",
        "exact_exposed_pdb_ids": ["1aaa"],
        "records": [{
            "id": "old", "pdb_id": "1aaa", "reference_id": "SV3R00001",
            "vh_sequence": "A", "vl_sequence": "C", "paired_cdr_sequence": "D",
            "cdr_h3_sequence": "E", "antigen_sequence": "F", "sources": ["old"],
        }],
    }), encoding="ascii")
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps({
        "status": "retrospective_exploratory_metadata_cohort",
        "candidates": [{"pdb_id": "2bbb"}],
    }), encoding="ascii")
    structural = tmp_path / "structural.json"
    structural.write_text(json.dumps({
        "classification": "model_free_structural_eligibility",
        "records": [{
            "instance": "pdb_2bbb-H-L", "pdb_id": "2bbb",
            "vh_sequence": "AA", "vl_sequence": "CC", "paired_cdr_sequence": "DD",
            "cdr_h3_sequence": "EE", "antigen_sequence": "FF",
        }],
    }), encoding="ascii")
    rcsb = tmp_path / "rcsb.json"
    rcsb.write_text(json.dumps({
        "classification": "metadata_entry_ids_only", "entry_ids": ["3CCC"],
    }), encoding="ascii")
    output = tmp_path / "v3.json"
    report = extend(base, discovery, structural, rcsb, output)
    assert report["exact_exposed_pdb_ids"] == ["1aaa", "2bbb", "3ccc"]
    assert report["counts"]["added_exploratory_sequence_records"] == 1
    assert [row["reference_id"] for row in report["records"]] == [
        "SV3R00001", "SV3R00002"]
    with pytest.raises(FileExistsError):
        extend(base, discovery, structural, rcsb, output)


def test_contact_followup_and_future_policy_are_exposure_aware():
    followup = json.loads((
        ROOT / "publication/successor_v3_contact_and_exposure_followup.json"
    ).read_text(encoding="ascii"))
    for section in (
            followup["deterministic_contact_diagnostic"],
            followup["exposure_update"]["reference_union"],
            followup["future_readiness"]["discovery"],
            followup["future_readiness"]["decision"],
            followup["future_readiness"]["policy"]):
        assert hashlib.sha256((ROOT / section["path"]).read_bytes()).hexdigest() == (
            section["sha256"])
    for path, expected in followup["fixed_scoring_defect"]["modified_code"].items():
        assert (ROOT / path).is_file()
        assert len(expected) == 64
    assert followup["future_readiness"]["exact_new_entry_ids"] == 0
    assert followup["future_readiness"]["ready"] is False
    assert followup["deterministic_contact_diagnostic"]["successor_auroc"] < 0.60

    policy = yaml.safe_load((
        ROOT / "configs/benchmarks/successor_v3_future_snapshot_policy.yml"
    ).read_text(encoding="ascii"))
    assert policy["reference_union"]["path"].endswith("reference_union_manifest_v3.json")
    assert policy["gate"]["pooling_prior_failed_snapshots"] == "forbidden"
    assert policy["current_state"] == {
        "discovery": "data/successor_v3_rcsb_snapshot_2026_08_07/discovery_after_exposure_v3.json",
        "readiness": "publication/successor_v3_future_readiness_after_exposure_2026_08_07.json",
        "exact_new_entry_ids": 0,
        "ready": False,
    }


def test_contact_development_folds_keep_components_together_and_balance_h3():
    records = {
        "a": {"cdr_h3_sequence": "AAAA"},
        "b": {"cdr_h3_sequence": "AAAA"},
        "c": {"cdr_h3_sequence": "AAAAAA"},
        "d": {"cdr_h3_sequence": "AAA"},
    }
    assignment, sizes = assign_folds([["a", "b"], ["c"], ["d"]], records, 2)
    assert assignment["a"] == assignment["b"]
    assert sum(sizes) == 17


def test_hard_antigen_mismatch_is_precomputed_and_preferred():
    from disorderflow.models.bfn_model import build_antigen_sequence_mismatch
    from disorderflow.utils.protein.constants import Fragment
    import torch

    batch = {
        "aa": torch.tensor([0, 1, 2, 3]),
        "fragment_type": torch.tensor([
            int(Fragment.Heavy), int(Fragment.Light),
            int(Fragment.Antigen), int(Fragment.Antigen)]),
    }
    attached = attach_hard_mismatch(batch, "WY")
    collated = {
        "aa": attached["aa"].unsqueeze(0),
        "hard_antigen_mismatch_aa": attached["hard_antigen_mismatch_aa"].unsqueeze(0),
        "hard_antigen_mismatch_valid": attached["hard_antigen_mismatch_valid"].unsqueeze(0),
    }
    mismatch, valid = build_antigen_sequence_mismatch(collated)
    assert mismatch[0, 2:].tolist() == [18, 19]
    assert valid.tolist() == [True]


def test_contact_v2_registry_and_future_gate_are_frozen_without_wet_results():
    registry = json.loads((
        ROOT / "publication/successor_v3_contact_v2_registry.json"
    ).read_text(encoding="ascii"))
    assert registry["future_readiness"] == {
        "eligible_exact_new_structures": 0,
        "independent_homology_components": 0,
        "required_independent_homology_components": 12,
        "ready": False,
    }
    assert registry["wet_experiment_summary"]["measurements_present"] is False
    for artifact in registry["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]

    panel = json.loads((
        ROOT / "experiments/successor_v3_contact_v2/panel.json"
    ).read_text(encoding="ascii"))
    assert panel["status"] == "planned_not_executed"
    assert len(panel["panel"]) == 16
    assert panel["power_analysis"]["independent_targets"] == 16
