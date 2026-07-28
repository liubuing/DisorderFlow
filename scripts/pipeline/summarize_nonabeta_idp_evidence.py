#!/usr/bin/env python
"""Build the final target- and antibody-family-isolated non-A-beta evidence audit."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "outputs" / "non_abeta_idp_final_audit_v1"


def load_json(path):
    with open(PROJECT_ROOT / path, encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with open(PROJECT_ROOT / path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    manifest = load_json("outputs/non_abeta_idp_ensemble_prep_v1/ensemble_manifest.json")
    poses = load_json("outputs/non_abeta_idp_pose_panels_v1/pose_panel_audit.json")
    design = load_json("outputs/non_abeta_idp_ensemble_design_v1/ensemble_design_summary.json")
    candidate_audit = load_json("outputs/non_abeta_idp_candidate_audit_v1/candidate_audit_summary.json")
    fold_summary = load_json("outputs/non_abeta_idp_fold_evidence_v1/fv_colabfold_evidence_summary.json")
    fold_rows = load_csv("outputs/non_abeta_idp_fold_evidence_v1/fv_colabfold_evidence.csv")
    repack = load_json("outputs/non_abeta_idp_fv_repack_v1/fv_repack_audit.json")
    panel = load_csv("outputs/non_abeta_idp_candidate_audit_v1/fv_fold_panel.csv")
    offstate = load_json("outputs/non_abeta_idp_offstate_contrast_v1/offstate_audit.json")
    complex_fold = load_json(
        "outputs/non_abeta_idp_complex_fold_evidence_v1/complex_fold_summary.json"
    )

    panel_index = {row["candidate_uid"]: row for row in panel}
    fold_by_target = defaultdict(lambda: {"requested": 0, "passed": 0})
    for row in fold_rows:
        source = panel_index[row["construct_id"]]
        target = source["target"]
        fold_by_target[target]["requested"] += 1
        fold_by_target[target]["passed"] += row["fold_sidecheck_status"] == "fold_sidecheck_pass"
    repack_by_target = defaultdict(lambda: {"requested": 0, "passed": 0})
    for row in repack["records"]:
        source = panel_index[row["construct_id"]]
        target = source["target"]
        repack_by_target[target]["requested"] += 1
        repack_by_target[target]["passed"] += row["status"] == "repack_pass"

    targets = {}
    for target in manifest["targets"]:
        name = target["target"]
        family = target["reference_id"]
        dev = candidate_audit["by_target"][name]
        record = {
            "antibody_family": family,
            "ensemble_source": target["source_id"],
            "ensemble_conformers": len(target["selected_conformers"]),
            "strict_pose_passed": poses["by_target"][name]["passed"],
            "strict_pose_selected": poses["by_target"][name]["selected"],
            "designed_selected": design["by_target"][name]["selected"],
            "loo_mean_delta_native": design["by_target"][name]["heldout_mean_delta_native"],
            "loo_beat_native_rate": design["by_target"][name]["heldout_beat_native_rate"],
            "developability_pass": dev["developability_pass"],
            "developability_input": dev["input"],
            "msa_fold_pass": fold_by_target[name]["passed"],
            "msa_fold_requested": fold_by_target[name]["requested"],
            "openmm_repack_pass": repack_by_target[name]["passed"],
            "openmm_repack_requested": repack_by_target[name]["requested"],
        }
        record["status"] = "pass" if (
            record["strict_pose_selected"] == 5
            and record["designed_selected"] >= 2
            and record["loo_beat_native_rate"] >= 0.8
            and record["developability_pass"] >= 2
            and record["msa_fold_pass"] == record["msa_fold_requested"] == 2
            and record["openmm_repack_pass"] == record["openmm_repack_requested"] == 2
        ) else "partial"
        targets[name] = record

    gates = {
        "ensemble_provenance": manifest["status"] == "pass",
        "strict_pose_panels": poses["status"] == "pass",
        "direct_generation_and_loo": design["status"] == "pass",
        "developability_and_family_isolation": candidate_audit["status"] == "pass",
        "msa_backed_fold": fold_summary["parsed_results"] == fold_summary["pass"] == 4,
        "openmm_repack": repack["status"] == "pass" and repack["passed"] == 4,
        "all_targets_complete": all(row["status"] == "pass" for row in targets.values()),
    }
    next_stage_gates = {
        "offstate_contrast": offstate["status"] == "pass",
        "complex_fold_computation": complex_fold["computation_status"] == "complete",
        "complex_interface_validation": complex_fold["status"] == "pass",
    }
    summary = {
        "schema_version": "nonabeta.idp_final_audit.v1",
        "status": "pass" if all(gates.values()) else "partial",
        "gates": gates,
        "next_stage_status": "pass" if all(next_stage_gates.values()) else "partial",
        "next_stage_gates": next_stage_gates,
        "offstate_by_target": offstate["by_target"],
        "complex_interface_pass_by_target": complex_fold["complex_interface_pass_by_target"],
        "by_target_and_family": targets,
        "family_isolation": candidate_audit["family_isolation"],
        "execution": {
            "fold_method": "MMseqs2 UniRef plus environmental MSA; AlphaFold2-multimer v3",
            "fv_fold_execution_backend": "CPU JAX on native Windows",
            "complex_fold_execution_backend": "WSL2 JAX CUDA GPU with bounded MSA 64:128",
            "repack_method": repack["method_boundary"],
        },
        "claim_boundary": (
            "Two within-target computational case studies from distinct reference antibody families. "
            "This is not experimental binding evidence or cross-family generalization."
        ),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "final_audit.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(OUTPUT / "final_audit_report.md", "w", encoding="utf-8") as handle:
        handle.write("# Non-A-beta IDP Final Computational Audit\n\n")
        handle.write(f"Status: `{summary['status']}`.\n\n")
        handle.write("| Target | Family | Poses | Designs | LOO beat native | Developability | MSA fold | OpenMM repack | Status |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for target, row in targets.items():
            handle.write(
                f"| {target} | {row['antibody_family']} | {row['strict_pose_selected']} | "
                f"{row['designed_selected']} | {row['loo_beat_native_rate']:.0%} | "
                f"{row['developability_pass']}/{row['developability_input']} | "
                f"{row['msa_fold_pass']}/{row['msa_fold_requested']} | "
                f"{row['openmm_repack_pass']}/{row['openmm_repack_requested']} | {row['status']} |\n"
            )
        handle.write(f"\nClaim boundary: {summary['claim_boundary']}\n")
    print(f"Non-A-beta final audit: status={summary['status']}")
    if summary["status"] != "pass":
        raise SystemExit("Final audit partial")


if __name__ == "__main__":
    main()
