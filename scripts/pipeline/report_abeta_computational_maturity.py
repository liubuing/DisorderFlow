#!/usr/bin/env python
"""Aggregate A-beta multi-conformation evidence into conservative maturity gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", default="outputs/abeta_multiconf_pose_panel_v1/pose_panel_audit.json")
    parser.add_argument("--ranking", default="outputs/abeta_multiconf_candidate_scoring_v1/multiconf_candidate_summary.json")
    parser.add_argument("--selection", default="outputs/abeta_multiconf_selection_eval_v1/selection_summary.json")
    parser.add_argument("--integrated", default="outputs/abeta_multiconf_integrated_evidence_v1/integrated_evidence_summary.json")
    parser.add_argument("--repack", default="outputs/abeta_multiconf_fv_repack_v1/fv_repack_audit.json")
    parser.add_argument("--out", default="outputs/abeta_computational_maturity_v1")
    return parser.parse_args()


def load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    args = parse_args()
    pose = load(args.pose)
    ranking = load(args.ranking)
    selection = load(args.selection)
    integrated = load(args.integrated)
    repack = load(args.repack)
    pose_pass = pose["status"] == "pass"
    gates = {
        "audited_pose_panel": {
            "status": "pass" if pose["status"] == "pass" else "fail",
            "evidence": f"{pose['passed']}/{pose['poses']} poses; fraction={pose['pass_fraction']:.2f}",
        },
        "ensemble_robust_ranking": {
            "status": (
                "pass" if pose_pass and ranking["status"] == "pass"
                else "blocked_by_pose_gate" if not pose_pass else "fail"
            ),
            "evidence": (
                f"{ranking['input_unique_candidates']} exploratory candidates; "
                f"{ranking['references']} references"
            ),
        },
        "heldout_conformer_selection": {
            "status": (
                "pass" if pose_pass and selection["status"] == "pass"
                else "blocked_by_pose_gate" if not pose_pass else "partial"
            ),
            "evidence": (
                f"n={selection['overall']['heldout_conformers']}; "
                f"mean vs native={selection['overall']['mean_minus_native']:.4f}; "
                f"mean vs random={selection['overall']['mean_minus_random']:.4f}"
            ),
        },
        "exact_full_chain_mapping": {
            "status": (
                "pass" if pose_pass and integrated["exact_sequence_matched_constructs"] > 0
                else "blocked_by_pose_gate" if not pose_pass else "fail"
            ),
            "evidence": f"{integrated['exact_sequence_matched_constructs']} exact sequence matches",
        },
        "developability_sidecheck": {
            "status": "blocked_by_pose_gate" if not pose_pass else (
                "partial" if integrated["developability_pass"] > 0 else "fail"
            ),
            "evidence": f"{integrated['developability_pass']} candidates pass",
        },
        "fold_sidecheck": {
            "status": "blocked_by_pose_gate" if not pose_pass else (
                "pass" if integrated["fold_pass"] > 0 else "blocked"
            ),
            "evidence": (
                f"{integrated['fold_pass']} pass; "
                f"{integrated.get('fold_reviewed_nonpassing', 0)} reviewed nonpassing; "
                f"{integrated.get('fold_pending', integrated['fold_pending_or_nonpassing'])} pending"
            ),
        },
        "sidechain_interface_repack": {
            "status": "pass" if repack["status"] == "pass" else "partial",
            "evidence": (
                f"{repack['passed']}/{repack['requested']} pass; "
                f"method={repack['method_boundary']}"
            ),
        },
        "experimental_binding_or_specificity": {
            "status": "not_available",
            "evidence": "No wet-lab binding or negative-state specificity measurements",
        },
    }
    core_pass = all(gates[name]["status"] == "pass" for name in (
        "audited_pose_panel", "ensemble_robust_ranking", "heldout_conformer_selection",
        "exact_full_chain_mapping",
    ))
    fold_pass = gates["fold_sidecheck"]["status"] == "pass"
    repack_pass = gates["sidechain_interface_repack"]["status"] == "pass"
    overall = (
        "pose_refinement_blocked" if not pose_pass
        else "computational_preclinical_package_ready_wetlab_pending" if core_pass and fold_pass and repack_pass
        else "computational_prioritization_ready_repack_pending" if core_pass and fold_pass
        else "computational_prioritization_ready_fold_pending" if core_pass
        else "computational_pipeline_partial"
    )
    report = {
        "schema_version": "abeta.computational_maturity.v1",
        "overall_status": overall,
        "gates": gates,
        "strongest_supported_claim": (
            "proof_of_concept_position_discovery_and_predicted_pose_ensemble_prioritization"
            if pose_pass else "proof_of_concept_position_discovery_and_n_terminal_pose_transfer_only"
        ),
        "unsupported_claims": [
            "binding affinity",
            "experimental specificity",
            "disease-state selectivity",
            "synthesis readiness",
            "generalization beyond the audited A-beta references and predicted poses",
        ],
        "next_blocking_evidence": (
            "Run local docking or side-chain/backbone repacking for failed middle-epitope poses, "
            "then rerun the zero-clash pose gate before candidate ranking and fold checks."
            if not pose_pass else
            "Obtain experimental binding, specificity, and developability measurements; "
            "computational evidence cannot replace these wet-lab gates."
            if fold_pass and repack_pass else
            "Run local side-chain rebuild/interface repack and audit the two fold-passing "
            "Fv candidates before any synthesis decision."
            if fold_pass else
            "Run candidate-level Fv/VH-VL fold prediction for exact-sequence matched or newly "
            "exported ensemble-ranked constructs, then rebuild/repack before synthesis decisions."
        ),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "computational_maturity.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(out_dir / "computational_maturity.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Computational Maturity\n\n")
        handle.write(f"Overall: `{overall}`\n\n")
        handle.write("| Gate | Status | Evidence |\n")
        handle.write("|---|---|---|\n")
        for name, gate in gates.items():
            handle.write(f"| {name} | {gate['status']} | {gate['evidence']} |\n")
        handle.write("\n## Claim Boundary\n\n")
        handle.write(f"Strongest supported claim: `{report['strongest_supported_claim']}`.\n\n")
        handle.write("This workflow does not establish binding, affinity, experimental specificity, or synthesis readiness.\n\n")
        handle.write("## Next Blocker\n\n")
        handle.write(report["next_blocking_evidence"] + "\n")
    print(f"Computational maturity: {overall}")


if __name__ == "__main__":
    main()
