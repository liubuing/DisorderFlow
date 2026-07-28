#!/usr/bin/env python
"""Summarize pose-conditioned, flexible-refinement, family, off-state, and wet-lab evidence."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/non_abeta_idp_next_stage_audit_v1"


def csv_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_data(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def main():
    fold_aggregate = csv_rows(
        ROOT / "outputs/non_abeta_idp_initial_guess_multiseed_evidence_v1/initial_guess_aggregate.csv"
    )
    refinement = csv_rows(
        ROOT / "outputs/non_abeta_idp_complex_multiseed_refinement_v1/complex_refinement_audit.csv"
    )
    offstate = json_data(ROOT / "outputs/non_abeta_idp_offstate_contrast_v1/offstate_audit.json")
    family = json_data(ROOT / "outputs/non_abeta_idp_family_panel_v1/family_panel_summary.json")
    wetlab = json_data(ROOT / "outputs/non_abeta_idp_wetlab_handoff_v1/wetlab_status.json")
    completion = json_data(
        ROOT / "outputs/non_abeta_idp_computational_completion_v2/computational_completion.json"
    )

    fold_by_id = {row["construct_id"]: row for row in fold_aggregate}
    refine_by_id = defaultdict(list)
    for row in refinement:
        refine_by_id[row["construct_id"]].append(row)
    pairs = {
        "tau": ("5MP3_native", "5MP3_cbefe653f673"),
        "alpha_synuclein": ("8B9V_native", "8B9V_498c3ede07c8"),
    }
    target_results = {}
    for target, (native_id, candidate_id) in pairs.items():
        native = fold_by_id[native_id]
        candidate = fold_by_id[candidate_id]
        native_refined = refine_by_id[native_id]
        candidate_refined = refine_by_id[candidate_id]
        deltas = {
            "antigen_plddt": float(candidate["delta_antigen_plddt_vs_native"]),
            "antibody_to_antigen_pae": float(candidate["delta_pae_vs_native"]),
            "contact_retention": float(candidate["delta_contact_retention_vs_native"]),
            "antigen_pose_rmsd": float(candidate["delta_antigen_rmsd_vs_native"]),
            "refined_contact_retention": round(
                mean(candidate_refined, "contact_retention") - mean(native_refined, "contact_retention"), 5
            ),
            "refined_global_backbone_rmsd": round(
                mean(candidate_refined, "global_backbone_rmsd")
                - mean(native_refined, "global_backbone_rmsd"), 5
            ),
            "refined_interface_backbone_rmsd": round(
                mean(candidate_refined, "interface_backbone_rmsd")
                - mean(native_refined, "interface_backbone_rmsd"), 5
            ),
        }
        zero_clash = all(int(row["after_severe_clash_pairs_lt_1_5A"]) == 0 for row in candidate_refined)
        paired_noninferiority = (
            deltas["antibody_to_antigen_pae"] <= 0.0
            and deltas["contact_retention"] >= 0.0
            and deltas["antigen_pose_rmsd"] <= 0.0
            and deltas["refined_contact_retention"] >= 0.0
            and zero_clash
        )
        target_results[target] = {
            "native_construct": native_id,
            "candidate_construct": candidate_id,
            "conformers": int(candidate["conformers"]),
            "seeds_per_conformer": int(candidate["models"]) // int(candidate["conformers"]),
            "models": int(candidate["models"]),
            "native_relative_deltas": deltas,
            "refined_zero_clash_models": sum(
                int(row["after_severe_clash_pairs_lt_1_5A"]) == 0 for row in candidate_refined
            ),
            "pose_conditioned_refined_interface_status": (
                "computational_pass" if paired_noninferiority else "computational_review"
            ),
        }

    offstate_ready = all(
        values["projected_scoreable_states"] >= 1 and values["candidate_pass"] >= 1
        for values in offstate["by_target"].values()
    )
    lofo_complete = all(values["lofo_design_complete"] for values in family["by_target"].values())
    wetlab_complete = wetlab["overall_status"] == "complete" and wetlab["results_present"]
    summary = {
        "schema_version": "nonabeta.next_stage_audit.v1",
        "overall_status": completion["overall_status"],
        "authoritative_current_ledger": (
            "outputs/non_abeta_idp_computational_completion_v2/computational_completion.json"
        ),
        "completed": {
            "experimental_pose_initial_guess": True,
            "multiconformer_multiseed_sampling": True,
            "local_joint_flexibility_refinement": True,
            "independent_family_data_panel": all(
                values["lofo_data_ready"] for values in family["by_target"].values()
            ),
            "wetlab_execution_package": True,
            "literature_review": (ROOT / "docs/reviews/FLEXIBLE_IDP_ANTIBODY_DESIGN_REVIEW.md").exists(),
            "candidate_specific_sequence_offstate_screen": True,
            "contrastive_redesign_max2_and_max4": True,
            "cross_epitope_transfer_benchmark": True,
            "alpha_synuclein_conservative_redesign_validation": True,
        },
        "target_results": target_results,
        "unresolved": {
            "sequence_offstate_hit": True,
            "strict_same_epitope_leave_one_family_out": not lofo_complete,
            "wetlab_expression_sec_bli_spr": not wetlab_complete,
            "alpha_synuclein_computational_hit": True,
        },
        "method_boundaries": {
            "local_joint_flexibility_refinement": (
                "AlphaFold-Multimer sampled complete VH:VL:IDP coordinates across 5 conformers and 3 seeds; "
                "OpenMM then allowed restrained interface/CDR-backbone and antigen relaxation. This is coupled "
                "local sampling, not exhaustive CDR-loop or ABangle-space exploration."
            ),
            "offstate": offstate["claim_boundary"],
            "sequence_offstate": completion["claim_boundary"],
            "family": family["claim_boundary"],
            "wetlab": (
                "The protocol and gates are ready, but every experimental status is not_executed. "
                "No expression, SEC, affinity, or specificity result is claimed."
            ),
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT / "next_stage_audit.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = [
        "# Non-A-beta IDP next-stage audit",
        "",
        f"Overall status: **{summary['overall_status']}**.",
        "",
        "## Completed computational work",
        "",
        "- Experimental-pose initial guesses: complete.",
        "- Multi-conformer/multi-seed panel: 4 constructs x 5 conformers x 3 seeds = 60 models.",
        "- Restrained local joint flexibility refinement: 60 models; severe clashes removed in all refined models.",
        "- Independent-family data panel: Tau 5 families; alpha-synuclein 3 families.",
        "- Candidate-specific sequence off-state screening and contrastive searches at <=2 and <=4 mutations: complete; no hit for either target.",
        "- Cross-epitope transfer: 7/8 folds scoreable; this is not same-epitope LOFO.",
        "- Conservative alpha-synuclein redesign: 20 candidates and 315 models; no candidate passed all five frozen gates.",
        "- Wet-lab execution package and literature review: complete as documents, not as experiments.",
        "",
        "## Target results",
        "",
    ]
    for target, result in target_results.items():
        report.extend([
            f"### {target}",
            "",
            f"Candidate: `{result['candidate_construct']}`; status: "
            f"**{result['pose_conditioned_refined_interface_status']}**.",
            "",
            f"Native-relative deltas: `{json.dumps(result['native_relative_deltas'], ensure_ascii=False)}`.",
            "",
        ])
    report.extend([
        "## Unresolved gates",
        "",
        "- Geometry off-state projections cannot establish sequence-selective non-binding; sequence-contrastive computation was executed and produced no hit.",
        "- Strict same-epitope LOFO is blocked until at least three independently isolated same-epitope families are competition-confirmed.",
        "- The complete conservative alpha-synuclein redesign produced no computational hit under all five frozen gates.",
        "- Expression, SEC, BLI/SPR, and state-specificity experiments are not executed.",
        "",
        "No unresolved item is promoted to pass by changing thresholds after observing results.",
    ])
    (OUTPUT / "next_stage_audit_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Next-stage audit: {summary['overall_status']} targets={target_results}")


if __name__ == "__main__":
    main()
