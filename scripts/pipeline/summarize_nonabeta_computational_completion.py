#!/usr/bin/env python
"""Final ledger for computationally executable work and experiment-only blocks."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(path):
    return json.load(open(ROOT / path, encoding="utf-8"))


def main():
    sequence_screen = load("outputs/non_abeta_idp_sequence_offstate_v1/sequence_offstate_summary.json")
    contrast2 = load("outputs/non_abeta_idp_sequence_contrastive_design_v1/contrastive_design_summary.json")
    contrast4 = load("outputs/non_abeta_idp_sequence_contrastive_design_v1_max4/contrastive_design_summary.json")
    cross = load("outputs/non_abeta_idp_cross_epitope_transfer_v1/cross_epitope_summary.json")
    alpha = load("outputs/alpha_synuclein_redesign_validation_v2/redesign_validation_summary.json")
    wetlab = load("outputs/non_abeta_idp_wetlab_handoff_v1/wetlab_status.json")
    ledger = {
        "schema_version": "nonabeta.computational_completion.v2",
        "overall_status": "computational_work_complete_experiments_pending",
        "computational": {
            "geometry_offstate_panel": {
                "status": "complete_geometry_control",
                "claim": "positive pose geometrically incompatible with audited off-state projection",
            },
            "sequence_offstate_screen": {
                "status": sequence_screen["status"],
                "passing_by_target": {target: values["passing"] for target, values in sequence_screen["by_target"].items()},
                "interpretation": "executed_no_hit",
            },
            "contrastive_design_max2": {
                "status": contrast2["status"], "by_target": contrast2["by_target"],
            },
            "contrastive_design_max4": {
                "status": contrast4["status"], "by_target": contrast4["by_target"],
            },
            "cross_epitope_transfer": {
                "status": cross["status"], "scoreable_folds": cross["scoreable_folds"],
                "families_total": cross["families_total"], "mean_coverage": cross["mean_coverage"],
                "mean_direction_accuracy": cross["mean_direction_accuracy"],
            },
            "same_epitope_lofo": cross["same_epitope_lofo"],
            "alpha_synuclein_redesign": {
                "status": alpha["status"], "candidates": alpha["candidates"], "models": alpha["models"],
                "refinement_pass_models": alpha["refinement_pass_models"],
                "computational_hits": alpha["computational_hits"],
            },
        },
        "experiment_only": {
            "expression_sec_dsf": wetlab["overall_status"],
            "bli_spr_binding_affinity": wetlab["overall_status"],
            "state_specificity": wetlab["overall_status"],
            "ptm_specificity": wetlab["overall_status"],
            "same_epitope_competition_bins": wetlab["overall_status"],
            "strict_same_epitope_lofo_release": "blocked_pending_family_acquisition_and_competition",
        },
        "claim_boundary": (
            "No computational no-hit proves nonbinding. No geometry block proves experimental specificity. "
            "No alpha-synuclein design is promoted after failing frozen gates."
        ),
    }
    out = ROOT / "outputs/non_abeta_idp_computational_completion_v2"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(ledger, open(out / "computational_completion.json", "w", encoding="utf-8"), indent=2)
    report = [
        "# Non-A-beta computational completion v2", "",
        "Status: **computational work complete; experiments pending**.", "",
        "## Completed", "",
        "- Geometry off-state controls retained without nonbinding claims.",
        "- Candidate-specific topology/sequence off-state screen completed: zero strict hits for both targets.",
        "- Explicit contrastive design completed at <=2 and <=4 mutations: no hit for either target.",
        f"- Cross-epitope transfer: {cross['scoreable_folds']}/{cross['families_total']} scoreable folds, "
        f"mean coverage {cross['mean_coverage']:.3f}, direction accuracy {cross['mean_direction_accuracy']:.3f}.",
        f"- Alpha-synuclein conservative redesign: {alpha['candidates']} candidates, {alpha['models']} models, "
        f"{alpha['refinement_pass_models']} refinement passes, zero candidates passing all frozen gates.", "",
        "## Experiment-only", "",
        "- Expression, purification, SEC, DSF, BLI/SPR, PTM/state specificity, and orthogonal confirmation.",
        "- Acquisition and competition-bin confirmation of at least three independent same-epitope families.", "",
        "All experimental statuses remain `not_executed`.",
    ]
    (out / "computational_completion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Computational completion: {ledger['overall_status']}")


if __name__ == "__main__":
    main()
