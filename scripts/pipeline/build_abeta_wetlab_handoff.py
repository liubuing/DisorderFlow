#!/usr/bin/env python
"""Build a conservative wet-lab handoff for computation-complete A-beta Fv candidates."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrated", default="outputs/abeta_multiconf_integrated_evidence_v1/integrated_candidate_evidence.csv")
    parser.add_argument("--out", default="outputs/abeta_wetlab_handoff_v1")
    return parser.parse_args()


def load_candidates(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["integrated_evidence_status"]
            == "computational_side_evidence_complete_wetlab_pending"
        ]


def main():
    args = parse_args()
    candidates = load_candidates(args.integrated)
    if not candidates:
        raise SystemExit("No computation-complete wet-lab candidates found")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "wetlab_priority", "candidate_uid", "construct_id", "reference_pdb",
        "candidate_id", "heavy_sequence", "light_sequence", "multiconf_rank_within_reference",
        "multiconf_robust_score", "native_delta_p25", "developability_status",
        "fold_sidecheck_status", "mean_plddt", "iptm", "repack_status",
        "repacked_pdb", "release_status",
    ]
    manifest = []
    for priority, row in enumerate(sorted(
        candidates,
        key=lambda item: (float(item["native_delta_p25"]), float(item["multiconf_robust_score"])),
        reverse=True,
    ), 1):
        manifest.append({
            **row,
            "wetlab_priority": priority,
            "release_status": "wetlab_candidate_not_synthesis_or_binding_validated",
        })
    with open(out_dir / "wetlab_candidate_manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest)
    with open(out_dir / "wetlab_candidate_sequences.fasta", "w", encoding="ascii") as handle:
        for row in manifest:
            handle.write(f">{row['construct_id']}|{row['candidate_uid']}|heavy\n{row['heavy_sequence']}\n")
            handle.write(f">{row['construct_id']}|{row['candidate_uid']}|light\n{row['light_sequence']}\n")

    gates = {
        "schema_version": "abeta.wetlab_handoff.v1",
        "status": "ready_for_protocol_review_and_execution",
        "candidate_count": len(manifest),
        "replication": {
            "independent_protein_preparations": 2,
            "technical_replicates_per_preparation": 3,
            "analysis": "Report estimates and confidence intervals; do not promote on a single replicate.",
        },
        "required_controls": [
            "parental reference antibody",
            "matched non-binding isotype or irrelevant Fv",
            "buffer-only blank",
            "scrambled A-beta peptide control",
            "assay-specific positive control",
        ],
        "ordered_gates": [
            {
                "gate": "expression_and_identity",
                "measurements": ["expression yield", "intact mass", "reducing mass or chain identity"],
                "provisional_acceptance": [
                    "detectable correctly assembled product in both independent preparations",
                    "measured mass consistent with the intended construct within assay tolerance",
                ],
                "failure_effect": "Stop binding interpretation for failed material.",
            },
            {
                "gate": "solution_developability",
                "measurements": ["SEC monomer fraction", "DLS polydispersity", "thermal transition", "visible precipitation"],
                "provisional_acceptance": [
                    "SEC monomer fraction >= 90% after purification",
                    "no visible precipitation at the assay concentration",
                    "single dominant thermal transition with Tm >= 55 C",
                ],
                "failure_effect": "Classify as developability failure; do not infer lack of binding from aggregated material.",
            },
            {
                "gate": "direct_binding",
                "measurements": ["SPR or BLI concentration series", "equilibrium response", "kinetic fit where justified"],
                "target_panel": ["the crystallographic N-terminal A-beta epitope", "full-length A-beta42"],
                "provisional_acceptance": [
                    "concentration-dependent, saturable signal above isotype and blank controls",
                    "replicate estimates are mutually consistent",
                    "a fitted KD is reported only when the selected model passes residual and identifiability checks",
                ],
                "failure_effect": "No binding claim; report assay detection limit and raw response.",
            },
            {
                "gate": "state_specificity",
                "measurements": ["matched-format binding to monomer", "soluble oligomer", "fibril polymorph panel"],
                "provisional_acceptance": [
                    "preferred-state response or affinity separated from every off-state by >= 10-fold",
                    "the direction of separation reproduces across two independent preparations",
                    "parent and isotype controls behave as expected",
                ],
                "failure_effect": "Binding may be claimed if direct binding passed, but state-selectivity may not be claimed.",
            },
            {
                "gate": "orthogonal_confirmation",
                "measurements": ["ELISA or dot blot in a non-kinetic format", "competition with cognate peptide", "optional microscopy or pull-down for fibril binding"],
                "provisional_acceptance": [
                    "at least one orthogonal assay agrees with the primary binding result",
                    "cognate competition reduces signal while scrambled control does not",
                ],
                "failure_effect": "Keep result provisional and investigate assay-format artifacts.",
            },
        ],
        "claim_upgrade_rules": {
            "binding": "Requires direct_binding and orthogonal_confirmation gates.",
            "affinity": "Requires an identifiable, quality-controlled quantitative binding fit.",
            "state_specificity": "Requires direct_binding, state_specificity, and orthogonal_confirmation gates.",
            "synthesis_ready": "Not defined by this package; requires manufacturing, formulation, and organizational review.",
        },
        "current_claim_boundary": (
            "Candidates are computationally prioritized and structurally side-checked. "
            "No experimental binding, affinity, state specificity, or synthesis-readiness claim is supported."
        ),
    }
    with open(out_dir / "wetlab_gate_spec.json", "w", encoding="utf-8") as handle:
        json.dump(gates, handle, indent=2)
    with open(out_dir / "wetlab_handoff_report.md", "w", encoding="utf-8") as handle:
        handle.write("# A-beta Wet-Lab Handoff\n\n")
        handle.write(f"Candidates: {len(manifest)}; status: `{gates['status']}`.\n\n")
        handle.write("## Candidate Order\n\n")
        handle.write("| Priority | Construct | UID | Ensemble score | Native delta P25 | Fold | Repack |\n")
        handle.write("|---:|---|---|---:|---:|---|---|\n")
        for row in manifest:
            handle.write(
                f"| {row['wetlab_priority']} | {row['construct_id']} | {row['candidate_uid']} | "
                f"{row['multiconf_robust_score']} | {row['native_delta_p25']} | "
                f"{row['fold_sidecheck_status']} | {row['repack_status']} |\n"
            )
        handle.write("\n## Execution Order\n\n")
        for index, gate in enumerate(gates["ordered_gates"], 1):
            handle.write(f"{index}. `{gate['gate']}`: {', '.join(gate['measurements'])}.\n")
        handle.write("\n## Claim Boundary\n\n")
        handle.write(gates["current_claim_boundary"] + "\n")
    print(f"Wet-lab handoff: {len(manifest)} candidates; output={out_dir}")


if __name__ == "__main__":
    main()
